# cleaning_providers/hybrid_cleaner.py
"""
策略：
1. 第一阶段：快速规则清洗（向量化，覆盖80%问题）
2. 第二阶段：智能过滤（跳过干净列）
3. 第三阶段：批量LLM清洗（只处理复杂列）
"""
from __future__ import annotations
import asyncio
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import CleaningProvider, CleaningReport
from .fast_cleaner import FastCleaningProvider
from .batch_llm_cleaner import BatchLLMCleaningProvider


class HybridCleaningProvider(CleaningProvider):
    """
    混合清洗器 - 规则引擎 + 智能LLM
    阶段：
    1. 快速规则清洗（80%问题）
    2. 智能列过滤（减少LLM调用）
    3. 批量LLM清洗（复杂问题）

    """

    name = "Hybrid-Cleaning"

    def __init__(
            self,
            llm_client=None,
            fast_first: bool = True,
            llm_batch_size: int = 8,
            max_sample_rows: int = 15,
            skip_clean_columns: bool = True
    ):
        """
        Args:
            llm_client: LLM客户端（可选）
            fast_first: 是否先执行快速规则清洗
            llm_batch_size: LLM批量处理列数（推荐6-10）
            max_sample_rows: LLM采样行数（推荐15-20）
            skip_clean_columns: 是否跳过已干净的列
        """
        super().__init__()
        self.llm_client = llm_client
        self.fast_first = fast_first
        self.llm_batch_size = llm_batch_size
        self.max_sample_rows = max_sample_rows
        self.skip_clean_columns = skip_clean_columns

        # 初始化子清洗器
        self.fast_cleaner = FastCleaningProvider(aggressive=False)
        self.llm_cleaner = None
        if llm_client:
            self.llm_cleaner = BatchLLMCleaningProvider(
                llm_client=llm_client,
                batch_size=llm_batch_size,
                max_sample_rows=max_sample_rows,
                max_concurrent_batches=3
            )

    async def clean(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """混合清洗（快速规则 + 智能LLM）"""
        if df.height == 0:
            return df, self._create_report([])

        print(f"[Hybrid-Cleaning] Processing {len(df.columns)} columns...")
        all_changes = []
        current_df = df.clone()

        # ====================================================================
        # 阶段1: 快速规则清洗（向量化，覆盖80%问题）
        # ====================================================================
        if self.fast_first:
            print(f"[Hybrid] Stage 1/3: Fast rule-based cleaning...")
            current_df, fast_report = await self.fast_cleaner.clean(
                current_df,
                primary_keys,
                rules
            )
            all_changes.extend(fast_report.changes)

            if fast_report.cleaned_cells > 0:
                print(f"[Hybrid] Stage 1 ✓ Fixed {fast_report.cleaned_cells} cells (rules)")

        # ====================================================================
        # 阶段2: 智能列过滤（跳过已干净的列）
        # ====================================================================
        if self.skip_clean_columns:
            print(f"[Hybrid] Stage 2/3: Filtering clean columns...")
            complex_cols = self._identify_complex_columns(
                current_df,
                primary_keys,
                fast_report.cleaned_columns if self.fast_first else []
            )
            print(f"[Hybrid] Stage 2 ✓ {len(complex_cols)} columns need LLM cleaning")
        else:
            complex_cols = [c for c in current_df.columns if c not in primary_keys]

        # ====================================================================
        # 阶段3: 批量LLM清洗（只处理复杂列）
        # ====================================================================
        if complex_cols and self.llm_cleaner:
            print(f"[Hybrid] Stage 3/3: Batch LLM cleaning ({len(complex_cols)} cols)...")

            # 创建子表（只包含复杂列 + 主键）
            subset_cols = list(set(primary_keys + complex_cols))
            subset_df = current_df.select(subset_cols)

            # LLM清洗
            cleaned_subset, llm_report = await self.llm_cleaner.clean(
                subset_df,
                primary_keys,
                rules
            )

            # 合并回原表
            for col in complex_cols:
                if col in cleaned_subset.columns:
                    current_df = current_df.with_columns([
                        cleaned_subset[col].alias(col)
                    ])

            all_changes.extend(llm_report.changes)

            if llm_report.cleaned_cells > 0:
                print(f"[Hybrid] Stage 3 ✓ Fixed {llm_report.cleaned_cells} cells (LLM)")

        # ====================================================================
        # 生成综合报告
        # ====================================================================
        report = self._create_report(all_changes)

        if report.cleaned_cells > 0:
            print(f"[Hybrid-Cleaning] ✓ Total: {report.cleaned_cells} cells in {report.cleaned_rows} rows")
            print(f"  - Type fixes: {report.type_fixes}")
            print(f"  - Format fixes: {report.format_fixes}")
            print(f"  - Outlier fixes: {report.outlier_fixes}")

        return current_df, report

    def _identify_complex_columns(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            already_cleaned: List[str]
    ) -> List[str]:
        """
        智能识别需要LLM清洗的复杂列

        跳过条件：
        1. 主键
        2. 已被规则清洗的列
        3. 唯一值比例极高（> 0.95，可能是ID）
        4. 唯一值极少（< 10，枚举类）
        5. 完美的数值列（无异常值）
        6. 完美的日期列（格式统一）
        7. 空值率极高（> 0.8）

        保留：
        - 类型混乱的列
        - 格式不统一的列
        - 有明显异常值的列
        """
        complex_cols = []

        for col in df.columns:
            # 跳过条件1: 主键
            if col in primary_keys:
                continue

            # 跳过条件2: 已清洗
            if col in already_cleaned:
                continue

            # 基本统计
            s = df[col]
            null_rate = s.null_count() / max(1, len(s))
            unique_rate = s.n_unique() / max(1, len(s))

            # 跳过条件3: 唯一值比例极高（可能是ID）
            if unique_rate > 0.95:
                continue

            # 跳过条件4: 唯一值极少（枚举类）
            if s.n_unique() < 10 and null_rate < 0.3:
                continue

            # 跳过条件7: 空值率极高
            if null_rate > 0.8:
                continue

            # 跳过条件5: 完美的数值列
            if df.schema[col].is_numeric():
                try:
                    # 检查异常值比例
                    is_outlier = self._detect_outliers_iqr(df, col)
                    outlier_rate = is_outlier.sum() / max(1, len(df))
                    if outlier_rate < 0.02:  # 异常值 < 2%
                        continue
                except:
                    pass

            # 跳过条件6: 完美的日期列
            if df.schema[col] in (pl.Date, pl.Datetime):
                # 检查格式一致性
                try:
                    date_str = s.cast(pl.Utf8)
                    # 如果所有非空值都符合标准格式，跳过
                    valid_format = date_str.str.contains(r"^\d{4}-\d{2}-\d{2}").fill_null(False)
                    if (valid_format | s.is_null()).all():
                        continue
                except:
                    pass

            # 保留：需要LLM分析的列
            complex_cols.append(col)

        return complex_cols