# cleaning_providers/batch_llm_cleaner.py
from __future__ import annotations
import json
import asyncio
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import CleaningProvider, CleaningReport


class BatchLLMCleaningProvider(CleaningProvider):
    """
    批量 LLM 清洗器
    1. 批量分析：一次请求分析 5-10 列
    2. 智能筛选：跳过简单列（类型一致、格式标准）
    3. 减少采样：20 行而非 50 行
    4. 并发处理：多批次并发
    """

    name = "Batch-LLM-Cleaning"

    def __init__(
            self,
            llm_client,
            batch_size: int = 5,
            max_sample_rows: int = 20,
            max_concurrent_batches: int = 3
    ):
        """
        Args:
            llm_client: LLM 客户端
            batch_size: 每批处理列数（5-10 最优）
            max_sample_rows: 采样行数（减少 token 消耗）
            max_concurrent_batches: 最大并发批次
        """
        super().__init__()
        self.llm = llm_client
        self.batch_size = batch_size
        self.max_sample_rows = max_sample_rows
        self.sem = asyncio.Semaphore(max_concurrent_batches)

    async def clean(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """批量清洗"""
        if df.height == 0:
            return df, self._create_report([])

        print(f"[Batch-LLM-Cleaning] Analyzing {len(df.columns)} columns (batch_size={self.batch_size})...")

        # 1. 智能过滤：找出需要 LLM 清洗的列
        complex_cols = self._filter_complex_columns(df, primary_keys)
        print(f"[Batch-LLM-Cleaning] {len(complex_cols)} columns need LLM cleaning")

        if not complex_cols:
            return df, self._create_report([])

        # 2. 分批
        batches = [complex_cols[i:i + self.batch_size] for i in range(0, len(complex_cols), self.batch_size)]

        # 3. 并发处理批次
        tasks = [self._process_batch(df, batch, primary_keys) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # 4. 合并结果
        all_changes = []
        for changes in batch_results:
            all_changes.extend(changes)

        # 5. 应用变更
        cleaned_df = df.clone()
        if all_changes:
            cleaned_df = self._apply_all_changes(cleaned_df, all_changes)

        report = self._create_report(all_changes)
        if report.cleaned_cells > 0:
            print(f"[Batch-LLM-Cleaning] ✓ Cleaned {report.cleaned_cells} cells in {report.cleaned_rows} rows")

        return cleaned_df, report

    def _filter_complex_columns(
            self,
            df: pl.DataFrame,
            primary_keys: List[str]
    ) -> List[str]:
        """
        智能过滤：找出需要 LLM 清洗的列

        跳过：
        - 主键
        - 类型一致的数值列
        - 格式标准的字符串列
        - 唯一值比例极高的列（ID 类）
        - 唯一值极少的列（枚举类，简单规则即可）

        保留：
        - 类型混乱的列
        - 格式不统一的列
        - 有明显异常值的列
        """
        complex_cols = []

        for col in df.columns:
            if col in primary_keys:
                continue

            # 基本统计
            s = df[col]
            null_rate = s.null_count() / max(1, len(s))
            unique_rate = s.n_unique() / max(1, len(s))

            # 跳过规则 1: 唯一值比例极高（> 0.95，可能是 ID）
            if unique_rate > 0.95:
                continue

            # 跳过规则 2: 唯一值极少（< 10 个，枚举类）
            if s.n_unique() < 10 and null_rate < 0.1:
                continue

            # 跳过规则 3: 完美的数值列
            if df.schema[col].is_numeric():
                # 检查是否有异常值
                try:
                    is_outlier = self._detect_outliers_iqr(df, col)
                    outlier_rate = is_outlier.sum() / max(1, len(df))
                    if outlier_rate < 0.05:  # 异常值 < 5%
                        continue
                except:
                    pass

            # 保留：需要 LLM 分析的列
            complex_cols.append(col)

        return complex_cols

    async def _process_batch(
            self,
            df: pl.DataFrame,
            batch_cols: List[str],
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """处理一批列（并发控制）"""
        async with self.sem:
            return await self._clean_batch(df, batch_cols, primary_keys)

    async def _clean_batch(
            self,
            df: pl.DataFrame,
            batch_cols: List[str],
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """批量清洗一批列"""
        if not batch_cols:
            return []

        # 1. 采样数据
        sample_df = df.head(self.max_sample_rows)

        # 2. 准备批量分析数据
        batch_data = {}
        for col in batch_cols:
            sample_values = sample_df[col].drop_nulls().to_list()[:10]  # 每列最多 10 个样本
            batch_data[col] = {
                "dtype": str(df.schema[col]),
                "sample_values": sample_values,
                "null_rate": round(df[col].null_count() / len(df), 3)
            }

        # 3. 批量分析（一次 LLM 请求）
        analysis_result = await self._analyze_batch(batch_cols, batch_data)

        # 4. 根据分析结果执行清洗
        all_changes = []
        for col, strategy in analysis_result.items():
            if col not in df.columns:
                continue

            if not strategy or not strategy.get("needs_cleaning"):
                continue

            # 执行清洗策略
            col_changes = await self._apply_strategy(df, col, strategy, primary_keys)
            all_changes.extend(col_changes)

        return all_changes

    async def _analyze_batch(
            self,
            batch_cols: List[str],
            batch_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, Any]]:
        """批量分析（关键优化：一次请求分析多列）"""
        prompt = f"""Analyze these data columns for quality issues (in one response).

Columns to analyze:
{json.dumps(batch_data, indent=2, ensure_ascii=False)}

For EACH column, identify:
1. Does it need cleaning? (true/false)
2. What cleaning strategies? (type_fix, format_standardize, outlier_removal)

Return ONLY JSON:
{{
  "column_name": {{
    "needs_cleaning": true/false,
    "strategies": {{
      "fix_type": true/false,
      "target_type": "numeric"/"date"/"text"/null,
      "standardize_format": true/false,
      "format_pattern": "YYYY-MM-DD"/null,
      "remove_outliers": true/false
    }}
  }},
  ...
}}

IMPORTANT: Be conservative - only suggest cleaning if truly needed.
"""

        try:
            response = await self.llm.chat("gpt-4o-mini", [
                {"role": "user", "content": prompt}
            ])

            # 解析 JSON
            result = json.loads(response.strip())

            # 验证并返回
            analyzed = {}
            for col in batch_cols:
                if col in result:
                    analyzed[col] = result[col].get("strategies", {})
                    analyzed[col]["needs_cleaning"] = result[col].get("needs_cleaning", False)

            return analyzed

        except Exception as e:
            print(f"[Batch-LLM-Cleaning] Batch analysis failed: {e}")
            # 回退：不清洗
            return {col: {"needs_cleaning": False} for col in batch_cols}

    async def _apply_strategy(
            self,
            df: pl.DataFrame,
            col: str,
            strategy: Dict[str, Any],
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """应用清洗策略（基于 LLM 分析结果）"""
        changes = []

        try:
            # 策略 1: 类型修正
            if strategy.get("fix_type"):
                target_type = strategy.get("target_type")
                changes.extend(self._fix_type_fast(df, col, target_type, primary_keys))

            # 策略 2: 格式标准化
            if strategy.get("standardize_format"):
                format_pattern = strategy.get("format_pattern")
                changes.extend(self._standardize_format_fast(df, col, format_pattern, primary_keys))

            # 策略 3: 异常值移除
            if strategy.get("remove_outliers"):
                changes.extend(self._remove_outliers_fast(df, col, primary_keys))

        except Exception as e:
            print(f"[Batch-LLM-Cleaning] Error applying strategy to {col}: {e}")

        return changes

    def _fix_type_fast(
            self,
            df: pl.DataFrame,
            col: str,
            target_type: Optional[str],
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """快速类型修正（向量化）"""
        changes = []

        if target_type == "numeric":
            try:
                s = df[col].cast(pl.Utf8).str.strip().str.replace_all(",", "")
                s_numeric = s.cast(pl.Float64, strict=False)

                for i in range(len(df)):
                    old_val = df[col][i]
                    new_val = s_numeric[i]

                    if old_val != new_val and new_val is not None:
                        pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                        changes.append({
                            "row_index": i,
                            **pk_info,
                            "column": col,
                            "before": old_val,
                            "after": new_val,
                            "reason": "LLM-suggested type fix: numeric"
                        })
            except Exception:
                pass

        return changes

    def _standardize_format_fast(
            self,
            df: pl.DataFrame,
            col: str,
            format_pattern: Optional[str],
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """快速格式标准化"""
        changes = []

        if format_pattern == "YYYY-MM-DD":
            try:
                s = df[col].cast(pl.Utf8)
                # 使用 Polars 内置日期解析
                s_date = pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                s_parsed = df.select(s_date).to_series()

                for i in range(len(df)):
                    old_val = s[i]
                    new_val = str(s_parsed[i]) if s_parsed[i] is not None else None

                    if old_val != new_val and new_val is not None:
                        pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                        changes.append({
                            "row_index": i,
                            **pk_info,
                            "column": col,
                            "before": old_val,
                            "after": new_val,
                            "reason": "LLM-suggested format fix: date"
                        })
            except Exception:
                pass

        return changes

    def _remove_outliers_fast(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """快速异常值移除"""
        changes = []

        try:
            is_outlier = self._detect_outliers_iqr(df, col)

            for i in range(len(df)):
                if is_outlier[i]:
                    old_val = df[col][i]
                    pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                    changes.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "before": old_val,
                        "after": None,
                        "reason": "LLM-suggested outlier removal"
                    })
        except Exception:
            pass

        return changes

    def _apply_all_changes(
            self,
            df: pl.DataFrame,
            all_changes: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """批量应用所有变更"""
        # 按列分组
        changes_by_col = {}
        for change in all_changes:
            col = change["column"]
            if col not in changes_by_col:
                changes_by_col[col] = []
            changes_by_col[col].append(change)

        # 逐列应用
        result_df = df.clone()
        for col, col_changes in changes_by_col.items():
            result_df = self._apply_column_changes(result_df, col, col_changes)

        return result_df

    def _apply_column_changes(
            self,
            df: pl.DataFrame,
            col: str,
            changes: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用单列变更"""
        if not changes:
            return df

        change_map = {c["row_index"]: c["after"] for c in changes}

        new_values = []
        for i in range(len(df)):
            if i in change_map:
                new_values.append(change_map[i])
            else:
                new_values.append(df[col][i])

        return df.with_columns([pl.Series(col, new_values)])