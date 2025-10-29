# cleaning_providers/llm_cleaner.py
"""
LLM-Driven Intelligent Data Cleaning

使用LLM自动识别数据质量问题并生成清洗策略
"""
from __future__ import annotations
import json
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import CleaningProvider, CleaningReport


class LLMCleaningProvider(CleaningProvider):
    """
    LLM驱动的智能数据清洗

    功能：
    1. 自动分析每列的数据特征
    2. 识别数据质量问题
    3. 生成清洗建议
    4. 执行清洗操作
    """

    name = "LLM-Cleaning"

    def __init__(self, llm_client, max_sample_rows: int = 50):
        super().__init__()
        self.llm = llm_client
        self.max_sample_rows = max_sample_rows

    async def clean(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """使用LLM智能清洗数据"""
        if df.height == 0:
            return df, self._create_report([])

        print(f"[LLM-Cleaning] Analyzing {len(df.columns)} columns...")

        changes = []
        cleaned_df = df.clone()

        # 对每列进行分析和清洗
        for col in df.columns:
            if col in primary_keys:
                # 跳过主键列
                continue

            try:
                col_changes = await self._clean_column(cleaned_df, col, primary_keys)
                if col_changes:
                    # 应用变更
                    cleaned_df = self._apply_column_changes(cleaned_df, col, col_changes)
                    changes.extend(col_changes)
            except Exception as e:
                print(f"[WARN] Failed to clean column {col}: {e}")
                continue

        report = self._create_report(changes)
        print(f"[LLM-Cleaning] ✓ Cleaned {report.cleaned_cells} cells in {report.cleaned_rows} rows")

        return cleaned_df, report

    async def _clean_column(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """清洗单个列"""
        # 采样数据
        sample_df = df.head(self.max_sample_rows)
        sample_values = sample_df[col].to_list()

        # 分析列
        analysis = await self._analyze_column(col, sample_values)

        if not analysis or not analysis.get("has_issues"):
            return []

        # 生成清洗策略
        cleaning_strategy = analysis.get("cleaning_strategy", {})

        # 执行清洗
        changes = []

        # 1. 类型修正
        if cleaning_strategy.get("fix_type"):
            target_type = cleaning_strategy.get("target_type")
            changes.extend(self._fix_type_issues(df, col, target_type, primary_keys))

        # 2. 格式标准化
        if cleaning_strategy.get("standardize_format"):
            format_pattern = cleaning_strategy.get("format_pattern")
            changes.extend(self._standardize_format(df, col, format_pattern, primary_keys))

        # 3. 清理异常值
        if cleaning_strategy.get("remove_outliers"):
            changes.extend(self._remove_outliers(df, col, primary_keys))

        return changes

    async def _analyze_column(
            self,
            col_name: str,
            sample_values: List[Any]
    ) -> Dict[str, Any]:
        """使用LLM分析列的数据质量"""
        # 准备样本（去除None）
        non_null_values = [v for v in sample_values if v is not None][:20]

        if not non_null_values:
            return {"has_issues": False}

        prompt = f"""Analyze this data column for quality issues.

Column name: {col_name}
Sample values: {json.dumps(non_null_values, ensure_ascii=False)}

Identify any data quality issues:
1. Type inconsistency (e.g., numbers mixed with text)
2. Format inconsistency (e.g., dates in different formats)
3. Obvious errors (e.g., typos, invalid values)
4. Outliers (e.g., extremely large/small values)

Return ONLY JSON:
{{
    "has_issues": true/false,
    "issues": ["issue1", "issue2"],
    "cleaning_strategy": {{
        "fix_type": true/false,
        "target_type": "numeric"/"date"/"text"/null,
        "standardize_format": true/false,
        "format_pattern": "YYYY-MM-DD"/null,
        "remove_outliers": true/false
    }}
}}
"""

        try:
            response = await self.llm.chat("gpt-4o-mini", [
                {"role": "user", "content": prompt}
            ])

            # 解析JSON
            analysis = json.loads(response.strip())
            return analysis
        except Exception as e:
            print(f"[WARN] LLM analysis failed for {col_name}: {e}")
            return {"has_issues": False}

    def _fix_type_issues(
            self,
            df: pl.DataFrame,
            col: str,
            target_type: Optional[str],
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """修正类型不一致问题"""
        changes = []

        if target_type == "numeric":
            # 转换为数值
            for i in range(len(df)):
                old_val = df[col][i]
                if old_val is None:
                    continue

                try:
                    # 尝试转换
                    str_val = str(old_val).strip().replace(",", "")
                    new_val = float(str_val)

                    if old_val != new_val:
                        pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                        changes.append({
                            "row_index": i,
                            **pk_info,
                            "column": col,
                            "before": old_val,
                            "after": new_val,
                            "reason": "Type correction: string to numeric"
                        })
                except:
                    # 无法转换，标记为None
                    if old_val not in (None, ""):
                        pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                        changes.append({
                            "row_index": i,
                            **pk_info,
                            "column": col,
                            "before": old_val,
                            "after": None,
                            "reason": "Type correction: invalid numeric value removed"
                        })

        return changes

    def _standardize_format(
            self,
            df: pl.DataFrame,
            col: str,
            format_pattern: Optional[str],
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """标准化格式"""
        changes = []

        if format_pattern == "YYYY-MM-DD":
            # 标准化日期格式
            import re
            date_patterns = [
                (r"(\d{4})-(\d{1,2})-(\d{1,2})", "{0}-{1:02d}-{2:02d}"),
                (r"(\d{1,2})/(\d{1,2})/(\d{4})", "{2}-{0:02d}-{1:02d}"),
            ]

            for i in range(len(df)):
                old_val = df[col][i]
                if old_val is None:
                    continue

                str_val = str(old_val).strip()

                for pattern, format_str in date_patterns:
                    match = re.match(pattern, str_val)
                    if match:
                        groups = [int(g) for g in match.groups()]
                        new_val = format_str.format(*groups)

                        if str_val != new_val:
                            pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                            changes.append({
                                "row_index": i,
                                **pk_info,
                                "column": col,
                                "before": old_val,
                                "after": new_val,
                                "reason": "Format standardization: date"
                            })
                        break

        return changes

    def _remove_outliers(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """移除异常值"""
        changes = []

        # 使用IQR方法检测异常值
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
                    "reason": "Outlier removed (IQR method)"
                })

        return changes

    def _apply_column_changes(
            self,
            df: pl.DataFrame,
            col: str,
            changes: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用列的变更"""
        if not changes:
            return df

        # 构建变更映射
        change_map = {c["row_index"]: c["after"] for c in changes}

        # 应用变更
        new_values = []
        for i in range(len(df)):
            if i in change_map:
                new_values.append(change_map[i])
            else:
                new_values.append(df[col][i])

        # 更新DataFrame
        return df.with_columns([
            pl.Series(col, new_values)
        ])