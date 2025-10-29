# cleaning_providers/type_cleaner.py
"""
Type Consistency Cleaner

确保数据类型一致性：
- 数值列只包含数值
- 日期列格式统一
- 布尔值标准化
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import CleaningProvider, CleaningReport


class TypeCleaningProvider(CleaningProvider):
    """
    类型一致性清洗

    功能：
    - 自动识别列的预期类型
    - 转换不一致的值
    - 移除无法转换的值
    """

    name = "Type-Cleaning"

    async def clean(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """清洗类型不一致问题"""
        if df.height == 0:
            return df, self._create_report([])

        print(f"[Type-Cleaning] Checking {len(df.columns)} columns...")

        changes = []
        cleaned_df = df.clone()

        for col in df.columns:
            if col in primary_keys:
                continue

            # 数值列清洗
            if self._is_numeric_column(df, col):
                col_changes = self._clean_numeric_column(cleaned_df, col, primary_keys)
                if col_changes:
                    cleaned_df = self._apply_numeric_cleaning(cleaned_df, col, col_changes)
                    changes.extend(col_changes)

            # 日期列清洗
            elif self._is_date_column(df, col):
                col_changes = self._clean_date_column(cleaned_df, col, primary_keys)
                if col_changes:
                    cleaned_df = self._apply_date_cleaning(cleaned_df, col, col_changes)
                    changes.extend(col_changes)

            # 布尔列清洗
            elif self._is_boolean_column(df, col):
                col_changes = self._clean_boolean_column(cleaned_df, col, primary_keys)
                if col_changes:
                    cleaned_df = self._apply_boolean_cleaning(cleaned_df, col, col_changes)
                    changes.extend(col_changes)

        report = self._create_report(changes)
        if report.cleaned_cells > 0:
            print(f"[Type-Cleaning] ✓ Fixed {report.type_fixes} type issues")

        return cleaned_df, report

    def _is_boolean_column(self, df: pl.DataFrame, col: str) -> bool:
        """判断是否是布尔列"""
        if col not in df.columns:
            return False

        dtype = df.schema[col]
        if dtype == pl.Boolean:
            return True

        # 检查唯一值
        unique_vals = df[col].unique().to_list()
        unique_vals = [v for v in unique_vals if v is not None]

        if len(unique_vals) <= 2:
            # 转为小写字符串
            str_vals = {str(v).lower() for v in unique_vals}
            bool_keywords = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
            if str_vals.issubset(bool_keywords):
                return True

        return False

    def _clean_numeric_column(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """清洗数值列"""
        changes = []

        for i in range(len(df)):
            old_val = df[col][i]
            if old_val is None:
                continue

            # 尝试转换为数值
            try:
                str_val = str(old_val).strip()
                # 移除千分位逗号
                str_val = str_val.replace(",", "")
                # 移除货币符号
                str_val = re.sub(r"[$€£¥]", "", str_val)

                # 尝试转换
                if "." in str_val:
                    new_val = float(str_val)
                else:
                    new_val = int(str_val)

                if str(old_val) != str(new_val):
                    pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                    changes.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "before": old_val,
                        "after": new_val,
                        "reason": "Type fix: converted to numeric"
                    })
            except:
                # 无法转换，标记为None
                pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                changes.append({
                    "row_index": i,
                    **pk_info,
                    "column": col,
                    "before": old_val,
                    "after": None,
                    "reason": "Type fix: invalid numeric value"
                })

        return changes

    def _clean_date_column(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """清洗日期列"""
        changes = []

        # 常见日期格式
        date_patterns = [
            r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
            r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
            r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
        ]

        for i in range(len(df)):
            old_val = df[col][i]
            if old_val is None:
                continue

            str_val = str(old_val).strip()

            # 尝试匹配并标准化为 YYYY-MM-DD
            matched = False
            for pattern in date_patterns:
                match = re.match(pattern, str_val)
                if match:
                    groups = match.groups()

                    # 判断格式并重组
                    if len(groups[0]) == 4:  # YYYY-...
                        year, month, day = groups
                    else:  # MM/DD/YYYY
                        month, day, year = groups

                    try:
                        new_val = f"{int(year)}-{int(month):02d}-{int(day):02d}"

                        if str_val != new_val:
                            pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                            changes.append({
                                "row_index": i,
                                **pk_info,
                                "column": col,
                                "before": old_val,
                                "after": new_val,
                                "reason": "Type fix: date format standardized"
                            })

                        matched = True
                        break
                    except:
                        continue

            if not matched and old_val not in (None, ""):
                # 无法解析的日期，标记为None
                pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                changes.append({
                    "row_index": i,
                    **pk_info,
                    "column": col,
                    "before": old_val,
                    "after": None,
                    "reason": "Type fix: invalid date format"
                })

        return changes

    def _clean_boolean_column(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """清洗布尔列"""
        changes = []

        true_values = {"true", "yes", "1", "t", "y"}
        false_values = {"false", "no", "0", "f", "n"}

        for i in range(len(df)):
            old_val = df[col][i]
            if old_val is None:
                continue

            str_val = str(old_val).lower().strip()

            if str_val in true_values:
                new_val = True
            elif str_val in false_values:
                new_val = False
            else:
                new_val = None

            if old_val != new_val:
                pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                changes.append({
                    "row_index": i,
                    **pk_info,
                    "column": col,
                    "before": old_val,
                    "after": new_val,
                    "reason": "Type fix: boolean standardized"
                })

        return changes

    def _apply_numeric_cleaning(
            self,
            df: pl.DataFrame,
            col: str,
            changes: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用数值列清洗"""
        if not changes:
            return df

        change_map = {c["row_index"]: c["after"] for c in changes}

        new_values = []
        for i in range(len(df)):
            if i in change_map:
                new_values.append(change_map[i])
            else:
                new_values.append(df[col][i])

        return df.with_columns([
            pl.Series(col, new_values).cast(pl.Float64, strict=False)
        ])

    def _apply_date_cleaning(
            self,
            df: pl.DataFrame,
            col: str,
            changes: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用日期列清洗"""
        if not changes:
            return df

        change_map = {c["row_index"]: c["after"] for c in changes}

        new_values = []
        for i in range(len(df)):
            if i in change_map:
                new_values.append(change_map[i])
            else:
                new_values.append(df[col][i])

        return df.with_columns([
            pl.Series(col, new_values)
        ])

    def _apply_boolean_cleaning(
            self,
            df: pl.DataFrame,
            col: str,
            changes: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用布尔列清洗"""
        if not changes:
            return df

        change_map = {c["row_index"]: c["after"] for c in changes}

        new_values = []
        for i in range(len(df)):
            if i in change_map:
                new_values.append(change_map[i])
            else:
                new_values.append(df[col][i])

        return df.with_columns([
            pl.Series(col, new_values).cast(pl.Boolean, strict=False)
        ])