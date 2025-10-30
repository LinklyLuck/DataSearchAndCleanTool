# cleaning_providers/type_cleaner.py
"""
Type Consistency Cleaner
确保数据类型一致性：
- 数值列只包含数值
- 日期列格式统一
- 布尔值标准化
"""
from __future__ import annotations
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
                cleaned_df, col_changes = self._clean_numeric_column(cleaned_df, col, primary_keys)
                if col_changes:
                    changes.extend(col_changes)

            # 日期列清洗
            elif self._is_date_column(df, col):
                cleaned_df, col_changes = self._clean_date_column(cleaned_df, col, primary_keys)
                if col_changes:
                    changes.extend(col_changes)

            # 布尔列清洗
            elif self._is_boolean_column(df, col):
                cleaned_df, col_changes = self._clean_boolean_column(cleaned_df, col, primary_keys)
                if col_changes:
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
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
        """清洗数值列（向量化实现）"""
        if df.height == 0:
            return df, []

        original = pl.col(col)
        original_str = original.cast(pl.Utf8, strict=False)

        sanitized = original_str.str.strip()
        sanitized = sanitized.str.replace_all(",", "")
        sanitized = sanitized.str.replace_all(r"[$€£¥]", "")
        sanitized = pl.when(
            sanitized.is_null() | (sanitized.str.len_chars() == 0)
        ).then(None).otherwise(sanitized)

        is_numeric = sanitized.str.match(r"^-?\d+(?:\.\d+)?$").fill_null(False)
        has_decimal = sanitized.str.contains(r"\.").fill_null(False)

        as_float = sanitized.cast(pl.Float64, strict=False)
        as_int = sanitized.cast(pl.Int64, strict=False)

        converted_numeric = pl.when(is_numeric & has_decimal).then(as_float).when(is_numeric).then(
            as_int.cast(pl.Float64, strict=False)
        ).otherwise(None)
        converted_str = pl.when(is_numeric & has_decimal).then(
            as_float.cast(pl.Utf8, strict=False)
        ).when(is_numeric).then(
            as_int.cast(pl.Utf8, strict=False)
        ).otherwise(None)

        invalid_mask = original.is_not_null() & sanitized.is_not_null() & (~is_numeric)
        converted_change = (converted_str != original_str).fill_null(False) & is_numeric
        needs_update = converted_change | invalid_mask

        report_df = df.select([
            pl.arange(0, pl.count()).alias("_row_index"),
            original.alias("_original"),
            converted_numeric.alias("_converted_numeric"),
            converted_str.alias("_converted_str"),
            invalid_mask.alias("_invalid"),
            converted_change.alias("_converted_change"),
            needs_update.alias("_needs_update")
        ])

        if not report_df["_needs_update"].any():
            return df, []

        changes: List[Dict[str, Any]] = []
        for row in report_df.filter(pl.col("_needs_update")).iter_rows(named=True):
            idx = row["_row_index"]
            pk_info = {k: df[k][idx] for k in primary_keys if k in df.columns}
            if row["_invalid"]:
                changes.append({
                    "row_index": idx,
                    **pk_info,
                    "column": col,
                    "before": row["_original"],
                    "after": None,
                    "reason": "Type fix: invalid numeric value"
                })
            else:
                changes.append({
                    "row_index": idx,
                    **pk_info,
                    "column": col,
                    "before": row["_original"],
                    "after": row["_converted_numeric"],
                    "reason": "Type fix: converted to numeric"
                })
            updated_df = df.with_columns([
                pl.when(is_numeric & has_decimal)
                .then(as_float)
                .when(is_numeric)
                .then(as_int.cast(pl.Float64, strict=False))
                .when(invalid_mask)
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(original.cast(pl.Float64, strict=False))
                .alias(col)
            ])

            return updated_df, changes

    def _clean_date_column(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
        """清洗日期列（向量化实现）"""
        if df.height == 0:
            return df, []

        # 已经是日期/日期时间列则跳过
        if df.schema[col] in (pl.Date, pl.Datetime):
            return df, []

        original = pl.col(col)
        original_str = original.cast(pl.Utf8, strict=False)
        stripped = original_str.str.strip()
        stripped = pl.when(
            stripped.is_null() | (stripped.str.len_chars() == 0)
        ).then(None).otherwise(stripped)

        parsed = pl.coalesce(
            stripped.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            stripped.str.strptime(pl.Date, "%m/%d/%Y", strict=False),
            stripped.str.strptime(pl.Date, "%Y%m%d", strict=False)
        )
        normalized = parsed.dt.strftime("%Y-%m-%d")

        invalid_mask = original.is_not_null() & stripped.is_not_null() & parsed.is_null()
        changed_mask = (normalized != original_str).fill_null(False) & normalized.is_not_null()
        needs_update = changed_mask | invalid_mask

        report_df = df.select([
            pl.arange(0, pl.count()).alias("_row_index"),
            original.alias("_original"),
            normalized.alias("_normalized"),
            invalid_mask.alias("_invalid"),
            changed_mask.alias("_changed"),
            needs_update.alias("_needs_update")
        ])

        if not report_df["_needs_update"].any():
            return df, []

        changes: List[Dict[str, Any]] = []
        for row in report_df.filter(pl.col("_needs_update")).iter_rows(named=True):
            idx = row["_row_index"]
            pk_info = {k: df[k][idx] for k in primary_keys if k in df.columns}
            if row["_invalid"]:
                changes.append({
                    "row_index": idx,
                    **pk_info,
                    "column": col,
                    "before": row["_original"],
                    "after": None,
                    "reason": "Type fix: invalid date format"
                })
            else:
                changes.append({
                    "row_index": idx,
                    **pk_info,
                    "column": col,
                    "before": row["_original"],
                    "after": row["_normalized"],
                    "reason": "Type fix: date format standardized"
                })

        updated_df = df.with_columns([
            pl.when(normalized.is_not_null())
            .then(normalized)
            .when(invalid_mask)
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(original_str)
            .alias(col)
        ])

        return updated_df, changes

    def _clean_boolean_column(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
        """清洗布尔列（向量化实现）"""
        if df.height == 0:
            return df, []

        if df.schema[col] == pl.Boolean:
            return df, []

        original = pl.col(col)
        original_str = original.cast(pl.Utf8, strict=False)
        lowered = original_str.str.to_lowercase().str.strip()
        lowered = pl.when(
            lowered.is_null() | (lowered.str.len_chars() == 0)
        ).then(None).otherwise(lowered)

        true_values = ["true", "yes", "1", "t", "y"]
        false_values = ["false", "no", "0", "f", "n"]

        converted = pl.when(lowered.is_in(true_values)).then(pl.lit(True)).when(
            lowered.is_in(false_values)
        ).then(pl.lit(False)).otherwise(None)

        invalid_mask = original.is_not_null() & lowered.is_not_null() & converted.is_null()
        changed_mask = (converted.cast(pl.Utf8, strict=False) != original_str).fill_null(False) & converted.is_not_null()
        needs_update = changed_mask | invalid_mask

        report_df = df.select([
            pl.arange(0, pl.count()).alias("_row_index"),
            original.alias("_original"),
            converted.alias("_converted"),
            invalid_mask.alias("_invalid"),
            changed_mask.alias("_changed"),
            needs_update.alias("_needs_update")
        ])


        if not report_df["_needs_update"].any():
            return df, []


        changes: List[Dict[str, Any]] = []
        for row in report_df.filter(pl.col("_needs_update")).iter_rows(named=True):
            idx = row["_row_index"]
            pk_info = {k: df[k][idx] for k in primary_keys if k in df.columns}
            if row["_invalid"]:
                changes.append({
                    "row_index": idx,
                    **pk_info,
                    "column": col,
                    "before": row["_original"],
                    "after": None,
                    "reason": "Type fix: boolean standardized"
                })
            else:
                changes.append({
                    "row_index": idx,
                    **pk_info,
                    "column": col,
                    "before": row["_original"],
                    "after": row["_converted"],
                    "reason": "Type fix: boolean standardized"
                })

        updated_df = df.with_columns([
            pl.when(converted.is_not_null())
            .then(converted)
            .when(invalid_mask)
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(original)
            .alias(col)
        ])

        return updated_df, changes

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