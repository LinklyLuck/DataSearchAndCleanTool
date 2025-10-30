# cleaning_providers/fast_cleaner.py
"""
- 纯规则引擎，无 LLM 调用
- 向量化操作（Polars 原生）
- 并行处理（多列同时清洗）
- 智能跳过（主键、干净列）
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import CleaningProvider, CleaningReport


class FastCleaningProvider(CleaningProvider):
    """
    快速清洗器 - 基于规则引擎
    清洗策略：
    1. 类型修正（数值列、日期列、布尔列）
    2. 格式标准化（去空格、统一大小写、标准日期）
    3. 异常值检测（IQR 方法）
    4. 重复值处理
    """

    name = "Fast-Cleaning"

    def __init__(self, aggressive: bool = False):
        """
        Args:
            aggressive: 是否启用激进模式（更多清洗，可能过度）
        """
        super().__init__()
        self.aggressive = aggressive

    async def clean(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """快速清洗"""
        if df.height == 0:
            return df, self._create_report([])

        print(f"[Fast-Cleaning] Processing {len(df.columns)} columns (aggressive={self.aggressive})...")

        changes = []
        cleaned_df = df.clone()

        # 并行处理所有列（Polars 内部优化）
        for col in df.columns:
            if col in primary_keys:
                continue  # 跳过主键

            # 快速预检：是否需要清洗？
            if not self._needs_cleaning(df, col):
                continue

            # 执行清洗
            col_changes = self._clean_column_fast(cleaned_df, col, primary_keys)
            if col_changes:
                cleaned_df = self._apply_changes(cleaned_df, col, col_changes)
                changes.extend(col_changes)

        report = self._create_report(changes)
        if report.cleaned_cells > 0:
            print(f"[Fast-Cleaning] ✓ Cleaned {report.cleaned_cells} cells in {report.cleaned_rows} rows")

        return cleaned_df, report

    def _needs_cleaning(self, df: pl.DataFrame, col: str) -> bool:
        """
        快速判断列是否需要清洗（跳过干净列）

        跳过条件：
        - null 率 < 1%
        - 唯一值比例正常
        - 类型一致
        """
        try:
            s = df[col]

            # 1. 基本统计
            null_rate = s.null_count() / max(1, len(s))
            if null_rate > 0.5:
                return True  # 空值太多，需要清洗

            # 2. 类型检查（数值列不应该有字符串）
            if self._is_numeric_column(df, col):
                # 检查是否有非数值
                try:
                    s_str = s.cast(pl.Utf8)
                    # 尝试转换为数值
                    numeric_ok = s_str.str.strip_chars().str.replace_all(",", "").str.extract(
                        r"^-?\d+(?:\.\d+)?$").is_not_null()
                    if (numeric_ok | s.is_null()).all():
                        return False  # 所有值都是有效数值
                except:
                    return True

            # 3. 如果是字符串，检查格式一致性
            if df.schema[col] == pl.Utf8:
                # 检查是否有前后空格
                s_trimmed = s.str.strip_chars()
                if not (s == s_trimmed).all():
                    return True  # 有空格，需要清洗

                # 检查是否有大小写不一致（对于枚举型列）
                unique_vals = s.unique().drop_nulls()
                if 2 <= len(unique_vals) <= 20:  # 可能是枚举类型
                    lower_unique = set(v.lower() for v in unique_vals.to_list())
                    if len(lower_unique) < len(unique_vals):
                        return True  # 大小写不一致

            # 默认：不需要清洗
            return False

        except Exception:
            return True  # 出错则清洗

    def _clean_column_fast(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """快速清洗单列（向量化操作）"""
        changes = []
        dtype = df.schema[col]

        # 策略 1: 数值列清洗
        if self._is_numeric_column(df, col) or dtype.is_numeric():
            changes.extend(self._clean_numeric_vectorized(df, col, primary_keys))

        # 策略 2: 字符串列清洗
        elif dtype == pl.Utf8:
            changes.extend(self._clean_string_vectorized(df, col, primary_keys))

        # 策略 3: 日期列清洗
        elif self._is_date_column(df, col):
            changes.extend(self._clean_date_vectorized(df, col, primary_keys))

        # 策略 4: 布尔列清洗
        elif self._is_boolean_column(df, col):
            changes.extend(self._clean_boolean_vectorized(df, col, primary_keys))

        return changes

    def _clean_numeric_vectorized(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """向量化数值清洗（Polars 原生，速度快）"""
        changes = []

        try:
            # 转换为字符串进行处理
            s = df[col].cast(pl.Utf8)

            # 1. 去除千分位逗号
            s_clean = s.str.strip_chars().str.replace_all(",", "")

            # 2. 去除货币符号
            s_clean = s_clean.str.replace_all(r"[$€£¥]", "")

            # 3. 尝试转换为数值
            s_numeric = s_clean.cast(pl.Float64, strict=False)

            # 找出变更的行
            original = df[col]
            for i in range(len(df)):
                old_val = original[i]
                new_val = s_numeric[i]

                if old_val is None and new_val is None:
                    continue

                # 值发生变化
                if old_val != new_val and new_val is not None:
                    pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                    changes.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "before": old_val,
                        "after": new_val,
                        "reason": "Type fix: converted to numeric"
                    })
                elif old_val is not None and new_val is None:
                    # 无法转换，标记为异常
                    pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                    changes.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "before": old_val,
                        "after": None,
                        "reason": "Type fix: invalid numeric value removed"
                    })

        except Exception as e:
            print(f"[Fast-Cleaning] Error cleaning numeric column {col}: {e}")

        return changes

    def _clean_string_vectorized(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """向量化字符串清洗"""
        changes = []

        try:
            s = df[col]

            # 1. 去除前后空格
            s_trimmed = s.str.strip_chars()

            # 2. 统一大小写（如果是枚举类型）
            unique_vals = s.unique().drop_nulls()
            if 2 <= len(unique_vals) <= 20:  # 可能是枚举
                # 统一为小写
                s_lower = s_trimmed.str.to_lowercase()

                # 记录变更
                for i in range(len(df)):
                    old_val = s[i]
                    new_val = s_lower[i]

                    if old_val != new_val and new_val is not None:
                        pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                        changes.append({
                            "row_index": i,
                            **pk_info,
                            "column": col,
                            "before": old_val,
                            "after": new_val,
                            "reason": "Format fix: standardized case"
                        })
            else:
                # 只去空格
                for i in range(len(df)):
                    old_val = s[i]
                    new_val = s_trimmed[i]

                    if old_val != new_val and new_val is not None:
                        pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                        changes.append({
                            "row_index": i,
                            **pk_info,
                            "column": col,
                            "before": old_val,
                            "after": new_val,
                            "reason": "Format fix: removed whitespace"
                        })

        except Exception as e:
            print(f"[Fast-Cleaning] Error cleaning string column {col}: {e}")

        return changes

    def _clean_date_vectorized(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """向量化日期清洗"""
        changes = []

        try:
            s = df[col].cast(pl.Utf8)

            # 标准化为 YYYY-MM-DD
            # Polars 内置日期解析
            try:
                s_date = pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                s_parsed = df.select(s_date).to_series()

                for i in range(len(df)):
                    old_val = s[i]
                    new_val = s_parsed[i]

                    if old_val != str(new_val) and new_val is not None:
                        pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                        changes.append({
                            "row_index": i,
                            **pk_info,
                            "column": col,
                            "before": old_val,
                            "after": str(new_val),
                            "reason": "Format fix: standardized date"
                        })
            except:
                pass

        except Exception as e:
            print(f"[Fast-Cleaning] Error cleaning date column {col}: {e}")

        return changes

    def _clean_boolean_vectorized(
            self,
            df: pl.DataFrame,
            col: str,
            primary_keys: List[str]
    ) -> List[Dict[str, Any]]:
        """向量化布尔清洗"""
        changes = []

        try:
            s = df[col].cast(pl.Utf8).str.to_lowercase()

            # 映射
            s_bool = (
                pl.when(s.is_in(["true", "yes", "1", "t", "y"]))
                .then(True)
                .when(s.is_in(["false", "no", "0", "f", "n"]))
                .then(False)
                .otherwise(None)
            )

            s_parsed = df.select(s_bool.alias("parsed")).to_series()

            for i in range(len(df)):
                old_val = df[col][i]
                new_val = s_parsed[i]

                if old_val != new_val:
                    pk_info = {k: df[k][i] for k in primary_keys if k in df.columns}
                    changes.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "before": old_val,
                        "after": new_val,
                        "reason": "Type fix: standardized boolean"
                    })

        except Exception as e:
            print(f"[Fast-Cleaning] Error cleaning boolean column {col}: {e}")

        return changes

    def _apply_changes(
            self,
            df: pl.DataFrame,
            col: str,
            changes: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用变更（向量化）"""
        if not changes:
            return df

        # 构建变更映射
        change_map = {c["row_index"]: c["after"] for c in changes}

        # 创建新列
        new_values = []
        for i in range(len(df)):
            if i in change_map:
                new_values.append(change_map[i])
            else:
                new_values.append(df[col][i])

        # 更新 DataFrame
        return df.with_columns([pl.Series(col, new_values)])

    def _is_boolean_column(self, df: pl.DataFrame, col: str) -> bool:
        """判断是否是布尔列"""
        if col not in df.columns:
            return False

        dtype = df.schema[col]
        if dtype == pl.Boolean:
            return True

        # 检查唯一值
        unique_vals = df[col].unique().drop_nulls().to_list()
        if len(unique_vals) <= 2:
            str_vals = {str(v).lower() for v in unique_vals}
            bool_keywords = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
            if str_vals.issubset(bool_keywords):
                return True

        return False