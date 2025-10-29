# cleaning_providers/base.py
"""
Data Cleaning Provider Base Class

专注于数据清洗：
- 类型一致性（数值列不应有字符串）
- 格式标准化（日期、电话、URL等）
- 异常值处理
- 重复值处理
- 不合理空值清理

不做：
- 缺失值填补（由MVI负责）
- 实体解析（由ER负责）
"""
from __future__ import annotations
import asyncio
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CleaningReport:
    """清洗报告"""
    cleaned_cells: int  # 清洗的单元格数
    cleaned_rows: int  # 涉及的行数
    cleaned_columns: List[str]  # 涉及的列

    # 按清洗类型分类
    type_fixes: int  # 类型修正
    format_fixes: int  # 格式修正
    outlier_fixes: int  # 异常值修正
    duplicate_fixes: int  # 重复值修正

    # 详细记录
    changes: List[Dict[str, Any]]  # [{row, column, before, after, reason}]

    def to_dict(self) -> dict:
        return asdict(self)


class CleaningProvider:
    """
    数据清洗Provider基类

    职责：
    - 识别数据质量问题
    - 执行清洗操作
    - 生成清洗报告

    子类需要实现：
    - clean(): 清洗数据的具体逻辑
    """

    name = "BaseCleaningProvider"

    def __init__(self):
        self._cache = {}

    async def clean(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """
        清洗数据（核心接口）

        Args:
            df: 输入数据
            primary_keys: 主键列（用于报告定位）
            rules: 清洗规则（可选，None时使用默认规则）

        Returns:
            (cleaned_df, report): 清洗后的数据和报告

        规则：
        - 只修正现有数据，不增加新行/新列
        - 保持主键不变
        - 记录所有变更
        """
        raise NotImplementedError("Subclass must implement clean()")

    def _is_numeric_column(self, df: pl.DataFrame, col: str) -> bool:
        """判断列是否应该是数值列"""
        if col not in df.columns:
            return False

        dtype = df.schema[col]
        if dtype.is_numeric():
            return True

        # 尝试转换，如果大部分能转换，则认为应该是数值列
        if dtype == pl.Utf8:
            try:
                s = df[col].cast(pl.Utf8).str.strip()
                # 去除逗号（千分位）
                s = s.str.replace_all(",", "")
                # 尝试转换
                converted = s.str.extract(r"^-?\d+(?:\.\d+)?$").is_not_null()
                ratio = converted.sum() / max(1, len(df))
                return ratio > 0.8  # 80%以上能转换
            except:
                return False

        return False

    def _is_date_column(self, df: pl.DataFrame, col: str) -> bool:
        """判断列是否应该是日期列"""
        if col not in df.columns:
            return False

        dtype = df.schema[col]
        if dtype in (pl.Date, pl.Datetime):
            return True

        # 检查列名
        col_lower = col.lower()
        date_keywords = ["date", "time", "year", "month", "day", "created", "updated"]
        if any(kw in col_lower for kw in date_keywords):
            return True

        return False

    def _detect_outliers_iqr(
            self,
            df: pl.DataFrame,
            col: str,
            multiplier: float = 1.5
    ) -> pl.Series:
        """
        使用IQR方法检测异常值

        Returns:
            布尔Series，True表示是异常值
        """
        if col not in df.columns:
            return pl.Series([False] * len(df))

        try:
            s = df[col].cast(pl.Float64, strict=False)

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            is_outlier = (s < lower_bound) | (s > upper_bound)
            return is_outlier
        except:
            return pl.Series([False] * len(df))

    def _create_report(
            self,
            changes: List[Dict[str, Any]]
    ) -> CleaningReport:
        """从变更列表创建报告"""
        if not changes:
            return CleaningReport(
                cleaned_cells=0,
                cleaned_rows=0,
                cleaned_columns=[],
                type_fixes=0,
                format_fixes=0,
                outlier_fixes=0,
                duplicate_fixes=0,
                changes=[]
            )

        cleaned_rows = len({c["row_index"] for c in changes})
        cleaned_columns = sorted({c["column"] for c in changes})

        # 统计各类型修正数量
        type_fixes = sum(1 for c in changes if "type" in c.get("reason", "").lower())
        format_fixes = sum(1 for c in changes if "format" in c.get("reason", "").lower())
        outlier_fixes = sum(1 for c in changes if "outlier" in c.get("reason", "").lower())
        duplicate_fixes = sum(1 for c in changes if "duplicate" in c.get("reason", "").lower())

        return CleaningReport(
            cleaned_cells=len(changes),
            cleaned_rows=cleaned_rows,
            cleaned_columns=cleaned_columns,
            type_fixes=type_fixes,
            format_fixes=format_fixes,
            outlier_fixes=outlier_fixes,
            duplicate_fixes=duplicate_fixes,
            changes=changes
        )


class CompositeCleaningProvider(CleaningProvider):
    """
    组合多个清洗Provider

    功能：
    - 依次调用多个cleaner
    - 累积清洗报告
    """

    name = "Composite"

    def __init__(self, providers: List[CleaningProvider]):
        super().__init__()
        self.providers = providers

    async def clean(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """依次调用所有cleaner"""
        current_df = df
        all_changes = []

        for provider in self.providers:
            try:
                print(f"[Cleaning] Running {provider.name}...")
                current_df, report = await provider.clean(
                    current_df,
                    primary_keys,
                    rules
                )
                all_changes.extend(report.changes)
            except Exception as e:
                print(f"[WARN] {provider.name} failed: {e}")
                continue

        # 合并报告
        combined_report = self._create_report(all_changes)
        return current_df, combined_report