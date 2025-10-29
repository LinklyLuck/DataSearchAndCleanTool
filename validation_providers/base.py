# validation_providers/base.py
"""
Data Validation Provider Base Class

专注于数据验证：
- 基于外部权威API验证数据准确性
- 只修正错误数据，不增加新列/新行
- 记录所有验证结果

不做：
- 缺失值填补（由MVI负责）
- 实体解析（由ER负责）
- 数据清洗（由Cleaning负责）
"""
from __future__ import annotations
import asyncio
import httpx
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from dataclasses import dataclass, asdict


@dataclass
class ValidationReport:
    """验证报告"""
    validated_cells: int  # 验证的单元格数
    corrected_cells: int  # 修正的单元格数
    validated_rows: int  # 验证的行数
    corrected_rows: int  # 修正的行数

    # 准确性统计
    accuracy_before: float  # 验证前准确率（估计）
    accuracy_after: float  # 验证后准确率

    # 详细记录
    corrections: List[Dict[str, Any]]  # [{row, column, old_value, new_value, source}]

    # 来源统计
    source_distribution: Dict[str, int]  # {"Wikipedia": 10, "TMDb": 5}

    def to_dict(self) -> dict:
        return asdict(self)


class ValidationProvider:
    """
    数据验证Provider基类

    职责：
    - 查询外部权威数据源
    - 对比现有数据
    - 修正不一致的数据
    - 生成验证报告

    子类需要实现：
    - validate(): 验证数据的具体逻辑
    """

    name = "BaseValidationProvider"

    def __init__(
            self,
            timeout: int = 20,
            max_concurrency: int = 5,
            api_key: Optional[str] = None
    ):
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.api_key = api_key
        self.sem = asyncio.Semaphore(max_concurrency)
        self.client = httpx.AsyncClient(timeout=timeout)
        self._cache = {}

    async def aclose(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

    async def validate(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            validate_columns: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, ValidationReport]:
        """
        验证数据（核心接口）

        Args:
            df: 输入数据
            primary_keys: 主键列（用于查询外部API）
            validate_columns: 要验证的列（None表示所有列）

        Returns:
            (validated_df, report): 验证后的数据和报告

        规则：
        - 只修正错误数据，不增加新列/新行
        - 保持主键不变
        - 记录所有修正
        - 优先信任外部权威数据源
        """
        raise NotImplementedError("Subclass must implement validate()")

    def _is_missing(self, value: Any) -> bool:
        """判断值是否缺失"""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    def _extract_entity_name(
            self,
            row: Dict[str, Any],
            primary_keys: List[str]
    ) -> Optional[str]:
        """
        从行中提取实体名称（用于API查询）

        策略：
        1. 优先使用第一个主键
        2. 查找常见名称字段
        3. 回退到第一个字符串字段
        """
        # 策略1: 使用主键
        if primary_keys:
            for pk in primary_keys:
                if pk in row and row[pk]:
                    val = str(row[pk]).strip()
                    if val:
                        return val

        # 策略2: 查找常见名称字段
        name_candidates = [
            "name", "title", "entity_name",
            "movie_title", "product_name", "company_name",
            "ticker", "symbol", "brand"
        ]
        for field in name_candidates:
            if field in row and row[field]:
                val = str(row[field]).strip()
                if val:
                    return val

        # 策略3: 第一个非空字符串字段
        for key, value in row.items():
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    async def _make_request(
            self,
            method: str,
            url: str,
            **kwargs
    ) -> Optional[httpx.Response]:
        """
        发起HTTP请求（带并发控制）

        Args:
            method: HTTP方法 (GET/POST)
            url: 请求URL
            **kwargs: 其他参数

        Returns:
            响应对象，失败返回None
        """
        async with self.sem:
            try:
                if method.upper() == "GET":
                    response = await self.client.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await self.client.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response
            except Exception:
                return None

    def _calculate_accuracy(
            self,
            total_cells: int,
            incorrect_cells: int
    ) -> float:
        """
        计算准确率

        Args:
            total_cells: 总单元格数
            incorrect_cells: 错误单元格数

        Returns:
            准确率 (0.0-1.0)
        """
        if total_cells == 0:
            return 1.0

        correct_cells = total_cells - incorrect_cells
        return correct_cells / total_cells

    def _create_report(
            self,
            corrections: List[Dict[str, Any]],
            total_validated: int,
            total_rows: int
    ) -> ValidationReport:
        """从修正列表创建报告"""
        if not corrections:
            return ValidationReport(
                validated_cells=total_validated,
                corrected_cells=0,
                validated_rows=total_rows,
                corrected_rows=0,
                accuracy_before=1.0,
                accuracy_after=1.0,
                corrections=[],
                source_distribution={}
            )

        corrected_rows = len({c["row_index"] for c in corrections})
        corrected_cells = len(corrections)

        # 统计来源
        source_dist = {}
        for c in corrections:
            source = c.get("source", "Unknown")
            source_dist[source] = source_dist.get(source, 0) + 1

        # 计算准确率
        accuracy_before = self._calculate_accuracy(total_validated, corrected_cells)
        accuracy_after = 1.0  # 假设修正后100%准确

        return ValidationReport(
            validated_cells=total_validated,
            corrected_cells=corrected_cells,
            validated_rows=total_rows,
            corrected_rows=corrected_rows,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            corrections=corrections,
            source_distribution=source_dist
        )


class CompositeValidationProvider(ValidationProvider):
    """
    组合多个验证Provider

    功能：
    - 依次调用多个validator
    - 合并验证结果
    - 按优先级应用修正
    """

    name = "Composite"

    def __init__(self, providers: List[ValidationProvider]):
        super().__init__()
        self.providers = providers

    async def validate(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            validate_columns: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, ValidationReport]:
        """依次调用所有validator"""
        current_df = df
        all_corrections = []
        total_validated = 0

        for provider in self.providers:
            try:
                print(f"[Validation] Running {provider.name}...")
                current_df, report = await provider.validate(
                    current_df,
                    primary_keys,
                    validate_columns
                )
                all_corrections.extend(report.corrections)
                total_validated += report.validated_cells
            except Exception as e:
                print(f"[WARN] {provider.name} failed: {e}")
                continue

        # 合并报告
        combined_report = self._create_report(
            all_corrections,
            total_validated,
            len(df)
        )

        return current_df, combined_report

    async def aclose(self):
        """关闭所有provider"""
        for provider in self.providers:
            await provider.aclose()