# mvi_providers/base.py
"""
Missing Value Imputation (MVI) Provider Base Class

专注于缺失值填补，不做实体解析(ER)
ER已由 entity_tools.py 和 enrichment_providers.py 通过LLM完成
"""
from __future__ import annotations
import asyncio
from typing import List, Dict, Optional, Any
import httpx


class MVIProvider:
    """
    MVI Provider 基类

    职责：
    - 只负责填补缺失值（Missing Value Imputation）
    - 基于外部API获取数据
    - 不做实体解析（ER由LLM负责）

    子类需要实现：
    - impute_missing(): 填补缺失值的具体逻辑
    """

    name = "BaseMVIProvider"

    def __init__(
            self,
            timeout: int = 20,
            max_concurrency: int = 6,
            api_key: Optional[str] = None
    ):
        """
        Args:
            timeout: HTTP请求超时时间（秒）
            max_concurrency: 最大并发请求数
            api_key: API密钥（如需要）
        """
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.api_key = api_key
        self.sem = asyncio.Semaphore(max_concurrency)
        self.client = httpx.AsyncClient(timeout=timeout)
        self._cache = {}  # 简单缓存避免重复请求

    async def aclose(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

    async def impute_missing(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            allowed_update_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        填补缺失值（核心接口）

        Args:
            rows: 数据行
            primary_keys: 主键列（用于查询外部API）
            allowed_update_cols: 允许更新的列（只填这些列的空值）

        Returns:
            填补后的数据行

        规则：
        - 只更新 allowed_update_cols 中的列
        - 只填补空值（None 或 空字符串）
        - 保持主键不变
        - 如果API查询失败，保持原值
        """
        raise NotImplementedError("Subclass must implement impute_missing()")

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
            **kwargs: 其他参数（params, headers等）

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
            except Exception as e:
                # 静默失败，不中断整个流程
                return None

    def _merge_with_original(
            self,
            original: Dict[str, Any],
            enriched: Dict[str, Any],
            allowed_cols: List[str],
            primary_keys: List[str]
    ) -> Dict[str, Any]:
        """
        合并原始行和填补后的行

        规则：
        - 只更新 allowed_cols 中的列
        - 只在原值为空时更新
        - 主键保持不变
        """
        result = dict(original)

        for col in allowed_cols:
            if col in enriched:
                original_val = original.get(col)
                enriched_val = enriched.get(col)

                # 只在原值为空且新值非空时更新
                if self._is_missing(original_val) and not self._is_missing(enriched_val):
                    result[col] = enriched_val

        # 确保主键不变
        for pk in primary_keys:
            if pk in original:
                result[pk] = original[pk]

        return result


class CompositeMVIProvider(MVIProvider):
    """
    组合多个MVI Provider

    功能：
    - 依次调用多个provider
    - 累积填补缺失值
    - 如果前一个provider已填补某列，后续provider跳过
    """

    name = "Composite"

    def __init__(self, providers: List[MVIProvider]):
        """
        Args:
            providers: MVI provider列表，按优先级排序
        """
        super().__init__()
        self.providers = providers

    async def impute_missing(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            allowed_update_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        依次调用所有provider填补缺失值
        """
        current_rows = rows

        for provider in self.providers:
            try:
                print(f"[MVI] Running {provider.name}...")
                current_rows = await provider.impute_missing(
                    current_rows,
                    primary_keys,
                    allowed_update_cols
                )
            except Exception as e:
                print(f"[WARN] {provider.name} failed: {e}")
                continue

        return current_rows

    async def aclose(self):
        """关闭所有provider"""
        for provider in self.providers:
            await provider.aclose()