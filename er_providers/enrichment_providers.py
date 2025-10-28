# enrichment_providers.py — LLM-driven Entity Resolution (ER) ONLY
"""
LLM驱动的实体解析(ER)

职责：
- 实体解析(ER)：识别并合并重复实体
- MVI已迁移到 mvi_providers/

架构：
- 使用 entity_tools.LLMEntityResolver 做 ER
- 不再做 MVI（由 mvi_providers 负责）
- 支持LLM自动判断领域（完全开放，不硬编码）
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any
import polars as pl
import json


# ============================================================================
# 领域自动识别（使用LLM，完全开放）
# ============================================================================

async def detect_domain_from_query_or_columns(
        query: str,
        columns: List[str],
        llm_client=None
) -> str:
    """
    自动识别数据领域 - 使用LLM判断，完全开放

    策略（按优先级）：
    1. 从列名中提取实体类型（如 "movie_title" → "movie"）
    2. 使用LLM分析列名和查询，推断领域
    3. 回退到"entity"（通用）

    Args:
        query: 用户查询
        columns: 数据集列名
        llm_client: LLM客户端（用于智能判断）

    Returns:
        领域名称: 自动推断的实体类型（完全开放，不限于预设领域）
    """
    # ========================================================================
    # 方法1: 从列名提取（最快，优先级最高）
    # ========================================================================
    for col in columns:
        col_lower = col.lower()
        # 常见实体类型后缀
        for suffix in ["_title", "_name", "_id", "_code"]:
            if suffix in col_lower:
                # 提取前缀作为实体类型
                entity_type = col_lower.split(suffix)[0]
                # 去除常见的修饰词
                entity_type = entity_type.replace("_", " ").strip()
                if entity_type and len(entity_type) > 2:
                    print(f"[DOMAIN] Detected from column name: {entity_type}")
                    return entity_type

    # ========================================================================
    # 方法2: 使用LLM智能判断（完全开放，不限制领域）
    # ========================================================================
    if llm_client:
        try:
            llm_domain = await _llm_detect_domain(query, columns, llm_client)
            if llm_domain and llm_domain != "unknown":
                print(f"[DOMAIN] Detected by LLM: {llm_domain}")
                return llm_domain
        except Exception as e:
            print(f"[WARN] LLM domain detection failed: {e}")
            # 继续使用启发式方法

    # ========================================================================
    # 方法3: 从query提取关键词（简单启发式）
    # ========================================================================
    if query:
        query_lower = query.lower()
        # 从query中提取第一个可能的实体类型词
        words = query_lower.split()
        for word in words:
            # 过滤掉常见的无关词
            if word in ["the", "a", "an", "with", "for", "about", "find", "get", "show"]:
                continue
            # 如果是名词且长度合适，可能是实体类型
            if len(word) > 3 and word.isalpha():
                # 简单检查是否是复数形式
                singular = word.rstrip('s') if word.endswith('s') else word
                if len(singular) > 3:
                    print(f"[DOMAIN] Detected from query: {singular}")
                    return singular

    # ========================================================================
    # 方法4: 回退到通用实体
    # ========================================================================
    print(f"[DOMAIN] Using default: entity")
    return "entity"


async def _llm_detect_domain(
        query: str,
        columns: List[str],
        llm_client
) -> Optional[str]:
    """
    使用LLM智能判断数据领域

    Args:
        query: 用户查询
        columns: 列名列表
        llm_client: LLM客户端

    Returns:
        领域名称（如 "movie", "stock", "music", "game" 等）
    """
    # 构建提示词
    prompt = f"""Analyze the following information and determine the domain/entity type of this dataset.

User Query: {query}

Column Names: {', '.join(columns[:20])}  

Task: Based on the query and column names, determine what type of entities this dataset contains.

Requirements:
1. Return a SINGLE WORD that best describes the entity type (e.g., "movie", "stock", "product", "person", "music", "game", "restaurant", "book", etc.)
2. Be specific but concise
3. Use singular form (e.g., "movie" not "movies")
4. If you cannot determine the type, return "unknown"

Response format: Just return the single word, nothing else.

Domain:"""

    try:
        # 调用LLM
        messages = [{"role": "user", "content": prompt}]
        response = await llm_client.chat(messages, max_tokens=20, temperature=0)

        # 提取领域名称
        domain = response.strip().lower()

        # 验证响应
        if domain and len(domain) < 30 and domain.isalpha():
            return domain
        else:
            return None

    except Exception as e:
        print(f"[WARN] LLM domain detection error: {e}")
        return None


def get_provider_for_auto_domain(domain: str, llm_client=None) -> "ERProvider":
    """
    根据领域返回对应的ER Provider

    Args:
        domain: 领域名称（任意，不限制）
        llm_client: LLM客户端

    Returns:
        ERProvider实例（只做ER，不做MVI）
    """
    # 所有领域都使用统一的LLM ER Provider
    # LLM足够智能，可以处理任何领域的实体解析
    return LLMERProvider(llm_client=llm_client)


# ============================================================================
# LLM驱动的ER Provider（只做实体解析，不做MVI）
# ============================================================================

class ERProvider:
    """基类：定义ER标准接口"""
    name = "BaseERProvider"

    async def resolve_entities(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            entity_type: str = "entity",
            max_comparisons: int = 500,
            confidence_threshold: float = 0.75
    ) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        实体解析（Entity Resolution）

        Args:
            rows: 原始数据行
            primary_keys: 主键列（用于实体识别）
            entity_type: 实体类型（用于LLM理解）
            max_comparisons: 最大LLM比较次数
            confidence_threshold: 合并阈值

        Returns:
            (resolved_rows, report): 解析后的数据行和报告
        """
        return rows, None


class LLMERProvider(ERProvider):
    """
    LLM驱动的实体解析Provider

    功能：
    - 使用LLM判断重复实体并合并
    - 支持任意领域（不限制）
    - 不做MVI（由 mvi_providers 负责）
    """
    name = "LLM-ER"

    def __init__(self, llm_client=None):
        self._resolver = None
        self._llm_client = llm_client

    async def _ensure_resolver(self, llm_client):
        """延迟初始化resolver（需要llm_client）"""
        if self._resolver is None and llm_client is not None:
            try:
                from entity_tools import LLMEntityResolver
                self._resolver = LLMEntityResolver(llm_client)
                self._llm_client = llm_client
            except ImportError as e:
                print(f"[WARN] Cannot import LLMEntityResolver: {e}")
                self._resolver = None

    async def resolve_entities(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            entity_type: str = "entity",
            max_comparisons: int = 500,
            confidence_threshold: float = 0.75
    ) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        使用LLM进行实体解析

        流程：
        1. 转换为Polars DataFrame
        2. 调用LLMEntityResolver进行ER
        3. 转回Dict列表
        """
        if not rows:
            return rows, None

        # 获取LLM client
        llm_client = self._llm_client
        if llm_client is None:
            print("[WARN] No LLM client available, skipping ER")
            return rows, None

        await self._ensure_resolver(llm_client)

        if self._resolver is None:
            print("[WARN] LLMEntityResolver not available, skipping ER")
            return rows, None

        try:
            # 1. 转换为Polars DataFrame
            df = pl.from_dicts(rows)

            # 2. 如果entity_type是generic的，尝试从数据推断
            if not entity_type or entity_type == "entity":
                entity_type = self._guess_entity_type(df.columns, primary_keys)

            # 3. 执行ER（实体解析 - 合并重复实体）
            print(f"[LLM-ER] Running entity resolution for '{entity_type}' entities...")

            resolved_df, report_df = await self._resolver.resolve(
                df=df,
                entity_type=entity_type,
                max_comparisons=max_comparisons,
                confidence_threshold=confidence_threshold
            )

            if len(resolved_df) < len(df):
                print(f"[LLM-ER] ✓ Merged {len(df)} → {len(resolved_df)} rows")

            # 4. 转回Dict
            resolved_rows = resolved_df.to_dicts()
            report = report_df.to_dicts() if report_df is not None else None

            return resolved_rows, report

        except Exception as e:
            print(f"[ERROR] LLM ER failed: {e}")
            import traceback
            traceback.print_exc()
            return rows, None

    def _guess_entity_type(self, columns: List[str], primary_keys: List[str]) -> str:
        """
        根据列名和主键猜测实体类型（不使用硬编码列表）

        策略：
        1. 优先从主键名提取（movie_title → movie）
        2. 从任何列名提取（product_id → product）
        3. 回退到"entity"
        """
        # 方法1: 从主键提取（最可靠）
        if primary_keys and len(primary_keys) > 0:
            pk = primary_keys[0].lower()
            # 去除常见后缀
            for suffix in ["_id", "_name", "_code", "_title", "id", "name", "title", "code"]:
                if pk.endswith(suffix):
                    entity_type = pk[:-len(suffix)].replace("_", " ").strip()
                    if entity_type and len(entity_type) > 2:
                        return entity_type
            # 如果主键名本身就是实体类型
            if pk and len(pk) > 2 and pk not in ["id", "key", "index"]:
                return pk

        # 方法2: 从任何列名提取
        for col in columns:
            col_lower = col.lower()
            for suffix in ["_title", "_name", "_id", "_code"]:
                if suffix in col_lower:
                    entity_type = col_lower.split(suffix)[0].replace("_", " ").strip()
                    if entity_type and len(entity_type) > 2:
                        return entity_type

        # 回退：通用实体
        return "entity"


# ============================================================================
# 辅助函数：兼容性
# ============================================================================

def pick_entity_field(row: Dict[str, Any]) -> Optional[str]:
    """
    在一行数据里挑选"像实体名"的字段
    """
    candidates = [
        "movie_title", "title", "name", "entity_name", "product_name",
        "ticker", "director_name", "company", "brand", "player_name"
    ]
    for k in candidates:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return k

    # 兜底：第一个字符串列
    for k, v in row.items():
        if isinstance(v, str) and v.strip():
            return k
    return None