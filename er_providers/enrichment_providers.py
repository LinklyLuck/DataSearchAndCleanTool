# enrichment_providers.py — LLM-driven Entity Resolution (ER) ONLY
from __future__ import annotations
from typing import List, Dict, Optional, Any
import polars as pl


# ============================================================================
# 领域自动识别
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
    3. 从query提取关键词（简单启发式）
    4. 回退到"entity"（通用）

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
                # 去除下划线，转为空格
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
        # 跳过常见的无关词
        skip_words = {"the", "a", "an", "with", "for", "about", "find", "get",
                      "show", "list", "all", "data", "dataset", "table"}

        for word in words:
            if word in skip_words:
                continue
            # 如果是名词且长度合适，可能是实体类型
            if len(word) > 3 and word.isalpha():
                # 简单处理复数形式
                singular = word.rstrip('s') if word.endswith('s') and len(word) > 4 else word
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
    使用LLM智能判断数据领域（无硬编码，完全开放）

    Args:
        query: 用户查询
        columns: 列名列表
        llm_client: LLM客户端

    Returns:
        领域名称（如 "movie", "stock", "music", "game" 等，任意领域）
    """
    # 构建提示词
    prompt = f"""Analyze the following information and determine the domain/entity type of this dataset.

User Query: {query}

Column Names: {', '.join(columns[:20])}  

Task: Based on the query and column names, determine what type of entities this dataset contains.

Requirements:
1. Return a SINGLE WORD that best describes the entity type
2. Examples: "movie", "stock", "product", "person", "music", "game", "restaurant", "book", "property", "athlete", "recipe", etc.
3. Be specific but concise
4. Use singular form (e.g., "movie" not "movies")
5. If you cannot determine the type, return "unknown"

Response format: Just return the single word, nothing else.

Domain:"""

    try:
        # 调用LLM - 修复：添加 model 参数，移除不支持的参数
        messages = [{"role": "user", "content": prompt}]
        response = await llm_client.chat("gpt-4o-mini", messages)

        # 提取领域名称
        domain = response.strip().lower()

        # 验证响应（确保是单个词）
        if domain and len(domain) < 30 and domain.replace('_', '').replace('-', '').isalpha():
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

    Note:
        所有领域都使用统一的LLM ER Provider
        LLM足够智能，可以处理任何领域的实体解析
    """
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
                from .entity_tools import LLMEntityResolver
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
            max_comparisons: int = 100,  # ← 优化：从 500 减少到 100（快 5 倍）
            confidence_threshold: float = 0.8  # ← 优化：从 0.75 提高到 0.8（更严格，更少合并）
    ) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """
        使用LLM进行实体解析（优化版：更快的默认参数）

        流程：
        1. 转换为Polars DataFrame
        2. 调用LLMEntityResolver进行ER
        3. 转回Dict列表

        性能提示：
        - max_comparisons=100: 快速模式（推荐）
        - max_comparisons=200: 平衡模式
        - max_comparisons=500: 完整模式（慢）
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
            print(f"[LLM-ER] 💡 Performance: max_comparisons={max_comparisons}, threshold={confidence_threshold}")

            resolved_df, report_df = await self._resolver.resolve(
                df=df,
                entity_type=entity_type,
                max_comparisons=max_comparisons,
                confidence_threshold=confidence_threshold
            )

            if len(resolved_df) < len(df):
                print(f"[LLM-ER] ✓ Merged {len(df)} → {len(resolved_df)} rows")
            else:
                print(f"[LLM-ER] ✓ No duplicates found (or below confidence threshold)")

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
        根据列名和主键猜测实体类型（使用通用规则，无硬编码）

        策略：
        1. 优先从主键名提取（movie_title → movie）
        2. 从任何列名提取（product_id → product）
        3. 回退到"entity"
        """
        # 方法1: 从主键提取（最可靠）
        if primary_keys and len(primary_keys) > 0:
            pk = primary_keys[0].lower()

            # 去除常见后缀
            suffixes = ["_id", "_name", "_code", "_title", "id", "name", "title", "code"]
            for suffix in suffixes:
                if pk.endswith(suffix):
                    entity_type = pk[:-len(suffix)].replace("_", " ").strip()
                    if entity_type and len(entity_type) > 2:
                        return entity_type

            # 如果主键名本身就是实体类型（且不是通用词）
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
# 辅助函数：完全通用的实体字段选择
# ============================================================================

def pick_entity_field(row: Dict[str, Any]) -> Optional[str]:
    """
    在一行数据里挑选"像实体名"的字段

    策略：
    1. 优先选择包含 "title", "name" 等后缀的字段
    2. 其次选择第一个非空字符串字段

    Args:
        row: 数据行

    Returns:
        字段名，如果找不到返回None
    """
    # 策略1: 查找带有常见实体名称后缀的字段（通用规则，非硬编码列表）
    entity_suffixes = ["_title", "_name", "title", "name"]

    for suffix in entity_suffixes:
        for key in row.keys():
            if key.lower().endswith(suffix):
                val = row[key]
                if isinstance(val, str) and val.strip():
                    return key

    # 策略2: 兜底 - 返回第一个非空字符串字段
    for key, val in row.items():
        if isinstance(val, str) and val.strip():
            return key

    return None