# enrichment_providers.py — LLM-driven ER/MVI
"""
LLM驱动的实体解析(ER)和缺失值填补(MVI)
- 零样本：无需训练数据
- 跨领域：自动识别领域并处理
- 语义理解：使用LLM判断实体相同性
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any
import json
import polars as pl


# ============================================================================
# 领域自动识别（保持原有接口兼容）
# ============================================================================

async def detect_domain_from_query_or_columns(
        query: str,
        columns: List[str],
        llm_client=None
) -> str:
    """
    自动识别数据领域 - 完全开放，不硬编码

    策略：
    1. 从列名中提取实体类型（如 "movie_title" → "movie"）
    2. 从query中提取关键词
    3. 回退到"entity"（通用）

    Args:
        query: 用户查询
        columns: 数据集列名
        llm_client: LLM客户端（可选，用于更精准识别）

    Returns:
        领域名称: 自动推断的实体类型（完全开放，不限于预设领域）
    """
    # 方法1: 从列名提取（优先级最高）
    # 例如: movie_title → movie, product_id → product, paper_name → paper
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
                    return entity_type

    # 方法2: 从query提取关键词
    if query:
        query_lower = query.lower()
        # 提取第一个名词（简单启发式）
        words = query_lower.split()
        for word in words:
            if word in ["movie", "film", "stock", "product", "paper", "book",
                        "patient", "customer", "user", "employee", "company",
                        "restaurant", "hotel", "car", "house", "article"]:
                return word

    # 方法3: 从列名中寻找主题词
    cols_str = " ".join([c.lower() for c in columns])
    # 常见主题词
    topics = {
        "movie": ["movie", "film", "director", "actor", "imdb"],
        "stock": ["stock", "ticker", "equity", "price", "volume"],
        "product": ["product", "sku", "order", "cart", "price"],
        "paper": ["paper", "author", "conference", "journal", "citation"],
        "patient": ["patient", "doctor", "hospital", "diagnosis", "treatment"],
        "restaurant": ["restaurant", "cuisine", "menu", "chef", "rating"]
    }

    for topic, keywords in topics.items():
        if any(k in cols_str for k in keywords):
            return topic

    # 回退：使用通用名称
    return "entity"


def get_provider_for_auto_domain(domain: str, llm_client=None) -> "EnrichProvider":
    """
    根据领域返回对应的Provider

    Args:
        domain: 领域名称
        llm_client: LLM客户端（可选，用于LLM驱动的ER/MVI）

    Returns:
        EnrichProvider实例
    """
    # 现在所有领域都使用统一的LLM Provider
    return LLMEnrichProvider(llm_client=llm_client)


# ============================================================================
# LLM驱动的通用Provider
# ============================================================================

class EnrichProvider:
    """基类：定义标准接口"""
    name = "BaseProvider"

    async def enrich_rows(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            allowed_update_cols: List[str],
            enrich_only: bool = True,
            no_title_change: bool = True
    ) -> List[Dict[str, Any]]:
        """
        增强数据行（ER + MVI）

        Args:
            rows: 原始数据行
            primary_keys: 主键列（用于实体识别）
            allowed_update_cols: 允许更新的列（只更新这些列的空值）
            enrich_only: 是否只填补（不改变原有值）
            no_title_change: 是否禁止改变主键值

        Returns:
            增强后的数据行
        """
        return rows


class LLMEnrichProvider(EnrichProvider):
    """
    LLM驱动的通用增强Provider

    功能：
    1. 实体解析(ER)：用LLM判断重复实体并合并
    2. 缺失值填补(MVI)：用LLM填补空值
    """
    name = "LLM-ER+MVI"

    def __init__(self, llm_client=None):
        self._resolver = None
        self._llm_client = llm_client

    async def _ensure_resolver(self, llm_client):
        """延迟初始化resolver（需要llm_client）"""
        if self._resolver is None and llm_client is not None:
            # 导入LLMEntityResolver（避免循环依赖）
            try:
                from entity_tools import LLMEntityResolver
                self._resolver = LLMEntityResolver(llm_client)
                self._llm_client = llm_client
            except ImportError as e:
                print(f"[WARN] Cannot import LLMEntityResolver: {e}")
                self._resolver = None

    async def enrich_rows(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            allowed_update_cols: List[str],
            enrich_only: bool = True,
            no_title_change: bool = True
    ) -> List[Dict[str, Any]]:
        """
        使用LLM进行ER和MVI

        流程：
        1. 转换为Polars DataFrame
        2. 调用LLMEntityResolver进行ER
        3. 调用LLM进行MVI（填补空值）
        4. 转回Dict列表
        """
        if not rows:
            return rows

        # 获取LLM client（从全局或环境）
        llm_client = await self._get_llm_client()
        if llm_client is None:
            print("[WARN] No LLM client available, skipping ER/MVI")
            return rows

        await self._ensure_resolver(llm_client)

        try:
            # 1. 转换为Polars DataFrame
            df = pl.from_dicts(rows)

            # 2. 识别实体类型（用于LLM理解）
            entity_type = self._guess_entity_type(df.columns, primary_keys)

            # 3. 执行ER（实体解析 - 合并重复实体）
            if self._resolver is not None:
                print(f"[LLM-ER] Running entity resolution for {entity_type}...")
                try:
                    # 限制比较次数（避免成本过高）
                    max_comparisons = min(500, len(df) * 5)

                    resolved_df, report_df = await self._resolver.resolve(
                        df=df,
                        entity_type=entity_type,
                        max_comparisons=max_comparisons,
                        confidence_threshold=0.75  # 稍微宽松一点
                    )

                    if len(resolved_df) < len(df):
                        print(f"[LLM-ER] Merged {len(df)} → {len(resolved_df)} rows")
                        df = resolved_df
                except Exception as e:
                    print(f"[WARN] ER failed: {e}, continuing without ER")

            # 4. 执行MVI（缺失值填补）
            df = await self._llm_impute_missing(
                df,
                allowed_update_cols,
                entity_type,
                llm_client
            )

            # 5. 转回Dict
            enriched_rows = df.to_dicts()

            # 6. 确保只更新allowed_update_cols，且只填空值
            result = []
            for i, original in enumerate(rows):
                if i < len(enriched_rows):
                    enriched = enriched_rows[i]
                    updated = dict(original)

                    for col in allowed_update_cols:
                        original_val = original.get(col)
                        enriched_val = enriched.get(col)

                        # 只在原值为空且新值非空时更新
                        if (original_val is None or str(original_val).strip() == "") and \
                                (enriched_val is not None and str(enriched_val).strip() != ""):
                            updated[col] = enriched_val

                    # 如果no_title_change，确保主键不变
                    if no_title_change:
                        for pk in primary_keys:
                            if pk in original:
                                updated[pk] = original[pk]

                    result.append(updated)
                else:
                    result.append(original)

            return result

        except Exception as e:
            print(f"[ERROR] LLM enrichment failed: {e}")
            import traceback
            traceback.print_exc()
            return rows

    def _guess_entity_type(self, columns: List[str], primary_keys: List[str]) -> str:
        """
        根据列名和主键猜测实体类型 - 完全开放，不硬编码

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

        # 方法3: 智能猜测（启发式，但开放式）
        cols_str = " ".join([c.lower() for c in columns])

        # 这里保留一些常见类型的识别，但作为启发式，不是限制
        common_patterns = {
            "movie": ["movie", "film", "director", "imdb", "runtime", "genre"],
            "stock": ["ticker", "ohlc", "close", "open", "volume", "exchange"],
            "product": ["sku", "price", "quantity", "cart", "order"],
            "paper": ["author", "conference", "journal", "citation", "abstract"],
            "patient": ["patient", "diagnosis", "treatment", "hospital"],
            "athlete": ["player", "team", "sport", "game", "score"],
            "restaurant": ["restaurant", "cuisine", "menu", "chef"],
            "book": ["book", "author", "publisher", "isbn"],
            "company": ["company", "ceo", "revenue", "employees", "industry"]
        }

        for entity_type, keywords in common_patterns.items():
            if any(k in cols_str for k in keywords):
                return entity_type

        # 回退：通用实体
        return "entity"

    async def _llm_impute_missing(
            self,
            df: pl.DataFrame,
            allowed_cols: List[str],
            entity_type: str,
            llm_client
    ) -> pl.DataFrame:
        """
        使用LLM填补缺失值

        策略：
        1. 对每个有缺失的列，选择几个有值的行作为示例
        2. 让LLM根据示例和实体信息推测缺失值
        3. 只填补置信度高的值
        """
        if llm_client is None:
            return df

        # 限制填补的行数（避免成本过高）
        max_rows_to_fill = 50

        for col in allowed_cols:
            if col not in df.columns:
                continue

            # 找出该列的缺失行
            null_mask = df[col].is_null() | (df[col].cast(pl.Utf8) == "")
            null_indices = df.select(pl.arange(0, pl.count()).alias("idx")).filter(null_mask)["idx"].to_list()

            if not null_indices or len(null_indices) == 0:
                continue

            # 限制填补数量
            null_indices = null_indices[:max_rows_to_fill]

            # 获取有值的示例行
            non_null_df = df.filter(~null_mask).head(5)
            if len(non_null_df) == 0:
                continue

            examples = non_null_df.to_dicts()

            # 批量处理缺失行
            print(f"[LLM-MVI] Imputing {len(null_indices)} values for column '{col}'...")

            for idx in null_indices[:10]:  # 进一步限制，避免API调用过多
                try:
                    row = df.row(idx, named=True)

                    # 构建prompt
                    prompt = self._build_impute_prompt(
                        entity_type, col, row, examples
                    )

                    # 调用LLM
                    response = await llm_client.chat("gpt-4o-mini", [
                        {"role": "user", "content": prompt}
                    ])

                    # 解析响应
                    result = self._parse_impute_response(response)

                    if result and result.get("value") and result.get("confidence", 0) > 0.7:
                        # 更新值
                        new_val = result["value"]
                        df = df.with_columns([
                            pl.when(pl.arange(0, pl.count()) == idx)
                            .then(pl.lit(new_val))
                            .otherwise(pl.col(col))
                            .alias(col)
                        ])

                except Exception as e:
                    print(f"[WARN] Failed to impute row {idx}, col {col}: {e}")
                    continue

        return df

    def _build_impute_prompt(
            self,
            entity_type: str,
            column: str,
            row: Dict[str, Any],
            examples: List[Dict[str, Any]]
    ) -> str:
        """构建MVI的prompt"""

        # 清理row（只保留非空字段）
        row_clean = {k: v for k, v in row.items() if v is not None and str(v).strip() != ""}

        prompt = f"""You are filling missing data for a {entity_type} dataset.

Column to fill: "{column}"

Current record (missing {column}):
{json.dumps(row_clean, indent=2)}

Example records with {column} values:
{json.dumps(examples[:3], indent=2)}

Based on the examples and the current record, what should the value of "{column}" be?

Respond in JSON:
{{
    "value": "your prediction",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Rules:
- Only predict if you are confident (>0.7)
- Return null if uncertain
- Keep the same data type as examples
- Be conservative"""

        return prompt

    def _parse_impute_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM的MVI响应"""
        try:
            # 尝试提取JSON
            response = response.strip()

            # 如果有markdown代码块，提取出来
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            result = json.loads(response)
            return result
        except Exception:
            return None

    async def _get_llm_client(self):
        """获取LLM客户端"""
        return self._llm_client


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