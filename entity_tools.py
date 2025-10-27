# llm_entity_resolver.py

from typing import List, Dict, Any, Tuple
import polars as pl
import asyncio
import json
from difflib import SequenceMatcher


class LLMEntityResolver:
    """
    LLM驱动的通用实体解析器

    核心创新:
    1. 零样本: 无需训练数据
    2. 跨领域: 适用于任何实体类型
    3. 语义理解: 不只是字符串匹配
    """

    def __init__(self, llm_client, cache=None):
        self.llm = llm_client
        self.cache = cache or {}

    async def resolve(
            self,
            df: pl.DataFrame,
            entity_type: str = "entity",
            max_comparisons: int = 500,
            confidence_threshold: float = 0.8
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        主入口: 对数据集进行实体解析

        Args:
            df: 输入数据
            entity_type: 实体类型描述 (如 "movie", "product")
            max_comparisons: 最大LLM比较次数
            confidence_threshold: 合并阈值

        Returns:
            (resolved_df, report_df)
        """

        print(f"[LLM-ER] Resolving {entity_type} entities...")

        # Step 1: 识别关键列
        print("[LLM-ER] Step 1: Identifying key columns...")
        key_cols = await self.identify_key_columns(df, entity_type)
        print(f"[LLM-ER] Key columns: {key_cols}")

        # Step 2: 候选对生成（启发式）
        print("[LLM-ER] Step 2: Generating candidate pairs...")
        candidates = self.generate_candidates(df, key_cols, max_comparisons)
        print(f"[LLM-ER] Found {len(candidates)} candidate pairs")

        # Step 3: LLM判断
        print("[LLM-ER] Step 3: LLM-based comparison...")
        merge_graph = await self.compare_pairs(
            df, candidates, entity_type, confidence_threshold
        )

        # Step 4: 合并
        print("[LLM-ER] Step 4: Merging entities...")
        resolved_df = self.merge_entities(df, merge_graph)

        # Step 5: 生成报告
        report_df = self.generate_report(df, resolved_df, merge_graph)

        return resolved_df, report_df

    async def identify_key_columns(
            self,
            df: pl.DataFrame,
            entity_type: str
    ) -> List[str]:
        """
        让LLM识别哪些列是关键标识列
        """

        # 缓存key
        cache_key = f"key_cols_{hash(str(df.columns))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 准备schema信息
        schema_info = {}
        for col in df.columns[:20]:  # 限制列数
            try:
                schema_info[col] = {
                    "dtype": str(df[col].dtype),
                    "sample": df[col].head(3).drop_nulls().to_list()[:3],
                    "null_pct": round(df[col].null_count() / len(df) * 100, 1)
                }
            except:
                continue

        prompt = f"""You are analyzing a dataset of {entity_type} records.

Schema:
{json.dumps(schema_info, indent=2)}

Identify 1-3 columns that are KEY IDENTIFIERS for determining if two records are the same {entity_type}.

Key columns typically:
- Uniquely identify entities (names, titles, IDs)
- Have low null percentage
- Are string or categorical types

Return ONLY a JSON array of column names:
["col1", "col2"]"""

        try:
            response = await self.llm.chat("gpt-4o-mini", [
                {"role": "user", "content": prompt}
            ])

            key_cols = json.loads(response.strip())

            # 验证列名
            key_cols = [c for c in key_cols if c in df.columns]

            if not key_cols:
                # 回退: 选择第一个字符串列
                key_cols = [c for c in df.columns
                            if df[c].dtype == pl.Utf8][:1]

            self.cache[cache_key] = key_cols
            return key_cols

        except Exception as e:
            print(f"[LLM-ER] Error in identify_key_columns: {e}")
            # 回退策略
            return [c for c in df.columns if df[c].dtype == pl.Utf8][:1]

    def generate_candidates(
            self,
            df: pl.DataFrame,
            key_cols: List[str],
            max_pairs: int
    ) -> List[Tuple[int, int]]:
        """
        生成候选重复对

        策略:
        1. Blocking by first character
        2. 组内简单相似度过滤
        3. 限制总数
        """

        if not key_cols:
            return []

        candidates = []
        primary_col = key_cols[0]

        # Blocking: 按首字母分组
        try:
            # 添加首字母列
            df_with_block = df.with_columns([
                pl.col(primary_col)
                .str.slice(0, 1)
                .str.to_lowercase()
                .alias("_block_key")
            ])

            # 按block分组
            for block_key, group_df in df_with_block.group_by("_block_key"):
                if len(group_df) <= 1:
                    continue

                indices = group_df.select(pl.arange(0, pl.count()).alias("idx"))["idx"].to_list()

                # 组内两两比较（限制数量）
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i + 1:i + 11]:  # 最多10个
                        # 简单预筛选
                        val1 = str(df[primary_col][idx1] or "")
                        val2 = str(df[primary_col][idx2] or "")

                        if self.quick_similarity(val1, val2) > 0.5:
                            candidates.append((idx1, idx2))

                        if len(candidates) >= max_pairs:
                            return candidates[:max_pairs]

        except Exception as e:
            print(f"[LLM-ER] Error in generate_candidates: {e}")
            # 回退: 随机采样
            n = min(len(df), 100)
            for i in range(n):
                for j in range(i + 1, min(i + 6, n)):
                    candidates.append((i, j))

        return candidates[:max_pairs]

    def quick_similarity(self, s1: str, s2: str) -> float:
        """快速相似度（用于预筛选）"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    async def compare_pairs(
            self,
            df: pl.DataFrame,
            candidates: List[Tuple[int, int]],
            entity_type: str,
            threshold: float
    ) -> Dict[int, int]:
        """
        用LLM比较候选对

        返回: merge_graph {from_idx: to_idx}
        """

        merge_graph = {}

        # 批量处理（减少API调用）
        batch_size = 10
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            tasks = [
                self.are_same_entity(
                    df.row(idx1, named=True),
                    df.row(idx2, named=True),
                    entity_type
                )
                for idx1, idx2 in batch
            ]

            results = await asyncio.gather(*tasks)

            for (idx1, idx2), (same, confidence) in zip(batch, results):
                if same and confidence >= threshold:
                    # Union-Find: 合并到同一个cluster
                    root1 = self.find_root(merge_graph, idx1)
                    root2 = self.find_root(merge_graph, idx2)
                    if root1 != root2:
                        merge_graph[root2] = root1

        return merge_graph

    def find_root(self, merge_graph: Dict[int, int], idx: int) -> int:
        """Union-Find: 找根节点"""
        if idx not in merge_graph:
            return idx
        root = idx
        while root in merge_graph:
            root = merge_graph[root]
        # 路径压缩
        while idx != root:
            next_idx = merge_graph[idx]
            merge_graph[idx] = root
            idx = next_idx
        return root

    async def are_same_entity(
            self,
            entity1: Dict[str, Any],
            entity2: Dict[str, Any],
            entity_type: str
    ) -> Tuple[bool, float]:
        """
        核心: 用LLM判断两个实体是否相同
        """

        # 缓存
        cache_key = f"{hash(str(sorted(entity1.items())))}_{hash(str(sorted(entity2.items())))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 简化显示（只显示非空字段）
        e1_clean = {k: v for k, v in entity1.items() if v is not None}
        e2_clean = {k: v for k, v in entity2.items() if v is not None}

        prompt = f"""Are these two {entity_type} records the same entity?

Record 1: {json.dumps(e1_clean)}
Record 2: {json.dumps(e2_clean)}

Consider variations like:
- Typos/spelling
- Missing words
- Abbreviations  
- Different formats

Respond in JSON:
{{"same": true/false, "confidence": 0.0-1.0}}"""

        try:
            response = await self.llm.chat("gpt-4o-mini", [
                {"role": "user", "content": prompt}
            ])

            result = json.loads(response.strip())
            same = result["same"]
            confidence = result["confidence"]

            self.cache[cache_key] = (same, confidence)
            return same, confidence

        except Exception as e:
            print(f"[LLM-ER] Error in are_same_entity: {e}")
            # 回退: 保守策略
            return False, 0.0

    def merge_entities(
            self,
            df: pl.DataFrame,
            merge_graph: Dict[int, int]
    ) -> pl.DataFrame:
        """
        应用合并
        """

        if not merge_graph:
            return df

        # 为每行分配cluster ID
        cluster_ids = []
        for i in range(len(df)):
            cluster_ids.append(self.find_root(merge_graph, i))

        df_with_cluster = df.with_columns([
            pl.Series("_cluster_id", cluster_ids)
        ])

        # 按cluster聚合
        # 策略: 数值取平均，文本取首个非空
        agg_exprs = []
        for col in df.columns:
            if df[col].dtype.is_numeric():
                agg_exprs.append(pl.col(col).mean().alias(col))
            else:
                agg_exprs.append(
                    pl.col(col).filter(pl.col(col).is_not_null()).first().alias(col)
                )

        resolved = df_with_cluster.group_by("_cluster_id").agg(agg_exprs)
        resolved = resolved.drop("_cluster_id")

        return resolved

    def generate_report(
            self,
            original_df: pl.DataFrame,
            resolved_df: pl.DataFrame,
            merge_graph: Dict[int, int]
    ) -> pl.DataFrame:
        """
        生成ER报告
        """

        # 统计每个cluster合并了多少行
        cluster_sizes = {}
        for idx in range(len(original_df)):
            root = self.find_root(merge_graph, idx)
            cluster_sizes[root] = cluster_sizes.get(root, 0) + 1

        report_data = []
        for cluster_id, size in cluster_sizes.items():
            if size > 1:  # 只报告实际合并的
                report_data.append({
                    "cluster_id": cluster_id,
                    "records_merged": size
                })

        if not report_data:
            return pl.DataFrame({
                "message": ["No duplicates found"]
            })

        report = pl.DataFrame(report_data).sort("records_merged", descending=True)
        return report


# ============= 使用示例 =============

async def example_usage():
    """演示如何使用"""

    # 1. 准备数据
    df = pl.DataFrame({
        "movie_title": ["The Dark Knight", "Dark Knight", "Inception", "Inception", "Interstellar"],
        "director": ["Nolan", "Nolan", "Nolan", "Christopher Nolan", "C. Nolan"],
        "year": [2008, 2008, 2010, 2010, 2014],
        "budget": [185, None, 160, 160, 165]
    })

    # 2. 初始化resolver
    from your_llm_client import LLMClient
    llm = LLMClient(api_key="...")
    resolver = LLMEntityResolver(llm)

    # 3. 执行ER
    resolved_df, report_df = await resolver.resolve(
        df,
        entity_type="movie",
        max_comparisons=100,
        confidence_threshold=0.8
    )

    print("Original:", len(df), "rows")
    print("Resolved:", len(resolved_df), "rows")
    print("\nReport:")
    print(report_df)

    # 预期输出:
    # Original: 5 rows
    # Resolved: 3 rows  (合并了Dark Knight和Inception的重复)

    return resolved_df, report_df


if __name__ == "__main__":
    asyncio.run(example_usage())