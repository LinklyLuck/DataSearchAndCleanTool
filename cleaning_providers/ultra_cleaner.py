# cleaning_providers/ultra_cleaner.py
"""
1) 规则优先（fast_cleaner）——向量化覆盖 80% 问题
2) 列选择器 —— 只对“可疑列”进行 LLM 清洗
3) 去重 + 频次优先 —— 仅对唯一值进行一次清洗，按出现频次排序
4) 微批 + 并发 —— 将多个唯一值打包给 LLM，并发多个微批
5) 持久化缓存（SQLite）——跨任务/多次运行复用清洗映射
6) 可选聚类（rapidfuzz）——相似值只清洗簇代表（开关可控）
"""

from __future__ import annotations
import os, re, json, math, sqlite3, asyncio, hashlib
from typing import List, Dict, Optional, Any, Tuple, Iterable
from pathlib import Path
import polars as pl

try:
    from rapidfuzz.clustering import cluster
    from rapidfuzz.distance import Levenshtein
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

from .base import CleaningProvider, CleaningReport
from .fast_cleaner import FastCleaningProvider


def _sha_key(col: str, raw: str, version: str = "v1") -> str:
    h = hashlib.sha256()
    h.update((version + "||" + col + "||" + (raw or "")).encode("utf-8", "ignore"))
    return h.hexdigest()


class _KVCache:
    """SQLite 持久化缓存: (col, key_hash) -> cleaned_text"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS clean_map (
                key_hash TEXT PRIMARY KEY,
                col_name TEXT,
                raw_text TEXT,
                cleaned_text TEXT,
                version TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_col ON clean_map(col_name);")

    def get_many(self, col: str, raws: Iterable[str], version: str = "v1") -> Dict[str, str]:
        keys = [(r, _sha_key(col, r, version)) for r in raws]
        if not keys:
            return {}
        with sqlite3.connect(self.db_path) as conn:
            qmarks = ",".join(["?"] * len(keys))
            rows = conn.execute(
                f"SELECT key_hash, cleaned_text FROM clean_map WHERE key_hash IN ({qmarks})",
                [k for _, k in keys]
            ).fetchall()
        kv = {k: v for (k, v) in rows}
        out = {}
        for raw, k in keys:
            if k in kv:
                out[raw] = kv[k]
        return out

    def put_many(self, col: str, mapping: Dict[str, str], version: str = "v1"):
        if not mapping:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO clean_map (key_hash, col_name, raw_text, cleaned_text, version) VALUES (?, ?, ?, ?, ?)",
                [(_sha_key(col, r, version), col, r, c, version) for r, c in mapping.items()]
            )
            conn.commit()


class UltraCleaningProvider(CleaningProvider):

    name = "Ultra-Cleaning"

    def __init__(
        self,
        llm_client,
        value_batch_size: int = 24,
        max_concurrency: int = 6,
        cache_db: str = ".cache/clean_cache.sqlite",
        use_clustering: bool = False,
        cluster_threshold: int = 90,  # 0-100
        llm_budget_ratio: float = 0.005,  # 只处理 Top 0.5% 唯一值
        version_tag: str = "v1"
    ):
        super().__init__()
        self.llm = llm_client
        self.value_batch_size = value_batch_size
        self.sem = asyncio.Semaphore(max_concurrency)
        self.cache = _KVCache(cache_db)
        self.use_clustering = use_clustering and _HAS_RAPIDFUZZ
        self.cluster_threshold = cluster_threshold
        self.llm_budget_ratio = llm_budget_ratio
        self.version_tag = version_tag
        self.fast = FastCleaningProvider()

    async def clean(
        self,
        df: pl.DataFrame,
        primary_keys: List[str],
        rules: Optional[Dict[str, Any]] = None
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        # 允许通过 rules 动态覆盖参数
        if rules:
            self.value_batch_size = int(rules.get('value_batch_size', self.value_batch_size))
            self.sem = asyncio.Semaphore(int(rules.get('max_concurrency', self.sem._value)))
            self.cache = _KVCache(rules.get('cache_db', self.cache.db_path))
            self.use_clustering = bool(rules.get('use_clustering', self.use_clustering)) and _HAS_RAPIDFUZZ

        if df.height == 0 or df.width == 0:
            return df, self._create_report([])

        # 1) 先跑超快规则清洗
        print("[Ultra] Stage 1/3: fast rule-based cleaning (vectorized)...")
        df_fast, fast_report = await self.fast.clean(df, primary_keys, rules)

        # 2) 选择“可疑列”
        print("[Ultra] Stage 2/3: column screening...")
        candidate_cols = self._select_columns(df_fast, primary_keys)
        if not candidate_cols:
            return df_fast, fast_report

        # 3) 唯一值→LLM 微批 + 缓存
        print(f"[Ultra] Stage 3/3: value-dictionary micro-batch cleaning on {len(candidate_cols)} columns")
        current = df_fast
        all_changes = list(fast_report.changes)

        for col in candidate_cols:
            vc = current.select(pl.col(col).cast(pl.Utf8)).to_series()
            unique_df = pl.DataFrame({"val": vc}).with_columns([
                pl.col("val").fill_null(""),
            ]).group_by("val").len().sort("len", descending=True)
            unique_df = unique_df.filter(pl.col("val") != "")
            total_unique = unique_df.height
            if total_unique == 0:
                continue

            budget = max(50, int(total_unique * self.llm_budget_ratio))
            top_df = unique_df.head(budget)
            raw_vals = top_df["val"].to_list()

            # 先查缓存
            cached = self.cache.get_many(col, raw_vals, self.version_tag)
            remaining = [v for v in raw_vals if v not in cached]

            # 可选：相似聚类（只清代表值）
            clusters, rep_values = [], []
            if self.use_clustering and remaining:
                try:
                    labs = cluster(remaining, scorer=Levenshtein.normalized_similarity, score_cutoff=self.cluster_threshold/100.0)
                    for group in labs:
                        g_vals = [remaining[i] for i in group]
                        rep = max(g_vals, key=lambda x: top_df.filter(pl.col("val")==x)["len"][0])
                        clusters.append(g_vals); rep_values.append(rep)
                except Exception:
                    rep_values = remaining
            else:
                rep_values = remaining

            # LLM 微批
            new_map = {}
            for batch in self._batches(rep_values, self.value_batch_size):
                batch_map = await self._clean_values_via_llm(col, batch)
                new_map.update(batch_map)

            # 聚类扩散
            if self.use_clustering and clusters and new_map:
                for g_vals, rep in zip(clusters, rep_values):
                    if rep in new_map:
                        for v in g_vals:
                            new_map.setdefault(v, new_map[rep])

            if new_map:
                self.cache.put_many(col, new_map, self.version_tag)

            col_map = {**cached, **new_map}
            if not col_map:
                continue

            # 应用映射（只替换 TopK）
            map_df = pl.DataFrame({"__raw": list(col_map.keys()), "__clean": list(col_map.values())})
            before = current
            current = before.join(map_df, left_on=col, right_on="__raw", how="left") \
                            .with_columns(pl.when(pl.col("__clean").is_not_null())
                                            .then(pl.col("__clean"))
                                            .otherwise(pl.col(col))
                                            .alias(col)) \
                            .drop(["__raw", "__clean"])

            # 记录变更（估算）
            changed_mask = before[col].cast(pl.Utf8) != current[col].cast(pl.Utf8)
            changed_rows = int(changed_mask.sum())
            if changed_rows > 0:
                all_changes.append({
                    "column": col,
                    "changed_rows": changed_rows,
                    "changed_cells": changed_rows,
                    "method": "ultra_micro_batch",
                })

        report = self._create_report(all_changes)
        return current, report

    # ---- 辅助 ----
    def _select_columns(self, df: pl.DataFrame, primary_keys: List[str]) -> List[str]:
        cols = []
        n = max(1, df.height)
        for col, dtype in df.schema.items():
            if col in primary_keys:
                continue
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Boolean):
                continue
            s = df[col].cast(pl.Utf8)
            null_rate = float(s.null_count()) / n
            nunique = int(s.n_unique())
            unique_rate = nunique / n
            punct_rate = float(s.str.count_matches(r"[^0-9A-Za-z\u4e00-\u9fa5 ]").fill_null(0).sum()) / max(1, int(s.str.len_chars().fill_null(0).sum()))
            if unique_rate > 0.02 and punct_rate > 0.02:
                cols.append(col)
            elif unique_rate > 0.2:
                cols.append(col)
            elif null_rate > 0.2 and unique_rate > 0.01:
                cols.append(col)
        return cols

    def _batches(self, arr: List[str], size: int) -> Iterable[List[str]]:
        for i in range(0, len(arr), size):
            yield arr[i:i+size]

    async def _clean_values_via_llm(self, col: str, values: List[str]) -> Dict[str, str]:
        if not values:
            return {}
        async with self.sem:
            prompt = self._build_prompt(col, values)
            try:
                resp = await self.llm.chat("gpt-4o-mini", [
                    {"role": "user", "content": prompt}
                ])
            except Exception as e:
                print(f"[Ultra] LLM chat error: {e}")
                return {}

        # 兼容 string/对象 两种返回
        try:
            msg = resp.choices[0].message.content
        except Exception:
            try:
                msg = resp["choices"][0]["message"]["content"]
            except Exception:
                msg = str(resp)

        data = self._extract_json(msg)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v is not None}
        if isinstance(data, list):
            out = {}
            if all(isinstance(x, dict) and "in" in x and "out" in x for x in data):
                for x in data:
                    out[str(x["in"])] = str(x["out"])
                return out
            elif len(data) == len(values):
                for raw, cleaned in zip(values, data):
                    out[str(raw)] = str(cleaned)
                return out
        return {}

    def _build_prompt(self, col: str, values: List[str]) -> str:
        examples = "\\n".join(f"- {v}" for v in values[:10])
        return f"""
你是一个数据标准化助手。请将同一列“{col}”中的值进行**最小改动的标准化**，例如去除多余空格、统一大小写/格式、去掉明显噪声（如尾随分隔符、括号内冗余等），不要推测新增信息。
输入是一个 JSON 列表，请返回 **等长** 的 JSON 列表，顺序与输入一一对应。

输入示例（仅展示前10个）:
{examples}

现在请清洗以下 JSON 列表（仅返回 JSON，勿加解释）：
{json.dumps(values, ensure_ascii=False)}
""".strip()

    def _extract_json(self, text: str) -> Any:
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None
