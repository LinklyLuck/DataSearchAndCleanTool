# er_providers/wikipedia_kaggle.py
from __future__ import annotations
import os, re, json, asyncio, base64
from typing import List, Dict, Any, Optional
import httpx

from .base import BaseProvider, pick_entity_field, _normalize

# =============== 领域识别（LLM + 规则） ==================
async def _llm_guess_domain(client, query: str, candidate_domains: List[str]) -> Optional[str]:
    """
    使用你现有的 LLMClient（可选）做精细识别；必须严格返回列表中的一个词。
    """
    if client is None:
        return None
    prompt = (
        "You are a domain router. Pick EXACTLY one domain from the list.\n"
        f"Query: {query!r}\n"
        f"Domains: {', '.join(candidate_domains)}\n"
        "Return only the domain keyword."
    )
    try:
        out = await client.chat("gpt-4o-mini", [{"role":"user","content":prompt}])
        cand = out.strip().split()[0].lower()
        if cand in [d.lower() for d in candidate_domains]:
            return cand
    except Exception:
        pass
    return None

def _rule_guess_domain(query: str) -> str:
    s = (query or "").lower()
    if any(k in s for k in ["movie","film","cinema","imdb","box office","director","runtime"]):
        return "movie"
    if any(k in s for k in ["stock","equity","ticker","ohlc","nasdaq","nyse"]):
        return "stock"
    if any(k in s for k in ["ecommerce","retail","sku","basket","order","product"]):
        return "ecommerce"
    return "generic"

async def detect_domain_from_query(query: str, llm_client=None) -> str:
    """
    自动识别领域（LLM 优先 + 规则兜底）。可直接调用。
    """
    candidates = ["movie","stock","ecommerce","generic"]
    cand = await _llm_guess_domain(llm_client, query, candidates)
    return cand or _rule_guess_domain(query)

# =============== Provider：Wikipedia + 可选 Kaggle ==================
class WikipediaKaggleProvider(BaseProvider):
    """
    面向“多领域”的基线 ER/MVI：
    - 统一：先从每行中找一个“实体名字段”(movie_title/title/name/… 或任何字符串列)
    - ER：用 Wikipedia 搜索 + summary 来标准化名称、拿到简介/可能的年份等
    - MVI：根据表头的“迹象”做轻量填补（电影/股票/电商常用字段），其他则 generic 填补
    - Kaggle：如设置 KAGGLE_USERNAME/KAGGLE_KEY 且检测到 ticker/name 等关键字，补充可用元数据（若本机未装 kaggle 包则跳过）
    """
    name = "Wikipedia+Kaggle"

    # -------- Wikipedia 工具 --------
    async def _wiki_search(self, q: str) -> Optional[str]:
        if not q: return None
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action":"opensearch", "search":q, "limit":1, "namespace":0, "format":"json"}
        async with self.sem:
            r = await self.client.get(url, params=params)
        try:
            js = r.json()
            if isinstance(js, list) and len(js) >= 2 and js[1]:
                return js[1][0]
        except Exception:
            pass
        return None

    async def _wiki_summary(self, title: str) -> Dict[str, Any]:
        if not title: return {}
        # REST summary
        safe = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}"
        async with self.sem:
            r = await self.client.get(url)
        try:
            js = r.json()
            # 常见字段：title, description, extract
            return {
                "wiki_title": js.get("title"),
                "wiki_description": js.get("description"),
                "wiki_summary": js.get("extract"),
            }
        except Exception:
            return {}

    # -------- Kaggle（可选）工具 --------
    async def _kaggle_meta(self, query: str) -> Dict[str, Any]:
        """
        如果本机安装了 kaggle 包且设置了 KAGGLE_USERNAME/KAGGLE_KEY，则查询一下与 query 相关的数据集元信息。
        只返回非常轻量级的头部信息；不可用时返回 {}。
        """
        if not query: return {}
        if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
            return {}

        try:
            import kaggle  # 如果没有会 ImportError
            # kaggle.api.datasets_list(search=query) -> list of Dataset
            # 该接口是同步的，用线程池跑一下；为简单起见，这里直接跑同步（调用方是小规模）
            def _call():
                api = kaggle.api
                api.authenticate()
                ds = api.datasets_list(search=query, page_size=3)
                out = []
                for d in ds[:3]:
                    # d.ref, d.title, d.size, d.lastUpdated
                    out.append({"kaggle_ref": d.ref, "kaggle_title": d.title})
                return {"kaggle_top": out}
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _call)
        except Exception:
            return {}

    # -------- 领域判定（基于列） --------
    @staticmethod
    def _guess_schema_kind(columns: List[str]) -> str:
        s = ",".join([c.lower() for c in columns])
        if any(k in s for k in ["movie_title","director_name","release_year","imdb_rating","genre"]):
            return "movie"
        if any(k in s for k in ["ticker","close_price","open_price","volume","exchange","date"]):
            return "stock"
        if any(k in s for k in ["order_id","product_id","sku","price","quantity","customer_id"]):
            return "ecommerce"
        return "generic"

    # -------- ER：实体解析 --------
    async def resolve_entities(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        tasks = []
        for row in rows:
            key = pick_entity_field(row)
            ent = _normalize(row.get(key, "")) if key else ""
            if not ent:
                out.append(dict(row))
                continue

            async def run_one(entity: str, original: Dict[str, Any]):
                title = await self._wiki_search(entity) or entity
                s = await self._wiki_summary(title)
                # 可选 kaggle
                k = await self._kaggle_meta(entity)
                return {**original, **s, **k, "normalized_name": title}

            tasks.append(run_one(ent, dict(row)))

        if tasks:
            out = await asyncio.gather(*tasks)
        else:
            out = rows
        return out

    # -------- MVI：缺失值填补（轻量） --------
    @staticmethod
    def _extract_year_from_text(txt: str) -> Optional[int]:
        if not txt: return None
        m = re.search(r"(19|20)\d{2}", txt)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        return None

    async def impute_missing(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return rows
        cols = list({k for r in rows for k in r.keys()})
        kind = self._guess_schema_kind(cols)

        filled = []
        for r in rows:
            rr = dict(r)
            # 通用：给 entity_name/title 统一归一
            if "entity_name" in rr and not rr.get("entity_name"):
                rr["entity_name"] = rr.get("normalized_name") or rr.get("wiki_title")
            if "title" in rr and not rr.get("title"):
                rr["title"] = rr.get("normalized_name") or rr.get("wiki_title")

            # 通用：如果有 description/summary 缺失，就互补
            if "description" in cols and not rr.get("description"):
                rr["description"] = rr.get("wiki_description") or rr.get("wiki_summary")
            if "summary" in cols and not rr.get("summary"):
                rr["summary"] = rr.get("wiki_summary") or rr.get("wiki_description")

            if kind == "movie":
                # 电影：release_year 缺失时从 summary/description 里猜一个年份
                if "release_year" in cols and not rr.get("release_year"):
                    yr = self._extract_year_from_text(rr.get("wiki_summary") or rr.get("wiki_description") or "")
                    if yr: rr["release_year"] = yr
            elif kind == "stock":
                # 股票：没有 exchange 时，尽量从 summary 里找一下（非常弱的启发式）
                if "exchange" in cols and not rr.get("exchange"):
                    txt = (rr.get("wiki_summary") or rr.get("wiki_description") or "").lower()
                    if "nasdaq" in txt:
                        rr["exchange"] = "NASDAQ"
                    elif "nyse" in txt:
                        rr["exchange"] = "NYSE"
            elif kind == "ecommerce":
                # 电商：category 或 product_name 缺失用 wiki 标题兜底
                if "category" in cols and not rr.get("category"):
                    rr["category"] = "unknown"
                if "product_name" in cols and not rr.get("product_name"):
                    rr["product_name"] = rr.get("normalized_name") or rr.get("wiki_title")
            # generic：尽量不做副作用太强的填补

            filled.append(rr)
        return filled
