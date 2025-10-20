# enrichment_providers.py — Wikipedia + Kaggle baseline ER/MVI (safe, no row creation)
from __future__ import annotations
from typing import List, Dict, Optional
import os, re, json, base64, hashlib
import httpx

# ---- 公共接口 ----
class EnrichProvider:
    name = "BaseProvider"
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)

    async def aclose(self):
        await self.client.aclose()

    async def enrich_rows(
        self,
        rows: List[Dict],
        primary_keys: List[str],
        allowed_update_cols: List[str],
        enrich_only: bool = True,
        no_title_change: bool = True
    ) -> List[Dict]:
        """返回与 rows 等长的列表；只对 allowed_update_cols 的空值进行填补"""
        return rows  # 基类不做事

# ---- 自动域识别（query + columns） ----
async def detect_domain_from_query_or_columns(query: str, columns: List[str], llm_client=None) -> str:
    s = (query or "").lower() + " " + " ".join([c.lower() for c in columns])
    # 粗规则：包含 NBA/球员列名
    nba_signals = ["nba", "player", "player_name", "team_abbreviation", "draft_year", "draft_round"]
    if any(k in s for k in nba_signals):
        return "sports_nba"
    # 电影
    movie_signals = ["movie", "film", "imdb", "director_name", "movie_title", "box office"]
    if any(k in s for k in movie_signals):
        return "movie"
    # 股票
    stock_signals = ["stock", "ticker", "ohlc", "close_price", "open_price"]
    if any(k in s for k in stock_signals):
        return "stock"
    # 回退
    return "fallback"

def get_provider_for_auto_domain(domain: str) -> EnrichProvider:
    if domain == "sports_nba":
        return WikipediaKaggleNBAProvider()
    if domain == "movie":
        return WikipediaKaggleMovieProvider()
    return WikipediaGenericProvider()

# ---- Wikipedia 基础封装 ----
class WikipediaClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search_first_page(self, query: str) -> Optional[str]:
        # MediaWiki API：opensearch
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"}
        r = await self.client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) >= 2 and data[1]:
            return data[1][0]
        return None

    async def get_page_summary(self, title: str) -> Optional[dict]:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        r = await self.client.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

# ---- Kaggle 基础封装（无凭据则跳过） ----
class KaggleClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.user = os.getenv("KAGGLE_USERNAME")
        self.key  = os.getenv("KAGGLE_KEY")
        self.ok = bool(self.user and self.key)

    async def search_dataset(self, q: str) -> Optional[dict]:
        if not self.ok:
            return None
        # Kaggle 官方 API 多用 CLI/token；这里仅做“占位”逻辑（无强依赖）
        # 你可以把内部替换成真正的 kaggle API/CLI 调用（subprocess）
        return None

# ---- NBA Provider：仅填补既有列的空值 ----
class WikipediaKaggleNBAProvider(EnrichProvider):
    name = "Wikipedia+Kaggle(NBA)"
    def __init__(self):
        super().__init__()
        self.wiki = WikipediaClient(self.client)
        self.kaggle = KaggleClient(self.client)

    async def enrich_rows(self, rows, primary_keys, allowed_update_cols, enrich_only=True, no_title_change=True):
        out = []
        for r in rows:
            # 识别键：优先 player_name
            pname = r.get("player_name") or r.get("name") or r.get("player") or ""
            team  = r.get("team_abbreviation") or r.get("team") or ""
            if not pname:
                out.append(r); continue

            # 去 Wikipedia 拿 summary（轻量，不增加行）
            enriched = dict(r)
            try:
                page = await self.wiki.search_first_page(f"{pname} basketball")
                if page:
                    summ = await self.wiki.get_page_summary(page)
                    # 只在空时填补：height/weight/college/country/draft_*（这些字段如果出现在 allowed_update_cols）
                    # 这里的解析很轻量，你可以替换成更精准的 infobox 解析逻辑
                    text = (summ or {}).get("extract", "") or ""
                    def try_put(col, value):
                        if col in allowed_update_cols:
                            v0 = enriched.get(col)
                            if (v0 is None or str(v0).strip() == "") and value not in (None, ""):
                                enriched[col] = value

                    # 朴素正则（示例）：不保证全覆盖，仅演示“只填空值”
                    if "player_height" in allowed_update_cols:
                        m = re.search(r"(\d+\s?cm|\d+\.?\d*\s?m|[5-7]'\s?\d{1,2}\")", text)
                        try_put("player_height", m.group(1) if m else None)
                    if "player_weight" in allowed_update_cols:
                        m = re.search(r"(\d+\s?kg|\d+\s?lb)", text)
                        try_put("player_weight", m.group(1) if m else None)
                    for key in ("college","country","draft_year","draft_round","draft_number"):
                        if key in allowed_update_cols:
                            # 非结构化提取示意（可自行优化）
                            m = re.search(fr"{key.replace('_',' ')}[:\s]+([A-Za-z0-9' \-]+)", text, re.I)
                            try_put(key, m.group(1) if m else None)

            except Exception:
                pass

            out.append(enriched)
        return out

# ---- Movie Provider（同样仅填补空值） ----
class WikipediaKaggleMovieProvider(EnrichProvider):
    name = "Wikipedia+Kaggle(Movie)"
    def __init__(self):
        super().__init__()
        self.wiki = WikipediaClient(self.client)
        self.kaggle = KaggleClient(self.client)

    async def enrich_rows(self, rows, primary_keys, allowed_update_cols, enrich_only=True, no_title_change=True):
        out = []
        for r in rows:
            title = r.get("movie_title") or r.get("title") or ""
            if not title:
                out.append(r); continue
            enriched = dict(r)
            try:
                page = await self.wiki.search_first_page(f"{title} film")
                if page:
                    summ = await self.wiki.get_page_summary(page)
                    text = (summ or {}).get("extract", "") or ""
                    def try_put(col, value):
                        if col in allowed_update_cols:
                            v0 = enriched.get(col)
                            if (v0 is None or str(v0).strip() == "") and value not in (None, ""):
                                enriched[col] = value
                    for key in ("runtime_minutes","country_code","language"):
                        if key in allowed_update_cols:
                            m = re.search(fr"{key.replace('_',' ')}[:\s]+([A-Za-z0-9' \-]+)", text, re.I)
                            try_put(key, m.group(1) if m else None)
            except Exception:
                pass
            out.append(enriched)
        return out

# ---- Fallback Provider（尽量不做事） ----
class WikipediaGenericProvider(EnrichProvider):
    name = "Wikipedia(Generic)"
    def __init__(self):
        super().__init__()
        self.wiki = WikipediaClient(self.client)
    async def enrich_rows(self, rows, primary_keys, allowed_update_cols, enrich_only=True, no_title_change=True):
        # 默认基本不改，确保稳定
        return rows
