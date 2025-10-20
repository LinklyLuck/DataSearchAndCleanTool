# er_providers/base.py
from __future__ import annotations
import re, asyncio
from typing import List, Dict, Optional, Any
import httpx

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def pick_entity_field(row: Dict[str, Any]) -> Optional[str]:
    """
    在一行数据里挑一个“像实体名”的字段；按常见优先级选取。
    """
    candidates = [
        "movie_title", "title", "name", "entity_name", "product_name",
        "ticker", "director_name", "company", "brand"
    ]
    for k in candidates:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return k
    # 兜底：第一个字符串列
    for k, v in row.items():
        if isinstance(v, str) and v.strip():
            return k
    return None

class BaseProvider:
    name = "BaseProvider"
    def __init__(self, timeout: int = 20, max_concurrency: int = 6):
        self.timeout = timeout
        self.sem = asyncio.Semaphore(max_concurrency)
        self.client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self):
        await self.client.aclose()

    # ---- 抽象接口 ----
    async def resolve_entities(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return rows

    async def impute_missing(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return rows
