# the_pipeline_v2.py — local-only mining → template-aware mapping → safe join → enrich (ER/MVI) → CSV
from __future__ import annotations
import os, re, sys, json, csv, asyncio, hashlib, sqlite3, importlib.util, glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import httpx
import polars as pl
from tqdm.asyncio import tqdm_asyncio

# ===== Template驱动的智能Schema映射（新增）=====
try:
    sys.path.insert(0, str(Path(__file__).parent / "templates"))
    from template_utils import TemplateManager, SchemaMapper, DynamicTemplate

    _HAS_TEMPLATE_UTILS = True
except ImportError:
    _HAS_TEMPLATE_UTILS = False
    print("[INFO] template_utils not found, using legacy template logic")

# === I/O：Polars 为主，Pandas 仅兜底读取（随后立刻转 Polars）===
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _read_any_backend(path: Path) -> pl.DataFrame:
    """多格式读取器：始终返回 Polars DataFrame；必要时用 pandas 读取后再 pl.from_pandas。"""
    suf = path.suffix.lower()

    # CSV
    if suf == ".csv":
        try:
            return pl.read_csv(
                path,
                infer_schema_length=10000,
                null_values=["--", "-", "", "NA", "N/A", "null", "None"],
                ignore_errors=True
            )
        except Exception as e:
            if _HAS_PANDAS:
                pdf = pd.read_csv(path, dtype=str, low_memory=False)
                return pl.from_pandas(pdf)
            raise

    # JSON / NDJSON
    if suf in {".json", ".ndjson", ".jsonl"}:
        try:
            return pl.read_json(path, infer_schema_length=10000)
        except Exception:
            # 容错：逐行解析
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                good = []
                for ln in lines:
                    s = ln.strip().rstrip(",")
                    if not s:
                        continue
                    try:
                        good.append(json.loads(s))
                    except Exception:
                        continue
                if good:
                    return pl.from_dicts(good)
            except Exception:
                pass
            if _HAS_PANDAS:
                try:
                    pdf = pd.read_json(path, lines=True)
                except Exception:
                    pdf = pd.read_json(path)
                return pl.from_pandas(pdf)
            raise

    # Excel
    if suf in {".xlsx", ".xls"}:
        try:
            return pl.read_excel(path)
        except Exception:
            if _HAS_PANDAS:
                pdf = pd.read_excel(path, sheet_name=0, dtype=str)
                return pl.from_pandas(pdf)
            raise RuntimeError(f"Excel read fail: {path}")

    # 默认走 CSV
    try:
        return pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)
    except Exception as e:
        if _HAS_PANDAS:
            pdf = pd.read_csv(path, dtype=str, low_memory=False)
            return pl.from_pandas(pdf)
        raise


# === 扩展扫描策略 ===
def _brace_expand(pat: str) -> List[str]:
    m = re.search(r"\{([^}]+)\}", pat)
    if not m:
        return [pat]
    head = pat[:m.start()]
    tail = pat[m.end():]
    alts = m.group(1).split(",")
    out = []
    for a in alts:
        out.extend(_brace_expand(head + a + tail))
    return out


def _expand_globs(patterns: List[str]) -> List[Path]:
    if not patterns:
        return []
    raw: List[str] = []
    for pat in patterns:
        if not pat:
            continue
        parts = [p.strip() for p in re.split(r"[;]", pat) if p.strip()]
        for p0 in parts:
            for p1 in _brace_expand(p0):
                raw.append(p1)

    files: List[Path] = []
    for p in raw:
        for hit in glob.glob(p, recursive=True):
            ph = Path(hit)
            if not (ph.exists() and ph.is_file()):
                continue
            # ✨ 跳过 Excel 临时锁文件
            if ph.suffix.lower() in {".xlsx", ".xls"} and ph.name.startswith("~$"):
                continue
            files.append(ph)

    uniq = sorted({f.resolve() for f in files})
    return uniq


def _expand_globs_mixed(patterns: List[str]) -> List[Path]:
    """
    扫描 datas/、datalake/，如果存在 data/ 则混合加入。
    """
    base_dirs = ["datas", "datalake"]
    if Path("data").exists():
        base_dirs.append("data")
    new_patterns = []
    for pat in patterns:
        for base in base_dirs:
            # 替换 datas/ 或 datalake/ 前缀为当前 base
            if "datas/" in pat:
                new_patterns.append(pat.replace("datas/", f"{base}/"))
            elif "datalake/" in pat:
                new_patterns.append(pat.replace("datalake/", f"{base}/"))
            else:
                new_patterns.append(str(Path(base) / pat))
    return _expand_globs(new_patterns)


# === 外部增强：ER（实体解析）+ MVI（缺失值填补）分离 ===
# ER: 使用 er_providers (LLM驱动)
from er_providers import (
    get_provider_for_auto_domain,
    detect_domain_from_query_or_columns,
)

# MVI: 使用 mvi_providers (外部API驱动)
try:
    from mvi_providers import (

        get_mvi_provider_for_domain,
        detect_domain_from_columns
    )

    _HAS_MVI_PROVIDERS = True
except ImportError:
    _HAS_MVI_PROVIDERS = False
    print("[INFO] mvi_providers not found, MVI will be skipped")

try:
    import yaml  # optional for .yaml config
except ImportError:
    yaml = None


# ========================= JSON 容错读取 & 温和扁平化 =========================
def _flatten_df(df: pl.DataFrame, max_rounds: int = 1) -> pl.DataFrame:
    out = df
    for _ in range(max_rounds):
        list_struct_cols = []
        for c, dt in zip(out.columns, out.dtypes):
            if isinstance(dt, pl.List):
                try:
                    tmp = out.select(pl.col(c).list.get(0))
                    inner = tmp.schema.get(c)
                    if isinstance(inner, pl.Struct):
                        list_struct_cols.append(c)
                except Exception:
                    pass
        for c in list_struct_cols:
            out = out.explode(c)

        struct_cols = [c for c, dt in zip(out.columns, out.dtypes) if isinstance(dt, pl.Struct)]
        if not struct_cols and not list_struct_cols:
            break
        for c in struct_cols:
            out = out.unnest(c)
    return out


def _json_sanitize(s: str) -> str:
    s = s.lstrip("\ufeff")
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def _read_json_lenient(path: Path) -> pl.DataFrame:
    try:
        return pl.read_json(path, infer_schema_length=10000)
    except Exception:
        pass
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    good = []
    for ln in lines:
        s = ln.strip().rstrip(",")
        if not s:
            continue
        try:
            good.append(json.loads(_json_sanitize(s)))
        except Exception:
            continue
    if good:
        return pl.from_dicts(good)
    obj = json.loads(_json_sanitize("\n".join(lines)))
    if isinstance(obj, list):
        return pl.from_dicts(obj)
    if isinstance(obj, dict):
        return pl.from_dicts([obj])
    raise RuntimeError(f"Failed to read JSON: {path}")


def _read_ndjson_lenient(path: Path) -> pl.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    good = []
    for ln in lines:
        s = ln.strip().rstrip(",")
        if not s:
            continue
        try:
            good.append(json.loads(_json_sanitize(s)))
        except Exception:
            continue
    if good:
        return pl.from_dicts(good)
    return _read_json_lenient(path)


# --- Excel 读取（pandas 兜底） ---
def _read_excel_df(path: Path) -> pl.DataFrame:
    try:
        return pl.read_excel(path)
    except Exception:
        try:
            import pandas as pd
            pdf = pd.read_excel(path, sheet_name=0, dtype=str)
            return pl.from_pandas(pdf)
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel: {path} ({e})")


def scan_any(path: Path) -> pl.LazyFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pl.scan_csv(
            path,
            infer_schema_length=10000,
            null_values=["--", "-", "", "NA", "N/A", "null", "None"],
            ignore_errors=True
        )
    elif suf == ".json":
        df = _flatten_df(_read_json_lenient(path))
        return df.lazy()
    elif suf in {".ndjson", ".jsonl"}:
        df = _flatten_df(_read_ndjson_lenient(path))
        return df.lazy()
    elif suf in {".xlsx", ".xls"}:
        df = _flatten_df(_read_excel_df(path))
        return df.lazy()
    # default as CSV
    return pl.scan_csv(
        path,
        infer_schema_length=10000,
        null_values=["--", "-", "", "NA", "N/A", "null", "None"],
        ignore_errors=True
    )


def read_any_df(path: Path) -> pl.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pl.read_csv(
            path,
            infer_schema_length=10000,
            null_values=["--", "-", "", "NA", "N/A", "null", "None"],
            ignore_errors=True
        )
    elif suf == ".json":
        return _flatten_df(_read_json_lenient(path))
    elif suf in {".ndjson", ".jsonl"}:
        return _flatten_df(_read_ndjson_lenient(path))
    elif suf in {".xlsx", ".xls"}:
        return _flatten_df(_read_excel_df(path))
    return pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["--", "-", "", "NA", "N/A", "null", "None"],
        ignore_errors=True
    )


def preview_any(path: Path) -> tuple[str, str]:
    lf = scan_any(path)
    schema = lf.collect_schema()
    cols = list(schema.names())
    row_df = lf.limit(1).collect()
    row = [str(row_df[c][0]) if (c in row_df.columns and row_df.height > 0) else "" for c in cols]
    return ",".join(cols), ",".join(row)


# ============================= 配置 & LLM ================================
@dataclass
class Config:
    question: str
    datasets: List[str]
    out: str = "merged_training.csv"
    target_feature: str | None = None
    left_model: str = "gpt-4o-mini"
    map_model: str = "gpt-4o-mini"
    join_model: str = "gpt-4o-mini"
    max_concurrency: int = 2
    cache_path: Path = Path(".llm_cache.sqlite")
    api_base: str = "https://goapi.gptnb.ai/v1/chat/completions"
    api_key: str = ""
    # 新增：通用合并（保留所有列/行）开关，默认启用，不改变原安全内连接的输出（写入 meta 供审计）
    universal_merge: bool = True

    # —— 安全模式（可选，不改原逻辑；只是限流避免内存爆）——
    max_files: int = 20  # 一次最多合并多少个文件（默认20）
    max_rows_per_file: int = 200_000  # 每个文件最多读取多少行（超出截断）
    filter_by_query_keywords: bool = True  # 根据 query 关键词粗过滤文件名


# 👉 默认扫描"用户上传 datas/ + 本地 datalake/"
DEFAULT_CONFIG = Config(
    question="please help me find a dataset for training linear regression model to predict red wine quality",
    datasets=[
        "datas/**/*.{csv,json,ndjson,jsonl,xlsx,xls}",
        "datalake/**/*.{csv,json,ndjson,jsonl,xlsx,xls}",
    ],
    out="merged_training.csv"
)


class LLMCache:
    def __init__(self, db: Path):
        self.db = db;
        self._init()

    def _init(self):
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY, completion TEXT)")

    def _h(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    async def get(self, prompt: str) -> Optional[str]:
        h = self._h(prompt)
        with sqlite3.connect(self.db) as con:
            row = con.execute("SELECT completion FROM cache WHERE hash=?", (h,)).fetchone()
            return row[0] if row else None

    async def set(self, prompt: str, completion: str):
        h = self._h(prompt)
        with sqlite3.connect(self.db) as con:
            con.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (h, completion));
            con.commit()


class LLMClient:
    def __init__(self, cache: LLMCache, cfg: Config):
        self.cache = cache
        self.sem = asyncio.Semaphore(cfg.max_concurrency)
        self.url = cfg.api_base
        key = (os.getenv("GPTNB_API_KEY", cfg.api_key) or "").strip().strip('"').strip("'")
        if not key:
            raise RuntimeError("Set GPTNB_API_KEY or cfg.api_key")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.client = httpx.AsyncClient(timeout=120,
                                        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))

    async def chat(self, model: str, messages: List[Dict]) -> str:
        payload = {"model": model, "stream": False, "messages": messages}
        prompt = json.dumps(payload, sort_keys=True)
        cached = await self.cache.get(prompt)
        if cached:
            return cached
        async with self.sem:
            for attempt in range(5):
                try:
                    r = await self.client.post(self.url, headers=self.headers, json=payload)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    await self.cache.set(prompt, content)
                    await asyncio.sleep(0.2)
                    return content
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429,) or e.response.status_code >= 500:
                        await asyncio.sleep(min(30, 2 ** attempt));
                        continue
                    raise
        raise RuntimeError("LLM call failed after retries")

    async def aclose(self):
        await self.client.aclose()


# ========= LLM 驱动：域识别 + 动态模板推断（无 if/elif 规则） =========
import json as _json_mod, re as _re_mod


def _extract_json_block(txt: str) -> dict:
    try:
        return _json_mod.loads(txt)
    except Exception:
        pass
    m = _re_mod.search(r"\{.*\}", txt, flags=_re_mod.DOTALL)
    if m:
        try:
            return _json_mod.loads(m.group(0))
        except Exception:
            pass
    m = _re_mod.search(r"```(?:json)?\s*({.*?})\s*```", txt, flags=_re_mod.DOTALL | _re_mod.IGNORECASE)
    if m:
        try:
            return _json_mod.loads(m.group(1))
        except Exception:
            pass
    raise ValueError("LLM did not return valid JSON.")


async def detect_domain_from_query_llm(client: LLMClient, query: str, templates: dict) -> str:
    domains = [d for d in templates.keys() if not d.startswith("_")]
    domains_with_other = domains + (["other"] if "other" not in domains else [])
    prompt = f"""
You are a data taxonomy expert. 
Given the user query below, pick ONE domain from this list ONLY (exact string): {domains_with_other}.
Return ONLY the domain string, nothing else.

Query:
{query}
"""
    out = await client.chat("gpt-4o-mini", [{"role": "user", "content": prompt}])
    cand = out.strip().split()[0].strip()
    return cand if cand in domains_with_other else "other"


async def llm_dynamic_template_from_query(client: LLMClient, query: str) -> tuple[list, list]:
    prompt = f"""
You are an expert data modeler. From the user's query, propose a minimal feature set 
(underscore_case; at least 10 items; first is a sensible target_* if applicable) 
and a set of join keys that are realistic identifiers for combining tables.

Return STRICT JSON ONLY, with this schema:
{{
  "feature_names": ["..."],
  "join_keys": ["..."]
}}

User query:
{query}
"""
    out = await client.chat("gpt-4o-mini", [{"role": "user", "content": prompt}])
    obj = _extract_json_block(out)
    feats = [s.strip() for s in obj.get("feature_names", []) if s.strip()]
    jks = [s.strip() for s in obj.get("join_keys", []) if s.strip()]
    if not feats: feats = ["target_variable", "entity_name", "entity_id", "date", "category",
                           "numeric_feature_1", "numeric_feature_2", "numeric_feature_3", "text_feature",
                           "label_or_score"]
    if not jks:   jks = ["entity_id", "entity_name", "date"]
    return feats, jks


async def llm_refine_join_keys_with_headers(client: LLMClient, query: str, join_keys: list[str],
                                            headers_union: list[str]) -> list[str]:
    header_sample = headers_union[:200]
    prompt = f"""
We want join keys to join multiple tabular datasets.
The user query: {query}

Candidate join_keys from your earlier reasoning: {join_keys}

Here are the ACTUAL column headers (union across candidate files):
{header_sample}

Pick 1-3 columns from the headers that best correspond to those join keys.
Return STRICT JSON ONLY:
{{
  "join_keys_in_headers": ["<must be picked from the headers above>"]
}}
"""
    out = await client.chat("gpt-4o-mini", [{"role": "user", "content": prompt}])
    try:
        obj = _extract_json_block(out)
        jk = [c for c in obj.get("join_keys_in_headers", []) if c in header_sample]
        if jk: return jk
    except Exception:
        pass
    return join_keys


# ===== 关键词提取（LLM + 正则兜底）=====
_TOPIC_PROMPT = """
You extract the SINGLE most specific topical keyword from a user's query about a database.
Return ONLY a single lowercase token without spaces (letters/numbers/underscores).
Examples:
- "I need wine database with id, ph and quality" -> wine
- "I need movie database with id" -> movie
- "NBA players database with ..." -> nba
If unsure, return a short noun from the query.
"""


async def extract_topic_keyword(client: LLMClient, query: str) -> str:
    try:
        out = await client.chat("gpt-4o-mini", [
            {"role": "user", "content": _TOPIC_PROMPT},
            {"role": "user", "content": query}
        ])
        kw = re.findall(r"[A-Za-z0-9_]{2,}", out.strip().lower())
        if kw:
            return kw[0]
    except Exception:
        pass
    # —— 兜底：优先抓 “<token> database”
    m = re.search(r"\b([A-Za-z][A-Za-z0-9_]{2,})\b(?=\s+database\b)", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # 再兜底：从关键词里挑一个最像主题名的
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", query) if
            t.lower() not in {"need", "with", "and", "the", "a", "an", "please", "help", "dataset", "database"}]
    return (toks[0].lower() if toks else "topic")


async def bootstrap_features_from_templates_llm_first(client: LLMClient, query: str, templates: dict) -> tuple[
    str, list[str], list[str], list[str]]:
    """
    增强版：优先使用template_utils动态生成template（带alias_rules）
    回退：使用原有逻辑
    """
    # ===== 新增：尝试使用TemplateManager动态生成template =====
    if _HAS_TEMPLATE_UTILS:
        try:
            # 创建TemplateManager（会自动缓存）
            tm = TemplateManager(client, cache_path=Path("templates/dynamic_templates.json"))

            # 注意：这里需要discovered_datasets信息，但这个函数被调用时还没发现数据集
            # 所以我们先用LLM模式生成一个基础template，后续在有数据集信息时再细化
            # 这里先返回基于query的动态推断

            # 使用LLM动态推断（不依赖预定义模板）
            feats, jks = await llm_dynamic_template_from_query(client, query)
            canon = list(dict.fromkeys(jks + feats))

            # 推断领域名称
            domain = await detect_domain_from_query_llm(client, query, templates)
            if domain == "other":
                domain = "dynamic"

            print(f"[TemplateManager] Using dynamic template generation for domain: {domain}")

            return domain, feats, jks, canon

        except Exception as e:
            print(f"[TemplateManager] Failed to use template_utils: {e}, falling back to legacy logic")

    # ===== 原有逻辑（回退）=====
    domain = await detect_domain_from_query_llm(client, query, templates)
    if domain != "other" and domain in templates:
        body = templates[domain]
        feats = list(body.get("feature_sets", [[]])[0])
        jks = list(body.get("join_keys", []))
        canon = list(body.get("canonical_order", feats))
        return domain, feats, jks, canon
    feats, jks = await llm_dynamic_template_from_query(client, query)
    canon = list(dict.fromkeys(jks + feats))
    return (domain if domain != "other" else "fallback"), feats, jks, canon


# ======================= 模板库（templates/ 目录）=========================
TEMPLATES_DIR = Path("templates")
DOMAIN_FILE = TEMPLATES_DIR / "last_domain.txt"
DOMAIN_DB = TEMPLATES_DIR / "domain_templates.json"


def _ensure_templates_dir():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def load_domain_templates() -> dict:
    _ensure_templates_dir()
    if not DOMAIN_DB.exists():
        fallback = {
            "_meta": {"version": 1},
            "fallback": {
                "keywords": [],
                "feature_sets": [[
                    "target_variable", "entity_name", "entity_id", "date", "category",
                    "numeric_feature_1", "numeric_feature_2", "numeric_feature_3",
                    "text_feature", "label_or_score"
                ]],
                "join_keys": ["entity_id", "entity_name", "date"],
                "canonical_order": [
                    "entity_name", "entity_id", "date", "category",
                    "numeric_feature_1", "numeric_feature_2", "numeric_feature_3",
                    "text_feature", "label_or_score"
                ],
                "policy": {
                    "strong_keys": ["entity_id", "entity_name"],
                    "join_min_keys": 2,
                    "same_origin": True,
                    "source_priority": [".csv", ".ndjson", ".jsonl", ".json", ".xlsx", ".xls"],
                    "max_blowup_ratio": 1.8,
                    "alias_rules": {}
                }
            }
        }
        DOMAIN_DB.write_text(json.dumps(fallback, indent=2, ensure_ascii=False))
    return json.loads(DOMAIN_DB.read_text(encoding="utf-8"))


def get_last_domain(default="fallback") -> str:
    _ensure_templates_dir()
    if DOMAIN_FILE.exists():
        s = DOMAIN_FILE.read_text(encoding="utf-8").strip()
        if s:
            return s
    return default


def save_last_domain(domain: str):
    _ensure_templates_dir()
    DOMAIN_FILE.write_text(domain.strip() or "fallback", encoding="utf-8")


# ======================== 关键词提取 ============================
def _keywords_from_query(q: str) -> list[str]:
    # 提取长度>=3的英文、数字片段做关键词
    toks = re.findall(r"[A-Za-z0-9_]+", q or "")
    toks = [t.lower() for t in toks if len(t) >= 3]
    # 常见停用词过滤（可再扩充）
    stop = {"dataset", "data", "file", "table", "model", "train", "test", "csv", "json"}
    return [t for t in toks if t not in stop]


# ======================== 策略（模板域可扩展）===========================
class DomainPolicy:
    def __init__(self, body: dict):
        pol = (body or {}).get("policy", {}) if isinstance(body, dict) else {}
        self.strong_keys = list(pol.get("strong_keys", []))
        self.join_min_keys = int(pol.get("join_min_keys", 2))
        self.same_origin = bool(pol.get("same_origin", True))
        self.source_priority = list(pol.get("source_priority", [".csv", ".ndjson", ".jsonl", ".json", ".xlsx", ".xls"]))
        self.max_blowup_ratio = float(pol.get("max_blowup_ratio", 1.8))
        self.alias_rules = dict(pol.get("alias_rules", {}))
        self.join_synonyms = dict(pol.get("join_synonyms", {}))  # optional
        self.passthrough_cols = list(pol.get("passthrough_cols", []))  # optional

    def normalize_col(self, col: str) -> str:
        c = col.strip().lower()
        for canon, syns in self.join_synonyms.items():
            syn_set = {s.lower() for s in ([canon] + syns)}
            if c in syn_set:
                return canon
        return col

    def is_join_equiv(self, a: str, b: str) -> bool:
        return self.normalize_col(a) == self.normalize_col(b)

    def dedupe_same_origin(self, paths: List[Path]) -> List[Path]:
        if not self.same_origin:
            return paths
        pr_map = {ext.lower(): i for i, ext in enumerate(self.source_priority[::-1], start=1)}
        keep: Dict[str, Path] = {}
        for p in paths:
            stem = p.with_suffix("").name.lower()
            old = keep.get(stem)
            if old is None:
                keep[stem] = p
            else:
                po = pr_map.get(p.suffix.lower(), 0)
                oo = pr_map.get(old.suffix.lower(), 0)
                if po > oo:
                    keep[stem] = p
        return list(keep.values())

    def alias_autofix(self, header: List[str], mapping: Dict[str, str]) -> Dict[str, str]:
        if not self.alias_rules:
            return mapping
        mapped = set(k for k, v in mapping.items() if v and v.lower() != "none")
        hdr = [h.strip() for h in header]
        new_mapping = dict(mapping)
        import re as _re
        for pattern, feat in self.alias_rules.items():
            if feat in mapped:
                continue
            reg = _re.compile(pattern, _re.IGNORECASE)
            for col in hdr:
                if reg.match(col):
                    new_mapping[feat] = col
                    mapped.add(feat)
                    break
        return new_mapping

    def require_strong_keys(self, join_cols: List[str]) -> bool:
        if len(join_cols) < self.join_min_keys:
            return False
        if self.strong_keys:
            return any(c in self.strong_keys for c in join_cols)
        return True

    def blowup_guard(self, left: pl.DataFrame, right: pl.DataFrame, keys: List[str]) -> bool:
        if not keys:
            return False
        try:
            a = left.group_by(keys).len().select(pl.col("len")).to_series()
            b = right.group_by(keys).len().select(pl.col("len")).to_series()
            if len(a) == 0 or len(b) == 0:
                return False
            est = a.mean() * b.mean()
            upper = max(left.height, right.height) * self.max_blowup_ratio
            return est <= upper
        except Exception:
            return True  # 出错就放行


# ======================== 映射→重命名→合并 ============================
MAP_RE = re.compile(r"^[\s\-•\*]*([\w\s\.\-\_/]+?)\s*(?:→|->|:)\s*([^\(\n]+?)(?:\s*\(index.*)?$", re.I)
KEY_RE = re.compile(r"^[\s•\-*\t]*([\w\.\-_/]+)", re.I)


def parse_mapping(text: str) -> Dict[str, str]:
    m = {}
    for line in text.splitlines():
        mm = MAP_RE.match(line.strip())
        if mm:
            feat, col = mm.groups()
            if col.lower() != "none":
                m[feat.strip()] = col.strip()
    return m


def parse_keys(text: str) -> List[str]:
    ks = []
    for line in text.splitlines():
        mm = KEY_RE.match(line.strip())
        if mm and mm.group(1).lower() != "none":
            ks.append(mm.group(1).strip())
    return ks


# ===== 从 Query 提取用户“点名列” =====
_REQ_SPLIT_RE = re.compile(r"[,\uFF0C，、/|;]|(?:\band\b)|(?:\bwith\b)", re.IGNORECASE)


def _norm_feat_name(s: str) -> str:
    """下划线小写化；只做匹配，不改动源数据列名。"""
    s = s.strip().strip(".")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return s.lower()


def extract_requested_columns_from_query(q: str) -> list[str]:
    """
    规则：
    - 英文：匹配 'with ...' 之后的部分（常见用法：with id, ph and quality）
    - 中文：匹配“包含/需要/字段为/列为”之后的部分
    - 用逗号/and/顿号等分割，做轻度归一化后去重
    """
    if not q:
        return []

    span = ""
    # 英文 "with ..."
    m = re.search(r"\bwith\b(.+)$", q, flags=re.IGNORECASE)
    if m:
        span = m.group(1)
    # 中文关键字
    if not span:
        m = re.search(r"(包含|需要|字段为|列为)(.+)$", q)
        if m:
            span = m.group(2)

    if not span:
        return []

    # 切分 & 轻度清洗（将 and 也作为分隔符）
    span = re.sub(r"\band\b", ",", span, flags=re.IGNORECASE)
    raw_tokens = [t for t in _REQ_SPLIT_RE.split(span) if t and t.strip()]
    feats = [_norm_feat_name(t) for t in raw_tokens if t.strip()]
    feats = [f for f in feats if len(f) >= 2]
    seen = set()
    uniq = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


# ===== 领域关键词抽取（通用）=====
def extract_domain_keyword(q: str) -> Optional[str]:
    """
    从 query 中抽取出 'xxx database' 的 xxx 作为领域关键词。
    例：'I need wine database with id...' -> 'wine'
        'I need NBA database with player id' -> 'nba'
        中文也支持：'需要电影数据库' -> '电影'
    仅做匹配用，不参与任何写死别名。
    """
    if not q:
        return None
    s = q.strip()

    # 英文：捕获 <something> database
    m = re.search(r"\b([A-Za-z0-9_\-\s]{1,60}?)\s+database\b", s, flags=re.IGNORECASE)
    if m:
        token = m.group(1).strip()
        # 取最后一个“词”（用户可能写 "red wine"），优先最具体的那个
        last_word = re.split(r"[\s_\-]+", token.strip())[-1]
        last_word = re.sub(r"[^A-Za-z0-9_]+", "", last_word).lower()
        return last_word or None

    # 中文：捕获 <something>数据库
    m = re.search(r"([\u4e00-\u9fa5A-Za-z0-9_]{1,30})数据库", s)
    if m:
        token = m.group(1).strip()
        # 中文不再切词，直接返回
        return token.lower()

    return None


# ===== 在一张表中，用领域关键词“偏好匹配”点名列（通用评分）=====
def resolve_requested_columns(headers: list[str], requested: list[str], domain: Optional[str]) -> Dict[
    str, Optional[str]]:
    """
    输入：headers 原始列名列表；requested 点名列（规范化后的 id/ph/quality...）；domain 领域关键词（如 wine）
    输出：字典 {requested_feat -> 实际匹配到的源列名或 None}
    规则（通用、无写死域）：
      - 完全同名优先（忽略大小写）
      - 若存在 domain，则 (domain + 分隔符 + feat) 或 (feat + 分隔符 + domain) 优先于裸 feat
      - 含 domain 的列加大权重；再看是否以 feat 开头；再看编辑距离短/列名短
      - 不做任何别名表；不固化“wine/movie/NBA”等字符串
    """
    if not headers or not requested:
        return {r: None for r in requested}

    def norm(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", s.strip()).strip("_").lower()

    H = headers[:]
    Hn = [norm(h) for h in H]
    mapping: Dict[str, Optional[str]] = {}

    for feat in requested:
        nf = norm(feat)

        best_idx = None
        best_score = -10 ** 9

        for i, (h, nh) in enumerate(zip(H, Hn)):
            score = 0
            # 基础：完全相等（忽略大小写/符号）
            if nh == nf:
                score += 50

            # 带领域关键词的优先（通用）
            if domain:
                nd = norm(domain)
                # 形如 wine_id / wineid / wine-id / id_wine
                if re.search(rf"(^|_)({nd})(_|$)", nh):
                    score += 20
                # 同时包含 feat 与 domain
                if nd in nh and nf in nh:
                    score += 30

                # 更严格的模式优先：{domain}[分隔符]{feat}
                if re.search(rf"(^|_){nd}[_-]*{nf}($|_)", nh):
                    score += 40
                # 或者 {feat}[分隔符]{domain}
                if re.search(rf"(^|_){nf}[_-]*{nd}($|_)", nh):
                    score += 25

            # 一般启发：以 feat 开头更好
            if nh.startswith(nf):
                score += 5

            # 次优：包含 feat
            if nf in nh:
                score += 2

            # 列名越短越好（更“干净”）
            score += max(0, 10 - len(nh))

            if score > best_score:
                best_score = score
                best_idx = i

        mapping[feat] = H[best_idx] if best_idx is not None else None

    return mapping


# ===== 列语义打分：基于列名 + 值分布（域感知强化版）=====
def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _series_numeric_ratio(df: pl.DataFrame, col: str) -> float:
    try:
        s = df.get_column(col)
        ok = s.cast(pl.Utf8).str.strip().str.replace_all(",", ".").str.extract(r"^-?\d+(?:\.\d+)?$").is_not_null()
        return float(ok.sum()) / max(1.0, float(df.height))
    except Exception:
        return 0.0


def _series_range_ratio(df: pl.DataFrame, col: str, lo: float, hi: float) -> float:
    try:
        s = df.get_column(col).cast(pl.Float64, strict=False)
        ok = s.is_not_null() & (s >= lo) & (s <= hi)
        return float(ok.sum()) / max(1.0, float(df.height))
    except Exception:
        return 0.0


def _series_int_band_ratio(df: pl.DataFrame, col: str, lo: int, hi: int) -> float:
    try:
        s = df.get_column(col).cast(pl.Int64, strict=False)
        ok = s.is_not_null() & (s >= lo) & (s <= hi)
        return float(ok.sum()) / max(1.0, float(df.height))
    except Exception:
        return 0.0


def _semantic_score_for_feature(df: pl.DataFrame, req_feat: str, header: str, topic_kw: Optional[str]) -> float:
    """
    自动语义打分（智能 domain-aware 版）
    —— 无写死字段名，但能自动理解 “wine id” ≠ “random id”。
    """
    nh = _norm_token(header)
    nf = _norm_token(req_feat)
    nt = _norm_token(topic_kw) if topic_kw else ""

    score = 0.0

    # === 1. 列名匹配基础 ===
    if nf and nf == nh:
        score += 15
    elif nf and nf in nh:
        score += 8
    if nf and nh.startswith(nf):
        score += 3
    score += max(0, 8 - len(nh)) * 0.3  # 短列名加分

    # === 2. Domain-aware 加权 ===
    if nt:
        # 1) 列名中包含 domain
        if nt in nh:
            score += 6
        # 2) domain + req_feat 组合（wine_id / player_id）
        if re.search(rf"{nt}[_\-]*{nf}", nh):
            score += 15
        # 3) req_feat + domain 组合（id_wine / id_player）
        if re.search(rf"{nf}[_\-]*{nt}", nh):
            score += 10
        # 4) domain 与 feat 都出现，强加分
        if nt in nh and nf in nh:
            score += 8

    # === 3. 值分布启发 ===
    try:
        s = df.get_column(header)
        num_ratio = _series_numeric_ratio(df, header)
        uniq_ratio = len(s.unique()) / max(1, len(s))

        # 数值列高相关
        score += 5 * num_ratio

        # 若唯一值比例高 → ID 类
        if uniq_ratio > 0.8 and num_ratio > 0.5:
            score += 6
        elif uniq_ratio > 0.3:
            score += 2

        # 如果列名含 domain 且数值分布正常，加额外分
        if nt and nt in nh and num_ratio > 0.5:
            score += 5
    except Exception:
        pass

    return float(score)


def _pick_best_header_for_req(df: pl.DataFrame, req: str, topic_kw: Optional[str]) -> Optional[str]:
    headers = df.columns
    if not headers:
        return None
    scored = [(h, _semantic_score_for_feature(df, req, h, topic_kw)) for h in headers]
    scored.sort(key=lambda x: x[1], reverse=True)

    # 动态阈值（更稳）：基础阈值略升；id 且有 topic_kw 时更加谨慎
    base_min = 2.5
    req_is_id = _norm_token(req) == "id"
    if req_is_id and topic_kw:
        base_min = 4.0  # 避免随便拿一个“泛 id”

    best, best_score = scored[0]
    # 若最佳分过低，放弃
    if best_score < base_min:
        return None

    # 专门处理 id：若第一名不含 domain，第二名含 domain 且分差不大，则换第二名
    if req_is_id and topic_kw:
        nt = _norm_token(topic_kw)
        best_has_domain = (nt in _norm_token(best))
        if not best_has_domain and len(scored) > 1:
            second, s_score = scored[1]
            if nt in _norm_token(second) and (best_score - s_score) <= 8:
                best, best_score = second, s_score

    return best


RIGHT_MAP_TMPL = """You are an expert DBA, who matches dataset columns to semantic feature names.

### INPUT
- Header: {header}
- Row: {row}
- Target features: {targets}
- Dataset: {dname}

### TASK
1. Map columns to target features where there's a clear match
2. For columns that don't match but look valuable (numeric, quality scores, measurements), 
   keep them with their ORIGINAL column name using format: <original_col_name> → <original_col_name>
3. Skip only truly useless columns (row_number, index, internal IDs)

Return bullet list:
feature_name → column_name (index = n)
or
column_name → column_name (for valuable unmapped columns)
"""

RIGHT_KEY_TMPL = """You are a data engineer.
Dataset = {dname}
Header  = {header}
Row     = {row}
Which column(s) look like a unique entity ID suitable for joining? Bullet list or `none`."""


def _prefer_domain_id_from_headers(hdr_list: List[str], req_cols: List[str], topic_kw: Optional[str]) -> Optional[str]:
    """
    仅用列名做域感知的 id 选择：优先 wine_id / id_wine / wineid 等。
    """
    if not topic_kw or not hdr_list:
        return None
    nt = re.sub(r"[^a-z0-9]+", "", topic_kw.lower())
    # 构造候选和优先级
    # 1) 完整分隔符形式
    patterns = [
        rf"^{nt}[_\-]*id$", rf"^id[_\-]*{nt}$",
        rf"^{nt}id$", rf"^id{nt}$",
        rf".*\b{nt}[_\-]*id\b.*", rf".*\bid[_\-]*{nt}\b.*"
    ]
    hdr_norm = {h: re.sub(r"[^a-z0-9]+", "", h.lower()) for h in hdr_list}
    for pat in patterns:
        reg = re.compile(pat)
        for h in hdr_list:
            if reg.match(h.lower()) or reg.match(hdr_norm[h]):
                return h
    return None


async def map_one_dataset(client, path, feats, cfg, policy: DomainPolicy | None = None, dynamic_template=None):
    """增强版：支持dynamic_template指导的schema映射"""
    header_str, row_str = preview_any(path)
    hdr_list = [h.strip() for h in header_str.split(",")]

    # ===== 新增：尝试使用SchemaMapper（带alias_rules指导） =====
    mapping = {}
    if _HAS_TEMPLATE_UTILS and dynamic_template is not None:
        try:
            tm = TemplateManager(client, cache_path=Path("templates/dynamic_templates.json"))
            mapper = SchemaMapper(client, tm)

            # 准备sample数据
            cols = [h.strip() for h in header_str.split(",") if h.strip()]
            sample_values = [s.strip() for s in row_str.split(",")]
            sample = dict(zip(cols, sample_values)) if len(cols) == len(sample_values) else {}

            # 使用template指导映射
            mapping = await mapper.map_schema_with_template(
                source_columns=hdr_list,
                source_sample=sample,
                template=dynamic_template,
                target_schema=feats
            )

            # 过滤掉None值
            mapping = {k: v for k, v in mapping.items() if v is not None}

            if mapping:
                print(f"[SchemaMapper] {path.name}: {len(mapping)} fields")
        except Exception as e:
            print(f"[SchemaMapper] Failed for {path.name}: {e}")
            mapping = {}

    # ===== 回退：使用原有的LLM映射 =====
    if not mapping:
        mapping_raw = await client.chat(cfg.map_model, [{
            "role": "user", "content": RIGHT_MAP_TMPL.format(
                header=header_str, row=row_str, targets=', '.join(feats), dname=path.name)
        }])
        mapping = parse_mapping(mapping_raw)

    # 获取join keys
    key_raw = await client.chat(cfg.map_model, [{
        "role": "user", "content": RIGHT_KEY_TMPL.format(
            dname=path.name, header=header_str, row=row_str)
    }])
    keys = parse_keys(key_raw)

    # 兜底：规范名≈同名列
    if not mapping:
        norm = lambda s: re.sub(r"[\s_\-]+", "", s).lower()
        hnorm = {norm(h): h for h in hdr_list}
        for f in feats:
            h = hnorm.get(norm(f))
            if h: mapping[f] = h

    # 模板别名规则再兜底（不改模板即不触发别名增强）
    if policy:
        mapping = policy.alias_autofix(hdr_list, mapping)

    # ===== 新增：仅在“点名列模式”下进行域感知纠偏（不影响原非点名场景）=====
    requested = extract_requested_columns_from_query(cfg.question)
    topic_kw = extract_domain_keyword(cfg.question)
    if requested:
        # 针对 id：如果用户确实点名了 id 且存在 domain_id，则优先替换为 domain id
        if any(_norm_token(c) == "id" for c in requested) and topic_kw:
            dom_id = _prefer_domain_id_from_headers(hdr_list, requested, topic_kw)
            if dom_id:
                # 只有当映射里的 id 不是这个 dom_id 时才替换；避免干扰已对的映射
                if "id" not in mapping or mapping.get("id") != dom_id:
                    mapping["id"] = dom_id

            # 加强 join key：把域 id 作为候选
            if dom_id and dom_id not in keys:
                keys.append(dom_id)

    return mapping, keys


def apply_mapping_and_rename(
        df: pl.DataFrame,
        mapping: dict,
        *,
        passthrough_all: bool = False,
        policy: DomainPolicy | None = None,
        keep_all_numeric: bool = True
) -> pl.DataFrame:
    """
    安全改名器：
    - 仅对存在的源列改名；
    - 同一目标名只保留第一条（后续冲突忽略）；
    - passthrough_all=True 时，把未被使用且不与特征名冲突的原列也带上；
    - keep_all_numeric=True 时，强制保留所有数值列
    """

    def _clean(val: str) -> str:
        s = re.split(r"\(index.*?\)", str(val))[0].split("|")[0].split(",")[0].strip()
        return s

    used_targets: set[str] = set()
    used_sources: set[str] = set()
    exprs: list[pl.Expr] = []

    # 1) 规范映射列 → 特征名
    for feat, src in mapping.items():
        s = _clean(src)
        if not s or s not in df.columns:
            continue
        if feat in used_targets:
            continue
        exprs.append(pl.col(s).alias(feat))
        used_targets.add(feat)
        used_sources.add(s)

    # 2) 直通策略
    passthrough_names: list[str] = []
    if policy and policy.passthrough_cols:
        passthrough_names.extend([c for c in policy.passthrough_cols if c in df.columns])

    if passthrough_all:
        for c in df.columns:
            if c in used_sources:
                continue
            if c in used_targets:
                continue
            passthrough_names.append(c)

    # 3) 保留所有数值列
    if keep_all_numeric:
        numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
        for c in numeric_cols:
            if c not in used_sources and c not in used_targets and c not in passthrough_names:
                passthrough_names.append(c)

    # 去重保序
    seen = set()
    clean_passthrough = []
    for c in passthrough_names:
        if c not in seen and c not in used_targets:
            seen.add(c)
            clean_passthrough.append(c)

    for c in clean_passthrough:
        exprs.append(pl.col(c))

    return df.select(exprs) if exprs else df.select([])


def choose_base_dataset(renamed_frames, join_keys):
    scored = [(sum(1 for k in join_keys if k in df.columns), df.width, p, df) for p, df in renamed_frames]
    scored.sort(reverse=True)
    _, _, p, df = scored[0]
    return p, df


def find_join_cols(cols_a, cols_b, join_keys, canon):
    common = [k for k in join_keys if k in cols_a and k in cols_b]
    if not common:
        common = [c for c in (canon or []) if c in cols_a and c in cols_b][:2]
    return common


def _coalesce_right_prefer(df: pl.DataFrame, *, suffix: str = "_r", skip_cols: list[str] | None = None) -> pl.DataFrame:
    """
    对于 join 后出现的一对同名列 c 和 c+suffix，使用“右侧优先（后来的表正确）”策略：
      c := if (c+suffix 非空) then c+suffix else c
    然后丢弃 c+suffix。
    skip_cols 为 join 键，永远不做覆盖。
    """
    if df.width == 0:
        return df
    skip = set(skip_cols or [])
    base_cols = [c for c in df.columns if c.endswith(suffix) is False and (c not in skip)]
    updates = []
    drops = []
    for c in base_cols:
        r = f"{c}{suffix}"
        if r in df.columns:
            # 右值按左列 dtype 尽量转换，非空即覆盖
            target_dt = df.schema.get(c, pl.Utf8)
            right_expr = pl.col(r).cast(target_dt, strict=False)
            left_expr = pl.col(c)
            updates.append(pl.when(right_expr.is_not_null()).then(right_expr).otherwise(left_expr).alias(c))
            drops.append(r)
        else:
            # 不涉及该列，原样带上
            updates.append(pl.col(c))
    # 把未处理的列也带上（包括 join 键、纯右侧新增列）
    rest = [c for c in df.columns if c not in {u.meta.output_name() for u in updates} and c not in drops]
    out = df.select([*updates, *[pl.col(c) for c in rest]])
    if drops:
        out = out.drop(drops)
    return out


# ============== Universal Merge（保留所有列与行） ==============
def _align_columns(df: pl.DataFrame, all_cols: list[str]) -> pl.DataFrame:
    add_expr = []
    for c in all_cols:
        if c not in df.columns:
            add_expr.append(pl.lit(None).alias(c))
    if add_expr:
        df = df.with_columns(add_expr)
    return df.select(all_cols)


def _choose_target_dtype(dtypes: list[pl.datatypes.DataType]) -> pl.datatypes.DataType:
    """Choose a safe common dtype for a column across frames."""
    nz = [dt for dt in dtypes if dt != pl.Null]
    if not nz:
        return pl.Utf8
    if any(dt in (pl.Utf8, pl.Categorical, pl.Binary) for dt in nz):
        return pl.Utf8
    if any(dt.is_numeric() for dt in nz):
        return pl.Float64
    if all(dt == pl.Boolean for dt in nz):
        return pl.Boolean
    if len(set(nz)) == 1 and list(set(nz))[0] in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return list(set(nz))[0]
    return pl.Utf8


def _unify_dtypes(frames: list[pl.DataFrame]) -> tuple[list[str], dict[str, pl.datatypes.DataType]]:
    all_cols: list[str] = []
    for f in frames:
        for c in f.columns:
            if c not in all_cols:
                all_cols.append(c)
    dtypes_map: dict[str, list[pl.datatypes.DataType]] = {c: [] for c in all_cols}
    for f in frames:
        for c in all_cols:
            dtypes_map[c].append(f.schema.get(c, pl.Null))
    target = {c: _choose_target_dtype(dts) for c, dts in dtypes_map.items()}
    return all_cols, target


def _cast_to_targets(df: pl.DataFrame, target_dtypes: dict[str, pl.datatypes.DataType]) -> pl.DataFrame:
    exprs = []
    for c, dt in target_dtypes.items():
        if c not in df.columns:
            exprs.append(pl.lit(None).cast(dt).alias(c))
        else:
            cur = df.schema[c]
            if cur != dt:
                exprs.append(pl.col(c).cast(dt, strict=False).alias(c))
            else:
                exprs.append(pl.col(c))
    return df.select([exprs[i] for i in range(len(exprs))])


def _outer_join_many(frames: list[pl.DataFrame], join_cols: list[str], policy: DomainPolicy):
    if not frames:
        return pl.DataFrame(), []
    merged = frames[0]
    for df in frames[1:]:
        jc = [c for c in join_cols if c in merged.columns and c in df.columns]
        if not jc:
            idx = next((i for i, f in enumerate(frames) if f is df), len(frames) - 1)
            return merged, frames[idx + 1:]
        merged = merged.unique(subset=jc, keep="first")
        df = df.unique(subset=jc, keep="first")
        if not policy.blowup_guard(merged, df, jc):
            idx = next((i for i, f in enumerate(frames) if f is df), len(frames) - 1)
            return merged, frames[idx + 1:]
        merged = merged.join(df, on=jc, how="full", suffix="_r")
        merged = _coalesce_right_prefer(merged, suffix="_r", skip_cols=jc)  # 右侧非空覆盖，然后删 *_r
    return merged, []


def universal_merge(renamed: list[Tuple[Path, pl.DataFrame]],
                    join_keys: list[str],
                    policy: DomainPolicy) -> Tuple[pl.DataFrame, dict]:
    if not renamed:
        return pl.DataFrame(), {}
    with_keys, without_keys = [], []
    for p, df in renamed:
        if any(k in df.columns for k in join_keys):
            with_keys.append((p, df))
        else:
            without_keys.append((p, df))

    stats = {
        "with_keys": [str(p) for p, _ in with_keys],
        "without_keys": [str(p) for p, _ in without_keys],
        "outer_join_chain": [],
        "vstack_files": []
    }

    merged_outer = None
    remaining = []
    if with_keys:
        base_p, base_df = choose_base_dataset(with_keys, join_keys)
        stats["outer_join_chain"].append(str(base_p))
        pool = [df for p, df in with_keys if p != base_p]
        merged_outer, leftover = _outer_join_many([base_df] + pool, join_keys, policy)
        for p, df in with_keys:
            if p != base_p:
                stats["outer_join_chain"].append(str(p))
        remaining = leftover
    else:
        merged_outer = pl.DataFrame()

    vstack_list: list[pl.DataFrame] = []
    if merged_outer is not None and merged_outer.width > 0:
        vstack_list.append(merged_outer)
    vstack_list.extend([df for df in remaining])
    vstack_list.extend([df for _, df in without_keys])

    if not vstack_list:
        return pl.DataFrame(), stats

    all_cols, target_dtypes = _unify_dtypes(vstack_list)
    aligned_casted = []
    for f in vstack_list:
        f2 = _align_columns(f, all_cols)
        f2 = _cast_to_targets(f2, target_dtypes)
        aligned_casted.append(f2)

    merged_all = aligned_casted[0]
    for f in aligned_casted[1:]:
        merged_all = merged_all.vstack(f, in_place=False)

    stats["vstack_files"] = [*(stats["without_keys"]), *[str(x) for x in stats["outer_join_chain"][1:]]]
    return merged_all, stats


# ================================ 主流程 ================================
async def process(cfg: Config):
    print(f"[Q] {cfg.question}")
    cache = LLMCache(cfg.cache_path)
    client = LLMClient(cache, cfg)

    # 1) 仅扫描本地文件（datas/ & datalake/；若存在 data/ 也混合）
    paths = _expand_globs_mixed(cfg.datasets)
    if not paths:
        print("[ERR] No datasets found under patterns:", cfg.datasets)
        await client.aclose();
        return

    # —— 文件上限初步裁剪（后面可能放宽） —— #
    if cfg.max_files and len(paths) > cfg.max_files:
        print(f"[SAFE] max_files: {len(paths)} -> {cfg.max_files}")
        paths = paths[:cfg.max_files]

    # 1.1 采样 headers_union（用于 LLM 细化 join 键）
    headers_union = []
    for p in paths[:50]:
        try:
            hdr, _ = preview_any(p)
            headers_union.extend([h.strip() for h in hdr.split(",") if h.strip()])
        except Exception:
            continue
    headers_union = list(dict.fromkeys(headers_union))

    # 0) 领域与特征（用于列语义对齐，不拉外部数据）
    templates = load_domain_templates()
    domain, feats, tpl_join, canon = await bootstrap_features_from_templates_llm_first(
        client, cfg.question, templates
    )

    # ===== 新增：使用完整数据集信息生成详细的dynamic template =====
    dynamic_template = None
    if _HAS_TEMPLATE_UTILS:
        try:
            # 准备数据集信息供template生成使用
            discovered_datasets = []
            for p in paths[:10]:  # 只用前10个数据集，避免prompt过长
                try:
                    hdr, sample_row = preview_any(p)
                    cols = [h.strip() for h in hdr.split(",") if h.strip()]
                    sample = dict(zip(cols, [s.strip() for s in sample_row.split(",")]))
                    discovered_datasets.append({
                        "path": str(p),
                        "columns": cols[:30],  # 限制列数
                        "sample": sample
                    })
                except Exception:
                    continue

            if discovered_datasets:
                tm = TemplateManager(client, cache_path=Path("templates/dynamic_templates.json"))
                dynamic_template = await tm.generate_template(cfg.question, discovered_datasets)

                # 用dynamic template的结果更新features和join_keys
                if dynamic_template.feature_sets:
                    feats = list(dynamic_template.feature_sets[0])
                if dynamic_template.join_keys:
                    tpl_join = list(dynamic_template.join_keys)
                if dynamic_template.canonical_order:
                    canon = list(dynamic_template.canonical_order)

                print(f"[DynamicTemplate] Generated template for domain: {dynamic_template.domain}")
                print(f"[DynamicTemplate] Alias rules: {len(dynamic_template.alias_rules)} rules")

                # 将dynamic_template存储供后续使用
                domain = dynamic_template.domain
        except Exception as e:
            print(f"[DynamicTemplate] Failed to generate detailed template: {e}")

    tpl_join = await llm_refine_join_keys_with_headers(client, cfg.question, tpl_join, headers_union)

    JOIN_KEYS = tpl_join or ["id", "name", "title", "entity_id", "entity_name", "date", "player_name",
                             "team_abbreviation"]
    CANON_FEATURES_ORDER = canon or []

    # —— 解析“点名列” —— #
    REQUESTED_COLS = extract_requested_columns_from_query(cfg.question)  # 例如 ["id","ph","quality"]
    if REQUESTED_COLS:
        # 将点名列放到 features 最前面（提升语义映射的命中率）
        feats = list(dict.fromkeys(REQUESTED_COLS + feats))
        # 点名列模式：为提高覆盖率，放宽抓取并关闭基于文件名的关键词过滤
        cfg.filter_by_query_keywords = False
        cfg.max_files = max(cfg.max_files, 50)
        cfg.max_rows_per_file = max(cfg.max_rows_per_file, 500_000)
        DOMAIN_KEYWORD = extract_domain_keyword(cfg.question)
    else:
        DOMAIN_KEYWORD = None

    print(f"[DOMAIN] {domain}\n[FEATURES] {feats[:10]}...\n[JOIN_KEYS] {JOIN_KEYS}")

    # —— 安全过滤：根据 query 关键词先过滤文件名（若开启且没被关闭） —— #
    if cfg.filter_by_query_keywords:
        kws = _keywords_from_query(cfg.question)
        if kws:
            def _hit(p: Path) -> bool:
                name = p.name.lower()
                return any(k in name for k in kws)

            filtered = [p for p in paths if _hit(p)]
            if filtered:
                print(f"[SAFE] filter_by_query_keywords: {len(paths)} -> {len(filtered)} by {kws}")
                paths = filtered

    # 策略 + 同源去重
    body = templates.get(domain, templates.get("fallback", {}))
    policy = DomainPolicy(body)
    orig_len = len(paths)
    paths = policy.dedupe_same_origin(paths)
    print(f"[INFO] Matched files: {orig_len} → after same-origin dedupe: {len(paths)}")
    for p in paths[:20]:
        print("  -", p)

    # 2) LLM 映射（支持dynamic_template指导）
    results = await tqdm_asyncio.gather(
        *[map_one_dataset(client, p, feats, cfg, policy, dynamic_template) for p in paths],
        total=len(paths), desc="Mapping"
    )

    # 3) 读取并改名（记录诊断信息）
    diagnostics = []
    renamed = []
    for path, (mapping, keys) in zip(paths, results):
        try:
            df_raw = _read_any_backend(path)
            # —— 单表行数上限，避免超大表引起 vstack OOM —— #
            if cfg.max_rows_per_file and df_raw.height > cfg.max_rows_per_file:
                df_raw = df_raw.head(cfg.max_rows_per_file)
            raw_rows = df_raw.height
        except Exception as e:
            print(f"[WARN] read fail {path.name}: {e}")
            continue

        # ✅ 宽松映射，保留上下文（不因点名列而收窄）
        df2 = apply_mapping_and_rename(
            df_raw,
            mapping,
            passthrough_all=True,
            policy=policy,
            keep_all_numeric=True
        )

        if df2.width == 0:
            print(f"[INFO] skip {path.name}: no columns after mapping")
            continue

        jks = [k for k in keys if k in df2.columns] or [k for k in JOIN_KEYS if k in df2.columns]
        uniq_keys = None
        if jks:
            try:
                uniq_keys = df2.unique(subset=jks).height
            except Exception:
                uniq_keys = None

        diagnostics.append({
            "file": str(path),
            "rows_raw": raw_rows,
            "rows_mapped": df2.height,
            "join_keys_used": jks,
            "rows_unique_on_keys": uniq_keys,
            "columns_after_map": df2.columns
        })
        renamed.append((path, df2))

    if not renamed:
        print("[WARN] no usable frames after mapping/rename")
        await client.aclose();
        return

    # === 点名列模式下：先用 LLM 做 per-file 相关性筛选（防止非 topic 的表混入） ===
    filtered_for_subset = renamed
    if REQUESTED_COLS:
        filtered_for_subset = []
        tasks = []
        for p, df in renamed:
            try:
                headers = df.columns
                sample_rows = df.head(3).to_dicts()
            except Exception:
                headers, sample_rows = [], []
            tasks.append(is_dataset_relevant_to_query(
                client, cfg.question, p.name, headers, sample_rows
            ))
        relevances = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="LLM relevance")
        for (p, df), ok in zip(renamed, relevances):
            if ok:
                filtered_for_subset.append((p, df))
        print(f"[RELEVANCE] keep {len(filtered_for_subset)}/{len(renamed)} files for requested-column union")

    # === 点名列并集抽取：语义选列（列名+值分布+主题词），并把源列别名为请求列名 ===
    requested_subset_df = None
    requested_subset_path = None
    if REQUESTED_COLS:
        # 主题关键词（如 wine / movie / nba）
        topic_keyword = await extract_topic_keyword(client, cfg.question)
        print(f"[TOPIC] keyword={topic_keyword}")

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        # （可选）将最相关的文件排到前面：文件名/列名是否包含主题词
        tk = _norm(topic_keyword) if topic_keyword else ""

        def _file_score(p: Path, df: pl.DataFrame) -> int:
            name_score = 2 if (tk and tk in _norm(p.name)) else 0
            head_score = 1 if (tk and any(tk in _norm(c) for c in df.columns)) else 0
            return name_score + head_score

        filtered_for_subset.sort(key=lambda x: _file_score(x[0], x[1]), reverse=True)

        subset_frames: list[pl.DataFrame] = []

        # 非 id 请求列，用于质量门控
        non_id_req = [c for c in REQUESTED_COLS if "id" not in c.lower()]

        for p, df in filtered_for_subset:
            headers = df.columns
            if not headers:
                continue

            # 语义打分选列
            chosen_map: dict[str, str] = {}
            for req in REQUESTED_COLS:
                best = _pick_best_header_for_req(df, req, topic_keyword)
                if best is not None:
                    chosen_map[req] = best

            # 质量门控：非 id 至少命中一个，且该列有效值比例 >= 2%
            def _enough(col: str) -> bool:
                try:
                    cnt = df.filter(
                        pl.col(col).is_not_null() &
                        (pl.col(col).cast(pl.Utf8) != "") &
                        (pl.col(col).cast(pl.Utf8) != "None")
                    ).height
                    return (float(cnt) / max(1.0, float(df.height))) >= 0.02
                except Exception:
                    return False

            if non_id_req:
                hit_non_ids = [chosen_map[r] for r in non_id_req if r in chosen_map]
                if not hit_non_ids or not any(_enough(c) for c in hit_non_ids):
                    continue

            if not chosen_map:
                continue

            # 构造子表：命中列 alias 为请求名，缺失列补 None
            exprs = []
            for feat in REQUESTED_COLS:
                src = chosen_map.get(feat)
                if src and src in df.columns:
                    exprs.append(pl.col(src).alias(feat))
                else:
                    exprs.append(pl.lit(None).alias(feat))
            sub = df.select(exprs)  # 列顺序 = REQUESTED_COLS
            subset_frames.append(sub)

        if subset_frames:
            # 统一 dtype，避免 vstack 冲突
            dtypes_map: dict[str, list[pl.datatypes.DataType]] = {c: [] for c in REQUESTED_COLS}
            for f in subset_frames:
                for c in REQUESTED_COLS:
                    dtypes_map[c].append(f.schema.get(c, pl.Null))
            target_dtypes = {c: _choose_target_dtype(dts) for c, dts in dtypes_map.items()}
            casted_frames = [_cast_to_targets(f, target_dtypes) for f in subset_frames]

            requested_subset_df = casted_frames[0]
            for s in casted_frames[1:]:
                requested_subset_df = requested_subset_df.vstack(s, in_place=False)
            try:
                requested_subset_df = requested_subset_df.unique(subset=REQUESTED_COLS, keep="first")
            except Exception:
                requested_subset_df = requested_subset_df.unique(keep="first")

            print(f"[REQUESTED-UNION] rows={requested_subset_df.height}, cols={requested_subset_df.width}")

    # 4A) 原“保守内连接”（审计对比，不变）
    base_path, merged_inner = choose_base_dataset(renamed, JOIN_KEYS)
    merged_safe = merged_inner
    print(f"[BASE/SAFE] {base_path.name}")

    for p, df in renamed:
        if p == base_path:
            continue
        join_cols = find_join_cols(merged_safe.columns, df.columns, JOIN_KEYS, CANON_FEATURES_ORDER)
        if not join_cols:
            print(f"[SAFE-JOIN] {p.name}: no common join keys — skip")
            continue

        if not policy.require_strong_keys(join_cols):
            print(
                f"[SAFE-JOIN-SKIP] {p.name}: weak join keys {join_cols} (need ≥{policy.join_min_keys}, strong={policy.strong_keys})")
            continue

        merged_safe = merged_safe.with_columns(
            [pl.col(k).cast(pl.Utf8) for k in join_cols if k in merged_safe.columns]).unique(subset=join_cols,
                                                                                             keep="first")
        df = df.with_columns([pl.col(k).cast(pl.Utf8) for k in join_cols if k in df.columns]).unique(subset=join_cols,
                                                                                                     keep="first")

        if not policy.blowup_guard(merged_safe, df, join_cols):
            print(f"[SAFE-JOIN-SKIP] {p.name}: potential blow-up on {join_cols} (>{policy.max_blowup_ratio}x)")
            continue

        print(f"[SAFE-JOIN] {p.name} ON {join_cols}")
        merged_safe = merged_safe.join(df, on=join_cols, how="inner", suffix="_r")
        merged_safe = _coalesce_right_prefer(merged_safe, suffix="_r", skip_cols=join_cols)  # 右侧非空覆盖，然后删 *_r

    # 4B) 通用合并（不变）
    try:
        merged_universal, u_stats = universal_merge(renamed, JOIN_KEYS, policy)
        print(f"[UNIVERSAL] rows={merged_universal.height}, cols={merged_universal.width}")
        merged = merged_universal if cfg.universal_merge else merged_safe
    except Exception as e:
        print(f"[UNIVERSAL-FAIL] fallback to SAFE INNER JOIN: {e}")
        u_stats = {"error": str(e)}
        merged = merged_safe

    # ====================================================================
    # 4.5) 两阶段增强：ER（实体解析）+ MVI（缺失值填补）
    # ====================================================================
    entity_report_path = None
    enrich_summary_path = None
    er_report_path = None

    # 识别领域
    auto_domain = await detect_domain_from_query_or_columns(
        cfg.question, list(merged.columns), llm_client=client
    )
    print(f"[DOMAIN] Detected: {auto_domain}")

    # 确定主键
    pk_candidates = [k for k in JOIN_KEYS if k in merged.columns]
    if not pk_candidates:
        pk_candidates = [c for c in merged.columns if merged.schema[c] == pl.Utf8][:2]
    print(f"[PRIMARY_KEYS] {pk_candidates}")

    # 确定可更新列（有缺失值的列）
    non_updatable = set(pk_candidates)
    allowed_update_cols = [c for c in merged.columns if c not in non_updatable
                           and merged.select(
        pl.col(c).is_null() | (pl.col(c).cast(pl.Utf8) == "")).to_series().any()]
    print(f"[UPDATABLE_COLS] {len(allowed_update_cols)} columns with missing values")

    # 采样（避免成本过高）
    sample_n = int(os.getenv("ER_MVI_SAMPLE_ROWS", "300"))
    sample_df = merged.head(sample_n) if sample_n else merged
    sample_rows = sample_df.to_dicts()

    # ====================================================================
    # 阶段1: ER (Entity Resolution) - LLM驱动
    # ====================================================================
    er_enabled = os.getenv("ER_ENABLED", "1") == "1"

    if er_enabled:
        try:
            print(f"\n[ER] Starting Entity Resolution...")
            er_provider = get_provider_for_auto_domain(auto_domain, llm_client=client)
            print(f"[ER] Provider: {er_provider.name}")

            # 执行ER
            resolved_rows, er_report = await er_provider.resolve_entities(
                rows=sample_rows,
                primary_keys=pk_candidates,
                entity_type=auto_domain,
                max_comparisons=500,
                confidence_threshold=0.75
            )

            if resolved_rows and len(resolved_rows) < len(sample_rows):
                print(f"[ER] ✓ Merged {len(sample_rows)} → {len(resolved_rows)} rows")
                sample_rows = resolved_rows

                # 保存ER报告
                if er_report:
                    er_path = Path(cfg.out).with_name(Path(cfg.out).stem + "_er_report.csv")
                    pl.from_dicts(er_report).write_csv(er_path)
                    er_report_path = str(er_path)
                    print(f"[ER] Report -> {er_path}")
            else:
                print(f"[ER] No duplicates found")

        except Exception as e:
            print(f"[WARN] ER failed: {e}")
            import traceback
            traceback.print_exc()

    # ====================================================================
    # 阶段2: MVI (Missing Value Imputation) - 外部API驱动
    # ====================================================================
    mvi_enabled = os.getenv("MVI_ENABLED", "1") == "1" and _HAS_MVI_PROVIDERS

    if mvi_enabled:
        try:
            print(f"\n[MVI] Starting Missing Value Imputation...")

            # 收集API密钥
            api_keys = {
                "tmdb": os.getenv("TMDB_API_KEY"),
                "omdb": os.getenv("OMDB_API_KEY"),
                "kaggle_username": os.getenv("KAGGLE_USERNAME"),
                "kaggle_key": os.getenv("KAGGLE_KEY"),
            }

            # 过滤掉空的API key
            api_keys = {k: v for k, v in api_keys.items() if v}

            if api_keys:
                print(f"[MVI] Available APIs: {list(api_keys.keys())}")
            else:
                print(f"[MVI] No API keys found, using Wikipedia only")

            # 获取MVI provider
            mvi_provider = get_mvi_provider_for_domain(
                domain=auto_domain,
                api_keys=api_keys
            )
            print(f"[MVI] Provider: {mvi_provider.name}")

            # 执行MVI
            enriched_rows = await mvi_provider.impute_missing(
                rows=sample_rows,
                primary_keys=pk_candidates,
                allowed_update_cols=allowed_update_cols
            )

            # 关闭provider
            await mvi_provider.aclose()

            if enriched_rows:
                # 生成before/after对比
                def _row_key(r: dict) -> tuple:
                    return tuple(r.get(k, "") for k in pk_candidates) if pk_candidates else (
                        hash(json.dumps(r, sort_keys=True)),)

                updates_map = {_row_key(r): r for r in enriched_rows}

                def _apply_row(i, r):
                    k = _row_key(r)
                    upd = updates_map.get(k, {})
                    out = dict(r)
                    for c in allowed_update_cols:
                        if c in upd:
                            v0 = r.get(c, None)
                            v1 = upd.get(c, None)
                            if (v0 is None or str(v0).strip() == "") and v1 not in (None, ""):
                                out[c] = v1
                    return out

                sample_rows_updated = [_apply_row(i, r) for i, r in enumerate(sample_rows)]

                # entity_report.csv (MVI changes)
                diffs = []
                for i in range(len(sample_rows)):
                    before = sample_rows[i]
                    after = sample_rows_updated[i]
                    for c in allowed_update_cols:
                        b = before.get(c, None)
                        a = after.get(c, None)
                        if (b in (None, "")) and (a not in (None, "")):
                            rec = {"row_index": i}
                            for k in pk_candidates:
                                rec[k] = after.get(k, "")
                            rec.update({"column": c, "before": b, "after": a})
                            diffs.append(rec)

                if diffs:
                    mvi_path = Path(cfg.out).with_name(Path(cfg.out).stem + "_mvi_report.csv")
                    base_keys = ["row_index"] + pk_candidates + ["column", "before", "after"]
                    with mvi_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=base_keys)
                        writer.writeheader()
                        for d in diffs:
                            writer.writerow({k: d.get(k, "") for k in base_keys})
                    entity_report_path = str(mvi_path)
                    print(f"[MVI] Changes -> {mvi_path} (rows={len(diffs)})")

                # enrich_summary.csv
                col_summary = []
                n = len(sample_rows)
                for c in allowed_update_cols:
                    filled = sum(
                        (sample_rows[i].get(c, None) in (None, "")) and
                        (sample_rows_updated[i].get(c, None) not in (None, ""))
                        for i in range(n)
                    )
                    if filled > 0:
                        col_summary.append({
                            "column": c,
                            "filled_cells": filled,
                            "sample_size": n,
                            "filled_ratio": round(filled / max(1, n), 4)
                        })
                if col_summary:
                    sum_path = Path(cfg.out).with_name(Path(cfg.out).stem + "_enrich_summary.csv")
                    with sum_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["column", "filled_cells", "sample_size", "filled_ratio"])
                        writer.writeheader()
                        for r in col_summary:
                            writer.writerow(r)
                    enrich_summary_path = str(sum_path)
                    print(f"[MVI] Summary -> {sum_path} (cols={len(col_summary)})")

                # 写回补全后的样本
                merged = pl.from_dicts(sample_rows_updated + merged.slice(len(sample_rows_updated)).to_dicts())

                rep_cols = sorted({r["column"] for r in diffs}) if diffs else sorted(set(allowed_update_cols))
                changed_cnt = len({d["row_index"] for d in diffs}) if diffs else 0
                print(
                    f"[MVI] ✓ Updated {changed_cnt}/{len(sample_rows)} rows on cols={rep_cols[:8]}{'...' if len(rep_cols) > 8 else ''}")

        except Exception as e:
            print(f"[WARN] MVI failed: {e}")
            import traceback
            traceback.print_exc()

    # >>> 导出点名列并集副产物（如有） <<<
    if REQUESTED_COLS and requested_subset_df is not None:
        req_out = Path(cfg.out).with_name(Path(cfg.out).stem + "_requested_subset.csv")
        requested_subset_df.write_csv(req_out, null_value="None")
        print(f"[OK] Requested-subset CSV -> {req_out} ({requested_subset_df.height}x{requested_subset_df.width})")
        requested_subset_path = str(req_out)

    # >>> 最终导出前，根据点名列裁剪（仅点名时生效；用“语义选列”再次映射 merged） <<<
    if REQUESTED_COLS:
        topic_keyword_final = await extract_topic_keyword(client, cfg.question)
        select_exprs = []
        select_log = {}
        for feat in REQUESTED_COLS:
            best = _pick_best_header_for_req(merged, feat, topic_keyword_final)
            if best and best in merged.columns:
                select_exprs.append(pl.col(best).alias(feat))
                select_log[feat] = best
            else:
                select_exprs.append(pl.lit(None).alias(feat))
                select_log[feat] = None
        merged = merged.select(select_exprs)
        print(f"[SELECT] requested-only via semantic mapping → {select_log} (topic={topic_keyword_final})")
    else:
        # 无点名列 → 保持原行为
        pass

    # 5) 导出主结果
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(out, null_value="None")

    # 附带保守内连接维度（对比记录）
    safe_info = {
        "rows": merged_safe.height,
        "cols": merged_safe.width,
        "columns": merged_safe.columns
    }

    meta = {
        "query": cfg.question,
        "domain": domain,
        "rows": merged.height,
        "columns": merged.columns,
        "matched_files": [str(p) for p, _ in renamed],
        "diagnostics": diagnostics,
        "join_keys": JOIN_KEYS,
        "merge_mode": "universal" if cfg.universal_merge else "safe_inner",
        "safe_inner_snapshot": safe_info,
        "universal_stats": u_stats,
        "requested_subset_csv": requested_subset_path,
        "requested_columns": REQUESTED_COLS if REQUESTED_COLS else [],
    }
    if entity_report_path:
        meta["entity_report"] = entity_report_path
    if enrich_summary_path:
        meta["enrich_summary_csv"] = enrich_summary_path
    if entity_report_path or enrich_summary_path:
        meta["enrich"] = {
            "provider": (provider.name if 'provider' in locals() else None),
            "domain": (auto_domain if 'auto_domain' in locals() else None),
            "sample_n": (len(sample_rows) if 'sample_rows' in locals() else None),
            "primary_keys": pk_candidates if 'pk_candidates' in locals() else [],
        }

    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] CSV -> {out} ({merged.height}x{merged.width})")
    print(f"[OK] Meta -> {out.with_suffix('.meta.json')}")
    await client.aclose()


# ===== LLM 文件相关性判定（通用、跨领域）=====
_RELEVANCE_PROMPT = """
You are a strict data curator. Given a user query and a dataset preview,
judge if this dataset is TOPIC-RELEVANT to the query (not merely sharing generic column names).

Return STRICT JSON ONLY:
{"relevant": true|false, "why": "<very short reason>"}

Rules:
- Consider file name, headers, and first rows.
- "Relevant" means the table CONTENT is about the query's topic/entity/domain, not just generic terms.
- Be conservative: if unsure, return false.
- Do NOT hallucinate columns that are not shown.
"""


async def is_dataset_relevant_to_query(
        client: LLMClient,
        query: str,
        file_name: str,
        headers: list[str],
        sample_rows: list[dict],
) -> bool:
    try:
        msg = [
            {"role": "user", "content": _RELEVANCE_PROMPT},
            {"role": "user", "content": json.dumps({
                "query": query,
                "file_name": file_name,
                "headers": headers[:80],
                "sample_rows": sample_rows[:3],
            }, ensure_ascii=False)}
        ]
        out = await client.chat("gpt-4o-mini", msg)
        obj = _extract_json_block(out)
        return bool(obj.get("relevant", False))
    except Exception:
        # 任何异常一律判为不相关，防止脏数据混入
        return False


# ============================== Loader & Main ==========================
def load_config(path: Optional[str] = None) -> Config:
    if not path:
        return DEFAULT_CONFIG
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        data = yaml.safe_load(p.read_text())
    elif p.suffix == ".json":
        data = json.load(p.open())
    elif p.suffix == ".py":
        spec = importlib.util.spec_from_file_location("cfg_module", p)
        mod = importlib.util.module_from_spec(spec);
        spec.loader.exec_module(mod)
        if not hasattr(mod, "CONFIG"):
            raise AttributeError("Python config must define CONFIG")
        data = mod.CONFIG
    else:
        raise ValueError(f"Unsupported config: {p.suffix}")
    allowed = Config.__annotations__.keys()
    data = {k: v for k, v in data.items() if k in allowed}
    return Config(**data)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(cfg_path)
    asyncio.run(process(cfg))