# the_pipeline_v2.py — local-only mining → template-aware mapping → safe join → enrich (ER/MVI) → CSV
from __future__ import annotations
import os, re, sys, json, csv, asyncio, hashlib, sqlite3, importlib.util, glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import httpx
import polars as pl
from tqdm.asyncio import tqdm_asyncio

# === 外部增强：仅"就地补全"，不新增行/列 ===
from enrichment_providers import (
    get_provider_for_auto_domain,
    detect_domain_from_query_or_columns,
)

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
    max_files: int = 20                 # 一次最多合并多少个文件（默认20）
    max_rows_per_file: int = 200_000    # 每个文件最多读取多少行（超出截断）
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


async def bootstrap_features_from_templates_llm_first(client: LLMClient, query: str, templates: dict) -> tuple[
    str, list[str], list[str], list[str]]:
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


# ======================== 通配/绝对路径安全展开 ============================
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


async def map_one_dataset(client, path, feats, cfg, policy: DomainPolicy | None = None):
    header_str, row_str = preview_any(path)
    hdr_list = [h.strip() for h in header_str.split(",")]
    mapping_raw = await client.chat(cfg.map_model, [{
        "role": "user", "content": RIGHT_MAP_TMPL.format(
            header=header_str, row=row_str, targets=', '.join(feats), dname=path.name)
    }])
    key_raw = await client.chat(cfg.map_model, [{
        "role": "user", "content": RIGHT_KEY_TMPL.format(
            dname=path.name, header=header_str, row=row_str)
    }])
    mapping = parse_mapping(mapping_raw)
    keys = parse_keys(key_raw)

    # 兜底：规范名≈同名列
    if not mapping:
        norm = lambda s: re.sub(r"[\s_\-]+", "", s).lower()
        hnorm = {norm(h): h for h in hdr_list}
        for f in feats:
            h = hnorm.get(norm(f))
            if h: mapping[f] = h

    # 模板别名规则再兜底
    if policy:
        mapping = policy.alias_autofix(hdr_list, mapping)

    return mapping, keys


def apply_mapping_and_rename(
        df: pl.DataFrame,
        mapping: dict,
        *,
        passthrough_all: bool = False,
        policy: DomainPolicy | None = None,
        keep_all_numeric: bool = True  # 👈 新增参数
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

    # 👇 新增：强制保留所有数值列
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
    # 过滤 Null
    nz = [dt for dt in dtypes if dt != pl.Null]
    if not nz:
        return pl.Utf8
    # 若任意为字符串/类别，统一为 Utf8
    if any(dt in (pl.Utf8, pl.Categorical, pl.Binary) for dt in nz):
        return pl.Utf8
    # 只要出现过数值，统一为 Float64（可容纳 Int/Float）
    if any(dt.is_numeric() for dt in nz):
        return pl.Float64
    # 只有布尔
    if all(dt == pl.Boolean for dt in nz):
        return pl.Boolean
    # 只有同一种时间类型
    if len(set(nz)) == 1 and list(set(nz))[0] in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return list(set(nz))[0]
    # 其它混合，退回 Utf8
    return pl.Utf8

def _unify_dtypes(frames: list[pl.DataFrame]) -> tuple[list[str], dict[str, pl.datatypes.DataType]]:
    """Compute union of columns and a target dtype per column."""
    all_cols: list[str] = []
    for f in frames:
        for c in f.columns:
            if c not in all_cols:
                all_cols.append(c)
    # 汇总每列的 dtype
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
            # 无公共键，无法外连接，交给 vstack 处理（返回剩余帧）
            idx = next((i for i, f in enumerate(frames) if f is df), len(frames) - 1)
            return merged, frames[idx+1:]
        # 去重并外连接
        merged = merged.unique(subset=jc, keep="first")
        df = df.unique(subset=jc, keep="first")
        if not policy.blowup_guard(merged, df, jc):
            # 避免爆炸；提前返回，剩余交给 vstack
            idx = next((i for i, f in enumerate(frames) if f is df), len(frames) - 1)
            return merged, frames[idx+1:]
        merged = merged.join(df, on=jc, how="full", suffix="_r")
        # 去掉 _r 重复列
        dup = [c for c in merged.columns if c.endswith("_r") and c[:-2] in merged.columns]
        if dup:
            merged = merged.drop(dup)
    return merged, []

def universal_merge(renamed: list[Tuple[Path, pl.DataFrame]],
                    join_keys: list[str],
                    policy: DomainPolicy) -> Tuple[pl.DataFrame, dict]:
    """
    目标：
      1) 能 join 的尽量用外连接横向合并（保留列）
      2) 不能 join 的，纵向 vstack（保留行）
      3) 最终保留全体列与行；缺失以 None 体现
    返回：
      merged_all, stats
    """
    if not renamed:
        return pl.DataFrame(), {}

    # 分组
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

    # 外连接链
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

    # 需要纵向追加的帧
    vstack_list: list[pl.DataFrame] = []
    if merged_outer is not None and merged_outer.width > 0:
        vstack_list.append(merged_outer)
    vstack_list.extend([df for df in remaining])
    vstack_list.extend([df for _, df in without_keys])

    if not vstack_list:
        return pl.DataFrame(), stats

    # 统一列集合 + 目标 dtype
    all_cols, target_dtypes = _unify_dtypes(vstack_list)

    # 对齐并统一 dtype
    aligned_casted = []
    for f in vstack_list:
        f2 = _align_columns(f, all_cols)
        f2 = _cast_to_targets(f2, target_dtypes)
        aligned_casted.append(f2)

    # 纵向合并
    merged_all = aligned_casted[0]
    for f in aligned_casted[1:]:
        merged_all = merged_all.vstack(f, in_place=False)

    # 填 None
    #merged_all = merged_all.fill_null(None)
    stats["vstack_files"] = [*(stats["without_keys"]), *[str(x) for x in stats["outer_join_chain"][1:]]]
    return merged_all, stats



# ================================ 主流程 ================================
async def process(cfg: Config):
    print(f"[Q] {cfg.question}")
    cache = LLMCache(cfg.cache_path)
    client = LLMClient(cache, cfg)

    # 1) 仅扫描本地文件（datas/ & datalake/）
    paths = _expand_globs(cfg.datasets)
    if not paths:
        print("[ERR] No datasets found under patterns:", cfg.datasets)
        await client.aclose();
        return

    # —— 安全过滤：根据 query 关键词先过滤文件名，防止把不相关大表都并进来 —— #
    if cfg.filter_by_query_keywords:
        kws = _keywords_from_query(cfg.question)
        if kws:
            def _hit(p: Path) -> bool:
                name = p.name.lower()
                return any(k in name for k in kws)
            filtered = [p for p in paths if _hit(p)]
            # 如果全被过滤掉，退回原集合，避免误杀
            if filtered:
                print(f"[SAFE] filter_by_query_keywords: {len(paths)} -> {len(filtered)} by {kws}")
                paths = filtered

    # —— 文件上限：防止一次拉太多 —— #
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
    tpl_join = await llm_refine_join_keys_with_headers(client, cfg.question, tpl_join, headers_union)

    JOIN_KEYS = tpl_join or ["id", "name", "title", "entity_id", "entity_name", "date", "player_name",
                             "team_abbreviation"]
    CANON_FEATURES_ORDER = canon or []

    print(f"[DOMAIN] {domain}\n[FEATURES] {feats[:10]}...\n[JOIN_KEYS] {JOIN_KEYS}")

    # 策略 + 同源去重
    body = templates.get(domain, templates.get("fallback", {}))
    policy = DomainPolicy(body)
    orig_len = len(paths)
    paths = policy.dedupe_same_origin(paths)
    print(f"[INFO] Matched files: {orig_len} → after same-origin dedupe: {len(paths)}")
    for p in paths[:20]:
        print("  -", p)

    # 2) LLM 映射
    results = await tqdm_asyncio.gather(
        *[map_one_dataset(client, p, feats, cfg, policy) for p in paths],
        total=len(paths), desc="Mapping"
    )

    # 3) 读取并改名（记录诊断信息）
    diagnostics = []
    renamed = []
    for path, (mapping, keys) in zip(paths, results):
        try:
            df_raw = read_any_df(path)
            # —— 单表行数上限，避免超大表引起 vstack OOM —— #
            if cfg.max_rows_per_file and df_raw.height > cfg.max_rows_per_file:
                df_raw = df_raw.head(cfg.max_rows_per_file)
            raw_rows = df_raw.height
        except Exception as e:
            print(f"[WARN] read fail {path.name}: {e}")
            continue

        df2 = apply_mapping_and_rename(
            df_raw,
            mapping,
            passthrough_all=True,
            policy=policy,
            keep_all_numeric=True  # 👈 启用数值列保留
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

    # 4A) —— 原“保守内连接”路径（不改变原有功能），用于审计对比 —— #
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

        merged_safe = merged_safe.with_columns([pl.col(k).cast(pl.Utf8) for k in join_cols if k in merged_safe.columns]) \
            .unique(subset=join_cols, keep="first")
        df = df.with_columns([pl.col(k).cast(pl.Utf8) for k in join_cols if k in df.columns]) \
            .unique(subset=join_cols, keep="first")

        if not policy.blowup_guard(merged_safe, df, join_cols):
            print(f"[SAFE-JOIN-SKIP] {p.name}: potential blow-up on {join_cols} (>{policy.max_blowup_ratio}x)")
            continue

        print(f"[SAFE-JOIN] {p.name} ON {join_cols}")
        merged_safe = merged_safe.join(df, on=join_cols, how="inner", suffix="_r")
        dup = [c for c in merged_safe.columns if c.endswith("_r") and c[:-2] in merged_safe.columns]
        if dup:
            merged_safe = merged_safe.drop(dup)

    # 4B) —— 新增“通用合并”路径：外连接 + 对齐列纵向合并（保留所有行与列）—— #
    try:
        merged_universal, u_stats = universal_merge(renamed, JOIN_KEYS, policy)
        print(f"[UNIVERSAL] rows={merged_universal.height}, cols={merged_universal.width}")
        merged = merged_universal if cfg.universal_merge else merged_safe
    except Exception as e:
        print(f"[UNIVERSAL-FAIL] fallback to SAFE INNER JOIN: {e}")
        u_stats = {"error": str(e)}
        merged = merged_safe

    # 4.5) —— 仅对"现有行、现有列"执行 ER/MVI（Wikipedia + Kaggle），不新增行/列 —— #
    entity_report_path = None
    enrich_summary_path = None
    try:
        auto_domain = await detect_domain_from_query_or_columns(
            cfg.question, list(merged.columns), llm_client=client
        )
        provider = get_provider_for_auto_domain(auto_domain)
        print(f"[ENRICH] provider={provider.name}, domain={auto_domain}")

        pk_candidates = [k for k in JOIN_KEYS if k in merged.columns]
        if not pk_candidates:
            pk_candidates = [c for c in merged.columns if merged.schema[c] == pl.Utf8][:2]

        non_updatable = set(pk_candidates)
        allowed_update_cols = [c for c in merged.columns if c not in non_updatable
                               and merged.select(
            pl.col(c).is_null() | (pl.col(c).cast(pl.Utf8) == "")).to_series().any()]

        sample_n = 300
        sample_df = merged.head(sample_n) if sample_n else merged
        sample_rows = sample_df.to_dicts()

        enrich_rows = await provider.enrich_rows(
            rows=sample_rows,
            primary_keys=pk_candidates,
            allowed_update_cols=allowed_update_cols,
            enrich_only=True,
            no_title_change=True
        )

        if enrich_rows:
            # --- 报告：行级 before/after 与列级汇总 ---
            def _row_key(r: dict) -> tuple:
                return tuple(r.get(k, "") for k in pk_candidates) if pk_candidates else (
                    hash(json.dumps(r, sort_keys=True)),)

            updates_map = {_row_key(r): r for r in enrich_rows}

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

            # === 生成 entity_report.csv（行级 diff 明细）===
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
                er_path = Path(cfg.out).with_name(Path(cfg.out).stem + "_entity_report.csv")
                # 统一字段顺序
                base_keys = ["row_index"] + pk_candidates + ["column", "before", "after"]
                # 写 CSV
                with er_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=base_keys)
                    writer.writeheader()
                    for d in diffs:
                        writer.writerow({k: d.get(k, "") for k in base_keys})
                entity_report_path = str(er_path)
                print(f"[ENRICH] entity report -> {er_path} (rows={len(diffs)})")

            # === 生成 enrich_summary.csv（列级汇总）===
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
                print(f"[ENRICH] enrich summary -> {sum_path} (cols={len(col_summary)})")

            # === 写回补全后的样本 ===
            merged = pl.from_dicts(sample_rows_updated + merged.slice(len(sample_rows_updated)).to_dicts())

            # 控台摘要
            rep_cols = sorted({r["column"] for r in diffs}) if diffs else sorted(set(allowed_update_cols))
            changed_cnt = len({d["row_index"] for d in diffs}) if diffs else 0
            print(
                f"[ENRICH] updated_rows={changed_cnt}/{len(sample_rows)} on cols={rep_cols[:8]}{'...' if len(rep_cols) > 8 else ''}")

    except Exception as e:
        print("[WARN] ER/MVI skipped:", e)

    # 5) 导出
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(out, null_value="None")

    # 附带保守内连接的维度（不改变原行为，只作为对比记录）
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
        "universal_stats": u_stats
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
