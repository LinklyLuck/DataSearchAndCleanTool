# validate_output_dataset.py
from __future__ import annotations
import json, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# ---------------- Templates I/O ----------------
TEMPLATES_DIR = Path("templates")
DOMAIN_DB     = TEMPLATES_DIR / "domain_templates.json"

def load_domain_templates(templates_dir: Path | None = None) -> dict:
    d = Path(templates_dir) if templates_dir else TEMPLATES_DIR
    db = d / "domain_templates.json"
    if not db.exists():
        # 最小 fallback，首次可用；pipeline 跑完后会把新领域写入
        fallback = {
            "_meta": {"version": 1, "note": "auto-created fallback"},
            "fallback": {
                "keywords": [],
                "feature_sets": [[
                    "target_variable","entity_name","entity_id","date","category",
                    "numeric_feature_1","numeric_feature_2","numeric_feature_3",
                    "text_feature","label_or_score"
                ]],
                "join_keys": ["entity_id","entity_name","date"],
                "canonical_order": [
                    "entity_name","entity_id","date","category",
                    "numeric_feature_1","numeric_feature_2","numeric_feature_3",
                    "text_feature","label_or_score"
                ]
            }
        }
        d.mkdir(parents=True, exist_ok=True)
        db.write_text(json.dumps(fallback, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.loads(db.read_text(encoding="utf-8"))

# -------------- Domain heuristics --------------
def pick_domain_by_columns(df_cols: List[str], templates: dict) -> str:
    """
    从列名出发，在模板库里选一个最匹配的 domain：
    - 优先匹配 join_keys 命中数最多的
    - 再看 canonical 覆盖度
    - 都不明显 → fallback
    """
    best = ("fallback", -1, -1)  # (domain, join_hits, canon_hits)
    cols_set = set(df_cols)
    for domain, body in templates.items():
        if domain.startswith("_"):
            continue
        join_keys = set([c for c in body.get("join_keys", [])])
        canon     = set([c for c in body.get("canonical_order", [])])
        join_hits = len(cols_set & join_keys)
        canon_hits= len(cols_set & canon)
        if (join_hits, canon_hits) > (best[1], best[2]):
            best = (domain, join_hits, canon_hits)
    return best[0] if best[1] >= 0 else "fallback"

# -------------- Validation core --------------
def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def validate_csv(
    csv_path: Path | str,
    templates: dict,
    out_report: Path | None = None,
    domain_hint: Optional[str] = None
) -> pd.DataFrame:
    """
    读取合并后的 CSV，生成一行 validation report（DataFrame）。
    如果 out_report 提供，则写出 CSV。
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    cols = list(df.columns)

    # 选择领域
    domain = domain_hint or pick_domain_by_columns(cols, templates)
    body   = templates.get(domain, templates.get("fallback", {}))

    join_keys   = body.get("join_keys", [])
    canonical   = body.get("canonical_order", [])

    # 指标 1：行列数
    n_rows, n_cols = df.shape

    # 指标 2：join_keys 覆盖率 & 非空率 & 唯一性
    jk_present = [k for k in join_keys if k in df.columns]
    jk_missing = [k for k in join_keys if k not in df.columns]

    jk_nonnull_ratio = {}
    jk_unique_ratio  = {}
    jk_dupes_count   = {}
    if jk_present:
        # 计算每个 join key 单列非空率、唯一率；以及全键重复行数
        for k in jk_present:
            s = df[k]
            jk_nonnull_ratio[k] = round(1.0 - (s.isna().mean()), 6)
            jk_unique_ratio[k]  = round(s.nunique(dropna=True) / max(1, len(s.dropna())), 6)
        if len(jk_present) == 1:
            dupes = df.duplicated(subset=jk_present, keep=False).sum()
            jk_dupes_count["__combined__"] = int(dupes)
        else:
            dupes = df.duplicated(subset=jk_present, keep=False).sum()
            jk_dupes_count["__combined__"] = int(dupes)

    # 指标 3：canonical 覆盖情况
    canon_present = [c for c in canonical if c in df.columns]
    canon_missing = [c for c in canonical if c not in df.columns]
    canon_coverage_ratio = round(len(canon_present) / max(1, len(canonical)), 6) if canonical else None

    # 指标 4：数值列基础质量（缺失率）
    numeric_cols = [c for c in cols if _is_numeric(df[c])]
    numeric_missing = {
        c: round(df[c].isna().mean(), 6) for c in numeric_cols
    }

    # 指标 5：示例值（前 3 行）
    sample_preview = df.head(3).to_dict(orient="list")

    # 汇总为单行记录
    record = {
        "file": str(p.name),
        "domain": domain,
        "rows": int(n_rows),
        "cols": int(n_cols),
        "join_keys_expected": "|".join(join_keys),
        "join_keys_present": "|".join(jk_present),
        "join_keys_missing": "|".join(jk_missing),
        "join_keys_nonnull_ratio": json.dumps(jk_nonnull_ratio, ensure_ascii=False),
        "join_keys_unique_ratio": json.dumps(jk_unique_ratio, ensure_ascii=False),
        "join_keys_dupes_combined": jk_dupes_count.get("__combined__", 0),
        "canonical_expected": "|".join(canonical),
        "canonical_present": "|".join(canon_present),
        "canonical_missing": "|".join(canon_missing),
        "canonical_coverage_ratio": canon_coverage_ratio if canon_coverage_ratio is not None else "",
        "numeric_cols": "|".join(numeric_cols),
        "numeric_missing_ratio": json.dumps(numeric_missing, ensure_ascii=False),
        "sample_preview": json.dumps(sample_preview, ensure_ascii=False)
    }

    out_df = pd.DataFrame([record])

    if out_report:
        out_path = Path(out_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_df

# ---------------- CLI ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Validate merged CSV and emit validation_report.csv")
    ap.add_argument("--csv", required=True, help="Merged CSV path")
    ap.add_argument("--templates", default="templates", help="Templates directory")
    ap.add_argument("--out", default="outputs/validation_report.csv", help="Path to write report CSV")
    ap.add_argument("--domain", default=None, help="Optional domain hint (e.g., movie/stock/ecommerce)")
    args = ap.parse_args()

    templates = load_domain_templates(Path(args.templates))
    df = validate_csv(args.csv, templates, out_report=Path(args.out), domain_hint=args.domain)
    print(f"[OK] Validation report -> {args.out}")
    # 打印一行摘要
    rec = df.iloc[0].to_dict()
    print(f"[SUMMARY] rows={rec['rows']}, cols={rec['cols']}, domain={rec['domain']}, "
          f"join_keys_present={rec['join_keys_present']}, canonical_coverage={rec['canonical_coverage_ratio']}")

if __name__ == "__main__":
    main()
