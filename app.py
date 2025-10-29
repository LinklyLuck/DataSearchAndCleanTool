# app.py -- Streamlit Web UI for your dataset-merge pipeline (+ ER/MVI + VLDB Validation)
import os, sys, json, time, zipfile, io, csv, subprocess
from pathlib import Path
import streamlit as st
from subprocess import CalledProcessError

ROOT = Path(__file__).parent.resolve()
PIPELINE = ROOT / "the_pipeline_v2.py"
VALIDATOR_V2 = ROOT / "validation_report_v2.py"  # NEW: VLDB-grade validator
DEFAULT_DATAS_GLOB = "datalake/*.{csv,json,ndjson,jsonl,xlsx,xls}"
RESULTS_DIR = ROOT / "results_webui"
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="DataSearchTool Web UI", layout="wide")
st.title("🧠 DataSearchTool – Natural Language Data Discovery · Auto-Merge · ER/MVI · VLDB Validation")

# ---------------------- Sidebar: Basic Config ----------------------
st.sidebar.header("⚙️ Configuration")
api_base = st.sidebar.text_input("LLM API Base URL", value="https://goapi.gptnb.ai/v1/chat/completions")
api_key = st.sidebar.text_input("LLM API Key (or set GPTNB_API_KEY env var)", type="password",
                                value=os.getenv("GPTNB_API_KEY", ""))
model = st.sidebar.text_input("Model Name (shared for left/map/join)", value="gpt-4o-mini")
max_conc = st.sidebar.number_input("max_concurrency", 1, 16, 4)
graph_out = st.sidebar.checkbox("Export join graph (.dot)", value=False)

# ---------------------- Sidebar: External API Enhancement ----------------------
st.sidebar.header("🔐 External API Enhancement (Optional)")
st.sidebar.caption("These keys enable ER/MVI to use domain-specific data sources automatically.")
k_user = st.sidebar.text_input("Kaggle Username (KAGGLE_USERNAME)", value=os.getenv("KAGGLE_USERNAME", ""))
k_key = st.sidebar.text_input("Kaggle API Key (KAGGLE_KEY)", type="password", value=os.getenv("KAGGLE_KEY", ""))
omdb_key = st.sidebar.text_input("OMDb API Key (OMDB_API_KEY)", type="password", value=os.getenv("OMDB_API_KEY", ""))
tmdb_key = st.sidebar.text_input("TMDb API Key (TMDB_API_KEY)", type="password", value=os.getenv("TMDB_API_KEY", ""))

# ---------------------- Sidebar: Cleaning & Validation ----------------------
st.sidebar.header("🧽 Cleaning & Validation")
perform_cleaning = st.sidebar.checkbox("Enable data cleaning & external validation", value=False)
cleaning_mode = st.sidebar.selectbox(
    "Cleaning mode",
    options=["comprehensive", "llm", "type", "minimal"],
    index=0,
    disabled=not perform_cleaning
)
validation_source = st.sidebar.selectbox(
    "Validation source",
    options=["wikipedia", "tmdb", "omdb", "comprehensive"],
    index=0,
    disabled=not perform_cleaning
)
validation_cols_text = st.sidebar.text_input(
    "Columns to validate (comma separated, optional)",
    value="",
    disabled=not perform_cleaning
)

validation_columns = None
if perform_cleaning:
    cols = [c.strip() for c in validation_cols_text.split(",") if c.strip()]
    validation_columns = cols if cols else None

# ---------------------- Queries Input ----------------------
st.subheader("📝 Natural Language Queries (one per line)")
text_queries = st.text_area(
    "Examples:\n"
    "I need a movie dataset with title, year, director, genre and budget\n"
    "Get a dataset for predicting movie revenue using numeric features\n"
    "Find stock OHLC tables for AAPL with dates and volumes",
    height=150
)

st.write("Or upload `queries_sig.csv` (must contain column **NL Questions**):")
csv_file = st.file_uploader("Upload queries CSV", type=["csv"])

queries = []
if csv_file is not None:
    content = csv_file.read().decode("utf-8", errors="ignore").splitlines()
    reader = csv.DictReader(content)
    for row in reader:
        if "NL Questions" in row and row["NL Questions"].strip():
            queries.append(row["NL Questions"].strip())

extra_lines = [q.strip() for q in text_queries.splitlines() if q.strip()]
queries.extend(extra_lines)
queries = [q for q in queries if q]

# ---------------------- Upload Data (Optional) ----------------------
st.subheader("📂 Upload Data (Optional)")
st.caption(
    "Optional; you can also use existing files in `datalake/*.{csv,json,ndjson,jsonl,xlsx,xls}`. Uploaded files will be saved to `datas/` for processing.")
uploads = st.file_uploader(
    "Drag and drop multiple CSV/JSON/NDJSON/JSONL/Excel files",
    type=["csv", "json", "ndjson", "jsonl", "xlsx", "xls"],
    accept_multiple_files=True
)
if uploads:
    data_dir = ROOT / "datas"
    data_dir.mkdir(exist_ok=True)
    for f in uploads:
        (data_dir / f.name).write_bytes(f.read())
    st.success(f"Saved to {data_dir} ({len(uploads)} files)")

# ---------------------- Run Button ----------------------
run = st.button("🚀 Start Processing", type="primary", disabled=(len(queries) == 0))
log_container = st.container()


def make_slug(idx: int, question: str) -> str:
    base = ''.join(ch if ch.isalnum() else '_' for ch in question.lower())[:40] or 'q'
    return f"q{idx}_{base}"


def run_one_query(idx: int, question: str) -> dict:
    """
    Process a single query: run pipeline + VLDB validation
    """
    slug = make_slug(idx, question)
    cfg_path = RESULTS_DIR / f"{slug}.json"
    out_csv = RESULTS_DIR / f"{slug}_merged.csv"
    dot_path = RESULTS_DIR / f"{slug}_graph.dot" if graph_out else ""
    meta_path = out_csv.with_suffix(".meta.json")

    # Assemble default data sources
    datasets_globs = [
        "datas/**/*.{csv,json,ndjson,jsonl,xlsx,xls}",
        "datalake/**/*.{csv,json,ndjson,jsonl,xlsx,xls}",
    ]

    cfg = {
        "question": question,
        "datasets": datasets_globs,
        "out": str(out_csv),
        "graph_out": str(dot_path),
        "left_model": model,
        "map_model": model,
        "join_model": model,
        "max_concurrency": int(max_conc),
        "api_base": api_base,
        "api_key": api_key or os.getenv("GPTNB_API_KEY", ""),
    }
    if perform_cleaning:
        cfg.update({
            "enable_cleaning": True,
            "cleaning_mode": cleaning_mode,
            "enable_validation": True,
            "validation_source": validation_source,
            "validation_columns": validation_columns,
            "validation_api_keys": {
                k: v for k, v in {
                    "tmdb": tmdb_key,
                    "omdb": omdb_key,
                }.items() if v
            }
        })
    else:
        cfg.update({
            "enable_cleaning": False,
            "enable_validation": False,
        })
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build subprocess environment: LLM + external enhancement
    env_for_child = os.environ.copy()
    if api_key:
        env_for_child["GPTNB_API_KEY"] = api_key
    if k_user: env_for_child["KAGGLE_USERNAME"] = k_user
    if k_key:  env_for_child["KAGGLE_KEY"] = k_key
    if omdb_key: env_for_child["OMDB_API_KEY"] = omdb_key
    if tmdb_key: env_for_child["TMDB_API_KEY"] = tmdb_key

    # Run main pipeline
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(PIPELINE), str(cfg_path)],
        capture_output=True,
        text=True,
        env=env_for_child,
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    out_csv_exists = out_csv.exists()
    meta_exists = meta_path.exists()
    dot_exists = (Path(dot_path).exists() if graph_out and dot_path else False)

    summary = {
        "question": question,
        "slug": slug,
        "elapsed_s": round(elapsed, 2),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "out_csv": str(out_csv) if out_csv_exists else "",
        "graph_dot": str(dot_path) if dot_exists else "",
        "meta_json": str(meta_path) if meta_exists else "",
        "entity_report": "",
        "validation_report_csv": "",
        "validation_report_html": "",
        "validation_score": None,
        "validation_stdout": "",
        "validation_stderr": "",
        "cleaning_report": "",
        "cleaning_summary": {},
        "validation_provider_report": "",
        "validation_summary": {},
    }

    # Read entity_report from meta (if exists)
    if meta_exists:
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            erp = meta.get("entity_report", "")
            if erp:
                erp_path = Path(erp)
                if not erp_path.is_absolute() and out_csv_exists:
                    erp_path = out_csv.with_name(out_csv.stem + "_entity_report.csv")
                if erp_path.exists():
                    summary["entity_report"] = str(erp_path)
            clean_report = meta.get("cleaning_report", "")
            if clean_report:
                clean_path = Path(clean_report)
                if not clean_path.is_absolute() and out_csv_exists:
                    clean_path = out_csv.with_name(out_csv.stem + "_cleaning_report.json")
                if clean_path.exists():
                    summary["cleaning_report"] = str(clean_path)
            validation_provider_rep = meta.get("validation_report", "")
            if validation_provider_rep:
                vpr_path = Path(validation_provider_rep)
                if not vpr_path.is_absolute() and out_csv_exists:
                    vpr_path = out_csv.with_name(out_csv.stem + "_validation_provider_report.json")
                if vpr_path.exists():
                    summary["validation_provider_report"] = str(vpr_path)
            if isinstance(meta.get("cleaning"), dict):
                summary["cleaning_summary"] = meta["cleaning"]
            if isinstance(meta.get("validation"), dict):
                summary["validation_summary"] = meta["validation"]
        except Exception:
            pass

    # Fallback: infer entity_report path if meta doesn't have it
    if not summary["entity_report"] and out_csv_exists:
        guess_rep = out_csv.with_name(out_csv.stem + "_entity_report.csv")
        if guess_rep.exists():
            summary["entity_report"] = str(guess_rep)

    # —— Generate VLDB-Grade Validation Report (CSV + HTML) ——
    report_csv = RESULTS_DIR / f"{slug}_validation_report.csv"
    report_html = RESULTS_DIR / f"{slug}_validation_report.html"

    if out_csv_exists and VALIDATOR_V2.exists() and meta_exists:
        try:
            vr = subprocess.run(
                [
                    sys.executable, str(VALIDATOR_V2),
                    "--csv", str(out_csv),
                    "--meta", str(meta_path),
                    "--out", str(RESULTS_DIR / f"{slug}_validation_report"),
                    "--format", "both",
                ],
                capture_output=True,
                text=True,
                env=env_for_child,
                cwd=str(ROOT),
            )

            if vr.returncode == 0:
                # Parse overall score from stdout
                overall_score = None
                for line in vr.stdout.splitlines():
                    if "Overall Quality Score:" in line:
                        try:
                            overall_score = float(line.split(":")[-1].strip().split("/")[0])
                        except:
                            pass

                # Check if files were generated
                if report_csv.exists():
                    summary["validation_report_csv"] = str(report_csv)
                if report_html.exists():
                    summary["validation_report_html"] = str(report_html)

                summary["validation_score"] = overall_score
                summary["validation_stdout"] = vr.stdout[-2000:]
            else:
                summary["validation_stdout"] = vr.stdout[-2000:]
                summary["validation_stderr"] = vr.stderr[-2000:]
        except Exception as e:
            summary["validation_stderr"] = f"Validation error: {e}"
    else:
        if not VALIDATOR_V2.exists():
            summary["validation_stderr"] = "validation_report_v2.py not found, skipping validation."
        elif not meta_exists:
            summary["validation_stderr"] = "Meta JSON not found, skipping validation."

    return summary


if run:
    st.toast(f"Starting processing ({len(queries)} queries)", icon="✅")
    results = []
    progress = st.progress(0.0)

    for i, q in enumerate(queries, 1):
        with log_container.expander(f"[{i}/{len(queries)}] {q}", expanded=True):
            res = run_one_query(i, q)
            results.append(res)

            if res["returncode"] == 0:
                st.success(f"✓ Completed in {res['elapsed_s']}s")

                # Merged output
                if res["out_csv"]:
                    st.write(f"📊 Merged Output: `{res['out_csv']}`")
                    st.download_button(
                        "⬇️ Download Merged CSV",
                        data=Path(res["out_csv"]).read_bytes(),
                        file_name=Path(res["out_csv"]).name,
                        mime="text/csv"
                    )

                # Meta
                if res["meta_json"]:
                    try:
                        meta = json.loads(Path(res["meta_json"]).read_text("utf-8"))
                        with st.expander("📋 Metadata (matched_files / diagnostics / enrichment)", expanded=False):
                            st.json(meta)
                    except Exception as e:
                        st.caption(f"Failed to read meta: {e}")

                # Join Graph
                if res["graph_dot"]:
                    st.write(f"🔗 Join Graph: `{res['graph_dot']}`")
                    st.download_button(
                        "⬇️ Download DOT Graph",
                        data=Path(res["graph_dot"]).read_bytes(),
                        file_name=Path(res["graph_dot"]).name,
                        mime="text/vnd.graphviz"
                    )

                # ER/MVI Report
                if res.get("entity_report"):
                    rp = Path(res["entity_report"])
                    if rp.exists():
                        st.write(f"🔍 ER/MVI Sample Report: `{rp}`")
                        st.download_button(
                            "⬇️ Download ER/MVI Report (CSV)",
                            data=rp.read_bytes(),
                            file_name=rp.name,
                            mime="text/csv"
                        )

                # Cleaning stage report (if enabled)
                if res.get("cleaning_report") or res.get("cleaning_summary"):
                    with st.expander("🧽 Cleaning Summary", expanded=False):
                        if res.get("cleaning_summary"):
                            st.json(res["cleaning_summary"])
                        if res.get("cleaning_report"):
                            clean_path = Path(res["cleaning_report"])
                            if clean_path.exists():
                                st.download_button(
                                    "⬇️ Download Cleaning Report (JSON)",
                                    data=clean_path.read_bytes(),
                                    file_name=clean_path.name,
                                    mime="application/json"
                                )

                # External validation provider report (post-cleaning)
                if res.get("validation_provider_report") or res.get("validation_summary"):
                    with st.expander("✅ External Validation Summary", expanded=False):
                        if res.get("validation_summary"):
                            st.json(res["validation_summary"])
                        if res.get("validation_provider_report"):
                            vpr = Path(res["validation_provider_report"])
                            if vpr.exists():
                                st.download_button(
                                    "⬇️ Download External Validation Report (JSON)",
                                    data=vpr.read_bytes(),
                                    file_name=vpr.name,
                                    mime="application/json"
                                )

                # ═══════════════════════════════════════════════════════════
                # 🎓 VLDB Validation Report (NEW)
                # ═══════════════════════════════════════════════════════════
                if res.get("validation_score") is not None:
                    score = res["validation_score"]
                    st.metric(
                        label="🎓 VLDB Validation Score",
                        value=f"{score:.1f}/100",
                        delta=f"{score - 70:.1f}" if score >= 70 else None
                    )

                # CSV Report
                if res.get("validation_report_csv"):
                    csv_path = Path(res["validation_report_csv"])
                    if csv_path.exists():
                        st.write(f"📊 Validation Report (CSV): `{csv_path.name}`")

                        # Display summary from CSV
                        try:
                            import polars as pl

                            summary_df = pl.read_csv(str(csv_path).replace(".csv", "_summary.csv"))
                            st.dataframe(summary_df, use_container_width=True)
                        except:
                            pass

                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "⬇️ Download Validation CSV",
                                data=csv_path.read_bytes(),
                                file_name=csv_path.name,
                                mime="text/csv"
                            )

                # HTML Report with Visualizations
                if res.get("validation_report_html"):
                    html_path = Path(res["validation_report_html"])
                    if html_path.exists():
                        st.write(f"📈 Validation Report (Interactive HTML): `{html_path.name}`")

                        with col2:
                            st.download_button(
                                "⬇️ Download Validation HTML",
                                data=html_path.read_bytes(),
                                file_name=html_path.name,
                                mime="text/html"
                            )

                        # Option to display inline
                        if st.checkbox(f"Show HTML Report Inline ({html_path.name})", key=f"show_html_{i}"):
                            st.components.v1.html(
                                html_path.read_text(encoding='utf-8'),
                                height=800,
                                scrolling=True
                            )

                # Validation logs (if any warnings/errors)
                if res.get("validation_stdout") or res.get("validation_stderr"):
                    with st.expander("⚙️ Validation Logs", expanded=False):
                        if res.get("validation_stdout"):
                            st.code(res["validation_stdout"], language="text")
                        if res.get("validation_stderr"):
                            st.error(res["validation_stderr"])

            else:
                st.error(f"✗ Failed (elapsed {res['elapsed_s']}s)")
                if res["stderr"]:
                    st.code(res["stderr"][-2000:], language="bash")

            st.caption("Pipeline stdout excerpt:")
            st.code(res["stdout"][-2000:], language="bash")

        progress.progress(i / len(queries))

    # Package all results for download
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in RESULTS_DIR.glob("*"):
            if p.is_file():
                zf.write(p, arcname=p.name)

    st.success("🎉 All queries completed!")
    st.download_button(
        "📦 Download All Results (ZIP)",
        data=buf.getvalue(),
        file_name="results_webui.zip",
        mime="application/zip"
    )
else:
    st.info("Fill in queries above and click **Start Processing**.")
