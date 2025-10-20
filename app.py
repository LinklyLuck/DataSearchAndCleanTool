# app.py -- Streamlit Web UI for your dataset-merge pipeline (+ ER/MVI + Validation)
import os, sys, json, time, zipfile, io, csv, subprocess
from pathlib import Path
import streamlit as st
from subprocess import CalledProcessError

ROOT = Path(__file__).parent.resolve()
PIPELINE = ROOT / "the_pipeline_v2.py"        # 你的现有脚本
VALIDATOR = ROOT / "validate_output_dataset.py"
DEFAULT_DATAS_GLOB = "datalake/*.{csv,json,ndjson,jsonl,xlsx,xls}"
RESULTS_DIR = ROOT / "results_webui"
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="DataSearchTool Web UI", layout="wide")
st.title("🧠 DataSearchTool – 自然语言找数 · 自动合并 · ER/MVI · 校验")

# ---------------------- Sidebar: 基础配置 ----------------------
st.sidebar.header("⚙️ 基础配置")
api_base = st.sidebar.text_input("LLM API Base URL", value="https://goapi.gptnb.ai/v1/chat/completions")
api_key  = st.sidebar.text_input("LLM API Key（优先用环境变量 GPTNB_API_KEY）", type="password", value=os.getenv("GPTNB_API_KEY", ""))
model    = st.sidebar.text_input("模型名（left/map/join 共用）", value="gpt-4o-mini")
max_conc = st.sidebar.number_input("max_concurrency", 1, 16, 4)
graph_out = st.sidebar.checkbox("导出 join graph（.dot）（若脚本支持）", value=False)

# ---------------------- Sidebar: 外部 API 增强 ----------------------
st.sidebar.header("🔐 外部 API 增强（可选）")
st.sidebar.caption("这些密钥将注入到子进程环境变量，ER/MVI 会自动按领域/数据选择合适的外部源。")
k_user = st.sidebar.text_input("Kaggle Username（KAGGLE_USERNAME）", value=os.getenv("KAGGLE_USERNAME", ""))
k_key  = st.sidebar.text_input("Kaggle API Key（KAGGLE_KEY）", type="password", value=os.getenv("KAGGLE_KEY", ""))
omdb_key = st.sidebar.text_input("OMDb API Key（OMDB_API_KEY）", type="password", value=os.getenv("OMDB_API_KEY", ""))
tmdb_key = st.sidebar.text_input("TMDb API Key（TMDB_API_KEY）", type="password", value=os.getenv("TMDB_API_KEY", ""))

st.sidebar.divider()
st.sidebar.header("🧩 ER / MVI 选项")
enable_er_mvi = st.sidebar.checkbox("启用 ER/MVI（自动域识别 + Wikipedia/Kaggle/…）", value=True)
er_sample_rows = st.sidebar.slider("ER/MVI 抽样行数（避免外部请求过大）", min_value=50, max_value=1000, value=200, step=50)
st.sidebar.caption("👉 未配置密钥时会仅使用无需密钥的源（例如 Wikipedia），或跳过该源。")

# ---------------------- Queries 输入 ----------------------
st.subheader("📝 自然语言问题（每行一条）")
text_queries = st.text_area(
    "示例：\n"
    "I need a movie dataset with title, year, director, genre and budget\n"
    "Get a dataset for predicting movie revenue using numeric features\n"
    "Find stock OHLC tables for AAPL with dates and volumes",
    height=150
)

st.write("或上传 `queries_sig.csv`（需包含列名 **NL Questions**）：")
csv_file = st.file_uploader("上传 queries CSV", type=["csv"])

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

# ---------------------- 上传数据（可选） ----------------------
st.subheader("📂 上传数据（可选）")
st.caption("可选；也可以直接用磁盘上的 `datalake/*.{csv,json,ndjson,jsonl,xlsx,xls}`。上传文件将保存到 `datas/` 参与抓取。")
uploads = st.file_uploader(
    "拖入多个 CSV/JSON/NDJSON/JSONL/Excel",
    type=["csv", "json", "ndjson", "jsonl", "xlsx", "xls"],
    accept_multiple_files=True
)
if uploads:
    data_dir = ROOT / "datas"
    data_dir.mkdir(exist_ok=True)
    for f in uploads:
        (data_dir / f.name).write_bytes(f.read())
    st.success(f"已保存到 {data_dir}（共 {len(uploads)} 个文件）")

# ---------------------- 运行按钮 ----------------------
run = st.button("🚀 开始运行", type="primary", disabled=(len(queries) == 0))
log_container = st.container()

def make_slug(idx: int, question: str) -> str:
    base = ''.join(ch if ch.isalnum() else '_' for ch in question.lower())[:40] or 'q'
    return f"q{idx}_{base}"

def run_one_query(idx: int, question: str) -> dict:
    """
    为单条 query 生成临时配置，调用 the_pipeline_v2.py，随后调用 validate_output_dataset.py。
    """
    slug = make_slug(idx, question)
    cfg_path = RESULTS_DIR / f"{slug}.json"
    out_csv  = RESULTS_DIR / f"{slug}_merged.csv"
    dot_path = RESULTS_DIR / f"{slug}_graph.dot" if graph_out else ""
    meta_path = out_csv.with_suffix(".meta.json")

    # 组装默认抓取：datas + datalake
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
        "er_mvi_sample_rows": int(er_sample_rows),
        "enable_er_mvi": bool(enable_er_mvi),
    }
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 构建子进程环境：LLM + 外部增强
    env_for_child = os.environ.copy()
    if api_key:
        env_for_child["GPTNB_API_KEY"] = api_key
    if k_user: env_for_child["KAGGLE_USERNAME"] = k_user
    if k_key:  env_for_child["KAGGLE_KEY"] = k_key
    if omdb_key: env_for_child["OMDB_API_KEY"] = omdb_key
    if tmdb_key: env_for_child["TMDB_API_KEY"] = tmdb_key
    env_for_child["ER_MVI_ENABLED"] = "1" if enable_er_mvi else "0"
    env_for_child["ER_MVI_SAMPLE_ROWS"] = str(er_sample_rows)

    # 运行主管道
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
        "validation_report": "",
        "validation_stdout": "",
        "validation_stderr": "",
    }

    # 从 meta 读取 entity_report（若存在）
    if meta_exists:
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            erp = meta.get("entity_report", "")
            if erp:
                erp_path = Path(erp)
                # 若是相对路径，则用 out_csv 的同目录解析；若 out_csv 不存在则跳过
                if not erp_path.is_absolute() and out_csv_exists:
                    erp_path = out_csv.with_name(out_csv.stem + "_entity_report.csv")
                if erp_path.exists():
                    summary["entity_report"] = str(erp_path)
        except Exception:
            pass

    # 兜底：仅当 out_csv 存在时，才根据 out_csv 推断 entity_report 路径
    if not summary["entity_report"] and out_csv_exists:
        guess_rep = out_csv.with_name(out_csv.stem + "_entity_report.csv")
        if guess_rep.exists():
            summary["entity_report"] = str(guess_rep)

    # —— 生成 validation_report.csv （独立脚本调用，保持解耦）——
    report_path = RESULTS_DIR / f"{slug}_validation_report.csv"
    if out_csv_exists and VALIDATOR.exists():
        try:
            vr = subprocess.run(
                [
                    sys.executable, str(VALIDATOR),
                    "--csv", str(out_csv),
                    "--templates", str(ROOT / "templates"),
                    "--out", str(report_path),
                ],
                capture_output=True,
                text=True,
                env=env_for_child,
                cwd=str(ROOT),
            )
            if vr.returncode == 0 and report_path.exists():
                summary["validation_report"] = str(report_path)
                summary["validation_stdout"] = vr.stdout[-2000:]
            else:
                summary["validation_stdout"] = vr.stdout[-2000:]
                summary["validation_stderr"] = vr.stderr[-2000:]
        except CalledProcessError as e:
            summary["validation_stderr"] = f"CalledProcessError: {e}"
    else:
        if not VALIDATOR.exists():
            summary["validation_stderr"] = "validate_output_dataset.py 不存在，跳过验证。"

    return summary



if run:
    st.toast(f"开始运行（共 {len(queries)} 条）", icon="✅")
    results = []
    progress = st.progress(0.0)

    for i, q in enumerate(queries, 1):
        with log_container.expander(f"[{i}/{len(queries)}] {q}", expanded=True):
            res = run_one_query(i, q)
            results.append(res)

            if res["returncode"] == 0:
                st.success(f"完成，耗时 {res['elapsed_s']} s")

                # 合并输出
                if res["out_csv"]:
                    st.write(f"合并输出：`{res['out_csv']}`")
                    st.download_button("下载合并 CSV", data=Path(res["out_csv"]).read_bytes(),
                                       file_name=Path(res["out_csv"]).name, mime="text/csv")

                # Meta
                if res["meta_json"]:
                    try:
                        meta = json.loads(Path(res["meta_json"]).read_text("utf-8"))
                        with st.expander("Meta（matched_files / diagnostics / enrichment）", expanded=False):
                            st.json(meta)
                    except Exception as e:
                        st.caption(f"读取 meta 失败：{e}")

                # Join Graph
                if res["graph_dot"]:
                    st.write(f"Join Graph：`{res['graph_dot']}`（.dot）")
                    st.download_button("下载 DOT", data=Path(res["graph_dot"]).read_bytes(),
                                       file_name=Path(res["graph_dot"]).name, mime="text/vnd.graphviz")

                # ER/MVI 报告
                if res.get("entity_report"):
                    rp = Path(res["entity_report"])
                    if rp.exists():
                        st.write(f"ER/MVI 样本报告：`{rp}`")
                        st.download_button("下载 ER/MVI 报告（CSV）",
                                           data=rp.read_bytes(),
                                           file_name=rp.name, mime="text/csv")

                # 验证报告
                if res.get("validation_report"):
                    rp = Path(res["validation_report"])
                    st.write(f"验证报告：`{rp}`")
                    st.download_button("下载 validation_report.csv",
                                       data=rp.read_bytes(),
                                       file_name=rp.name, mime="text/csv")
                elif res.get("validation_stdout") or res.get("validation_stderr"):
                    st.caption("validation output:")
                    if res.get("validation_stdout"):
                        st.code(res["validation_stdout"], language="bash")
                    if res.get("validation_stderr"):
                        st.code(res["validation_stderr"], language="bash")
            else:
                st.error(f"失败（耗时 {res['elapsed_s']} s）")
                if res["stderr"]:
                    st.code(res["stderr"][-2000:], language="bash")

            st.caption("stdout 摘要：")
            st.code(res["stdout"][-2000:], language="bash")
        progress.progress(i / len(queries))

    # 打包所有结果为 zip 提供下载
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in RESULTS_DIR.glob("*"):
            if p.is_file():
                zf.write(p, arcname=p.name)
    st.success("全部完成 🎉")
    st.download_button("打包下载所有结果（ZIP）", data=buf.getvalue(), file_name="results_webui.zip", mime="application/zip")
else:
    st.info("填好问题后点击上面的 **开始运行**。")
