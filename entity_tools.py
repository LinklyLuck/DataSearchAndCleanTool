# entity_tools.py
from __future__ import annotations
import re, asyncio
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
from difflib import SequenceMatcher
from enrichment_providers import MovieEnricher

# --------- 实体归一化工具 ----------

def _norm_title(s: Optional[str]) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # 去标点
    return s

def _norm_name(s: Optional[str]) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def make_canonical_key(title: Optional[str], director: Optional[str], year: Optional[str|int]) -> str:
    return f"{_norm_title(title)}::{_norm_name(director)}::{str(year or '').strip()}"

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()

def entity_resolution_movies(df: pl.DataFrame,
                             title_col="movie_title",
                             director_col="director_name",
                             year_col="release_year",
                             min_title_sim=0.92) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    通过规范 key 聚合 + 近似标题吸附，合并重复实体。
    返回 (resolved_df, report_df)
    """
    if not all(c in df.columns for c in [title_col, director_col, year_col]):
        return df, pl.DataFrame()

    # 初始 canonical key
    df = df.with_columns([
        pl.struct([pl.col(title_col), pl.col(director_col), pl.col(year_col)])
          .map_elements(lambda r: make_canonical_key(r[title_col], r[director_col], r[year_col]))
          .alias("_canon")
    ])

    # 对标题近似（同导演+年份下，标题相似的吸附到最常见的canonical）
    # 1) 统计每个 (director, year) 下 title 的“众数”
    grp = (df
           .group_by([director_col, year_col, title_col])
           .len().rename({"len": "_cnt"}))
    # 找每个 (director, year) 下计数最高的 title 作为锚点
    anchor = (grp
              .group_by([director_col, year_col])
              .agg([pl.col(title_col).filter(pl.col("_cnt") == pl.col("_cnt").max()).first().alias("_title_anchor")]))

    df2 = df.join(anchor, on=[director_col, year_col], how="left")
    # 2) 如果 title 与 anchor 相似度很高，则重写 _canon 的标题部分
    def _maybe_recanon(r):
        t = r[title_col]; a = r["_title_anchor"]
        if not a:
            return r["_canon"]
        if _sim(_norm_title(t), _norm_title(a)) >= min_title_sim:
            return make_canonical_key(a, r[director_col], r[year_col])
        return r["_canon"]

    df2 = df2.with_columns([
        pl.struct([pl.col(title_col), pl.col("_title_anchor"), pl.col(director_col), pl.col(year_col), pl.col("_canon")])
          .map_elements(_maybe_recanon).alias("_canon2")
    ]).drop(["_title_anchor"])
    df2 = df2.drop("_canon").rename({"_canon2": "_canon"})

    # 3) 聚合：按 _canon 合并，优先非空
    def _first_non_null(col: str):
        return pl.when(pl.col(col).is_not_null()).then(pl.col(col)).otherwise(None).first()

    keep_cols = [c for c in df2.columns if c != "_canon"]
    # 简易聚合策略：数值均值、文本首个非空
    num_cols = [c for c, dt in zip(df2.columns, df2.dtypes) if dt.is_numeric() and c in keep_cols]
    txt_cols = [c for c in keep_cols if c not in num_cols]

    agg_exprs = []
    for c in num_cols:
        agg_exprs.append(pl.col(c).mean().alias(c))
    for c in txt_cols:
        agg_exprs.append(pl.col(c).filter(pl.col(c).is_not_null()).first().alias(c))

    resolved = (df2.group_by("_canon").agg(agg_exprs)).drop("_canon", ignore_errors=True)

    # 报告：每个 _canon 聚合了多少条
    report = (df2.group_by("_canon").len().rename({"len": "records_merged"})
              .sort("records_merged", descending=True))
    return resolved, report


# --------- 缺失值填补（电影） ----------
async def impute_missing_movies(df: pl.DataFrame,
                                title_col="movie_title",
                                director_col="director_name",
                                year_col="release_year",
                                numeric_cols: Optional[List[str]] = None,
                                cat_cols: Optional[List[str]] = None,
                                use_external=True) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    先内源填补（组内中位数/众数）→ 可选外源补齐（维基/OMDb/TMDB）。
    返回 (df_imputed, imputation_report)
    """
    if numeric_cols is None:
        numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    if cat_cols is None:
        cat_cols = [c for c in df.columns if c not in numeric_cols]

    # A) 组内（按 title 或 director）填补
    work = df.clone()

    # 数值：按导演分组中位数，仍空再全局中位数
    for c in numeric_cols:
        med_by_dir = (work.group_by(director_col).agg(pl.col(c).median().alias(f"__med_{c}")))
        work = work.join(med_by_dir, on=director_col, how="left")
        work = work.with_columns([
            pl.when(pl.col(c).is_null()).then(pl.col(f"__med_{c}")).otherwise(pl.col(c)).alias(c)
        ]).drop(f"__med_{c}")
        # 全局中位数
        glob_med = work.select(pl.col(c).median()).item()
        work = work.with_columns([
            pl.when(pl.col(c).is_null()).then(pl.lit(glob_med)).otherwise(pl.col(c)).alias(c)
        ])

    # 类别：按导演众数，再全局众数
    def col_mode(series: pl.Series):
        try:
            # 取出现频率最高的非空
            vc = series.drop_nulls().to_list()
            if not vc:
                return None
            from collections import Counter
            return Counter(vc).most_common(1)[0][0]
        except Exception:
            return None

    for c in cat_cols:
        # director 层面的众数
        by_dir = work.group_by(director_col).agg(pl.col(c)).lazy().collect()
        fill_map = {}
        for i in range(by_dir.height):
            d = by_dir[director_col][i]
            fill_map[d] = col_mode(by_dir[c][i])
        work = work.with_columns([
            pl.when(pl.col(c).is_null()).then(
                pl.col(director_col).map_elements(lambda d: fill_map.get(d))
            ).otherwise(pl.col(c)).alias(c)
        ])
        # 全局众数
        gmode = col_mode(work[c])
        work = work.with_columns([
            pl.when(pl.col(c).is_null()).then(pl.lit(gmode)).otherwise(pl.col(c)).alias(c)
        ])

    impute_rows_before_external = work.height

    # B) 外源补齐（可选）
    report_rows = []
    if use_external and all(k in work.columns for k in [title_col, year_col]):
        enricher = MovieEnricher(
            use_wiki=True,
            use_omdb=bool(os.getenv("OMDB_API_KEY", "").strip()),
            use_tmdb=bool(os.getenv("TMDB_API_KEY", "").strip())
        )
        # 仅对仍缺失的行发起补齐
        need_cols = ["genre","runtime_minutes","country_code","imdb_rating","budget_usd","revenue_usd","director_name"]
        missing_mask = None
        for c in need_cols:
            if c in work.columns:
                m = work[c].is_null()
                missing_mask = m if missing_mask is None else (missing_mask | m)

        if missing_mask is not None:
            idxs = [i for i, v in enumerate(missing_mask) if v]
            # 适当限流并发
            sem = asyncio.Semaphore(6)
            async def one(i):
                async with sem:
                    t = work[title_col][i]
                    y = work[year_col][i]
                    info = await enricher.enrich(str(t), str(y) if y is not None else None)
                    return i, info
            tasks = [one(i) for i in idxs[:2000]]  # 防止过大样本时触发配额；可调
            for coro in asyncio.as_completed(tasks):
                i, info = await coro
                if info:
                    # 写回可补的字段
                    updates = {}
                    for k in need_cols:
                        if k in work.columns and (work[k][i] is None) and (info.get(k) is not None):
                            updates[k] = info[k]
                    if updates:
                        # polars 行级更新：使用 with_columns + when/then
                        for k, v in updates.items():
                            work = work.with_columns([
                                pl.when(pl.arange(0, work.height) == i).then(pl.lit(v)).otherwise(pl.col(k)).alias(k)
                            ])
                        report_rows.append({"row_index": i, **updates})
            await enricher.aclose()

    report = pl.DataFrame(report_rows) if report_rows else pl.DataFrame()
    return work, report
