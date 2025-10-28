# mvi_providers/__init__.py
"""
MVI (Missing Value Imputation) Providers

专注于填补缺失值，不做实体解析(ER)
ER已由 entity_tools.py 和 enrichment_providers.py 通过LLM完成

架构：
- base.py: 基类定义
- wikipedia.py: Wikipedia API填补
- kaggle.py: Kaggle元数据填补
- tmdb.py: TMDb电影数据填补
- omdb.py: OMDb电影数据填补

使用示例：
    provider = get_mvi_provider_for_domain("movie")
    enriched = await provider.impute_missing(rows, primary_keys, allowed_cols)
"""

from .base import MVIProvider, CompositeMVIProvider
from .wikipedia import WikipediaMVIProvider
from .kaggle import KaggleMVIProvider
from .tmdb import TMDbMVIProvider
from .omdb import OMDbMVIProvider

__all__ = [
    "MVIProvider",
    "CompositeMVIProvider",
    "WikipediaMVIProvider",
    "KaggleMVIProvider",
    "TMDbMVIProvider",
    "OMDbMVIProvider",
    "get_mvi_provider_for_domain",
    "detect_domain_from_columns",
]


def detect_domain_from_columns(columns: list[str], query: str = "") -> str:
    """
    根据列名和查询自动检测领域

    Args:
        columns: 数据列名
        query: 用户查询（可选）

    Returns:
        领域名称: "movie", "stock", "ecommerce", "generic"
    """
    cols_lower = " ".join([c.lower() for c in columns])
    query_lower = query.lower() if query else ""
    combined = f"{cols_lower} {query_lower}"

    # 电影/电视领域
    movie_keywords = [
        "movie", "film", "director", "actor", "imdb", "runtime",
        "genre", "box_office", "cinema", "tmdb", "release"
    ]
    if any(kw in combined for kw in movie_keywords):
        return "movie"

    # 股票/金融领域
    stock_keywords = [
        "stock", "ticker", "equity", "ohlc", "close", "open",
        "volume", "exchange", "nasdaq", "nyse", "price"
    ]
    if any(kw in combined for kw in stock_keywords):
        return "stock"

    # 电商领域
    ecommerce_keywords = [
        "product", "sku", "order", "cart", "quantity", "price",
        "customer", "ecommerce", "retail", "basket"
    ]
    if any(kw in combined for kw in ecommerce_keywords):
        return "ecommerce"

    # 学术论文领域
    paper_keywords = [
        "paper", "author", "conference", "journal", "citation",
        "abstract", "publication", "doi"
    ]
    if any(kw in combined for kw in paper_keywords):
        return "paper"

    # 通用领域
    return "generic"


def get_mvi_provider_for_domain(
        domain: str,
        api_keys: dict = None
) -> MVIProvider:
    """
    根据领域返回合适的MVI Provider（或组合）

    Args:
        domain: 领域名称 ("movie", "stock", "ecommerce", "generic")
        api_keys: API密钥字典，如 {"tmdb": "xxx", "omdb": "yyy"}

    Returns:
        MVI Provider实例

    策略：
    - movie: TMDb + OMDb + Wikipedia（组合）
    - stock: Wikipedia + Kaggle
    - ecommerce: Wikipedia + Kaggle
    - generic: Wikipedia（通用）
    """
    api_keys = api_keys or {}

    if domain == "movie" or domain == "film":
        # 电影领域：优先使用专业API
        providers = []

        # TMDb（需要API key）
        if api_keys.get("tmdb"):
            providers.append(TMDbMVIProvider(api_key=api_keys["tmdb"]))

        # OMDb（需要API key）
        if api_keys.get("omdb"):
            providers.append(OMDbMVIProvider(api_key=api_keys["omdb"]))

        # Wikipedia（无需key，兜底）
        providers.append(WikipediaMVIProvider())

        # 如果有Kaggle
        if api_keys.get("kaggle_username") and api_keys.get("kaggle_key"):
            import os
            os.environ["KAGGLE_USERNAME"] = api_keys["kaggle_username"]
            os.environ["KAGGLE_KEY"] = api_keys["kaggle_key"]
            providers.append(KaggleMVIProvider())

        if len(providers) > 1:
            return CompositeMVIProvider(providers)
        elif len(providers) == 1:
            return providers[0]
        else:
            # 如果没有任何provider可用，返回Wikipedia
            return WikipediaMVIProvider()

    elif domain == "stock" or domain == "equity":
        # 股票领域：Wikipedia + Kaggle
        providers = [WikipediaMVIProvider()]

        if api_keys.get("kaggle_username") and api_keys.get("kaggle_key"):
            import os
            os.environ["KAGGLE_USERNAME"] = api_keys["kaggle_username"]
            os.environ["KAGGLE_KEY"] = api_keys["kaggle_key"]
            providers.append(KaggleMVIProvider())

        return CompositeMVIProvider(providers) if len(providers) > 1 else providers[0]

    elif domain == "ecommerce" or domain == "retail":
        # 电商领域：Wikipedia + Kaggle
        providers = [WikipediaMVIProvider()]

        if api_keys.get("kaggle_username") and api_keys.get("kaggle_key"):
            import os
            os.environ["KAGGLE_USERNAME"] = api_keys["kaggle_username"]
            os.environ["KAGGLE_KEY"] = api_keys["kaggle_key"]
            providers.append(KaggleMVIProvider())

        return CompositeMVIProvider(providers) if len(providers) > 1 else providers[0]

    else:
        # 通用领域：只用Wikipedia
        return WikipediaMVIProvider()


# 便捷函数：一站式MVI
async def impute_missing_values(
        rows: list[dict],
        primary_keys: list[str],
        allowed_cols: list[str],
        domain: str = "generic",
        api_keys: dict = None
) -> list[dict]:
    """
    便捷函数：一站式填补缺失值

    Args:
        rows: 数据行
        primary_keys: 主键列
        allowed_cols: 允许更新的列
        domain: 领域名称（auto自动检测）
        api_keys: API密钥

    Returns:
        填补后的数据行

    使用示例：
        enriched = await impute_missing_values(
            rows=data,
            primary_keys=["movie_title"],
            allowed_cols=["year", "director", "genre"],
            domain="movie",
            api_keys={"tmdb": "xxx", "omdb": "yyy"}
        )
    """
    if not rows:
        return rows

    # 自动检测领域
    if domain == "auto":
        columns = list(rows[0].keys()) if rows else []
        domain = detect_domain_from_columns(columns)
        print(f"[MVI] Auto-detected domain: {domain}")

    # 获取provider
    provider = get_mvi_provider_for_domain(domain, api_keys)

    try:
        # 执行填补
        enriched = await provider.impute_missing(rows, primary_keys, allowed_cols)
        return enriched
    finally:
        # 清理
        await provider.aclose()