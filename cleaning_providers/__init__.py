# cleaning_providers/__init__.py
"""
Data Cleaning Providers

专注于数据清洗，不做缺失值填补（由MVI负责）和实体解析（由ER负责）

使用示例：
    from cleaning_providers import get_cleaning_provider, clean_data

    # 获取清洗器（推荐使用 ultra 模式，速度更快）
    cleaner = get_cleaning_provider("ultra", llm_client=client)

    # 清洗数据
    cleaned_df, report = await cleaner.clean(df, primary_keys=["id"])
"""

from .base import CleaningProvider, CompositeCleaningProvider, CleaningReport
from .llm_cleaner import LLMCleaningProvider
from .type_cleaner import TypeCleaningProvider
from .fast_cleaner import FastCleaningProvider
from .batch_llm_cleaner import BatchLLMCleaningProvider
from .hybrid_cleaner import HybridCleaningProvider
from .ultra_cleaner import UltraCleaningProvider

__all__ = [
    "CleaningProvider",
    "CompositeCleaningProvider",
    "CleaningReport",
    "LLMCleaningProvider",
    "TypeCleaningProvider",
    "FastCleaningProvider",
    "BatchLLMCleaningProvider",
    "HybridCleaningProvider",
    "UltraCleaningProvider",
    "get_cleaning_provider",
    "clean_data",
]


def get_cleaning_provider(
        mode: str = "ultra",  # 默认 ultra（更快），若无 llm_client 会降级 fast
        llm_client=None
) -> CleaningProvider:
    """
    获取数据清洗Provider

    Args:
        mode: 清洗模式
            - "ultra": 终极加速清洗（规则→TopK唯一值微批LLM→缓存；极快，推荐）
            - "fast": 超快速规则清洗（10-20x，纯规则，适合简单数据）
            - "hybrid": 混合清洗（3-10x，规则+LLM）
            - "batch": 批量LLM清洗（3-5x）
            - "comprehensive": 综合清洗（LLM + Type，较慢但更全面）
            - "llm": 仅LLM（最慢）
            - "type"/"minimal": 仅类型清洗（快速）
        llm_client: LLM客户端（LLM相关模式需要）

    Returns:
        CleaningProvider实例
    """

    # Ultra：终极加速（无 llm_client 时自动降级 fast）
    if mode in ("ultra", "hyper", "turbo"):
        if not llm_client:
            print("[WARN] Ultra mode requires llm_client. Falling back to fast cleaning.")
            return FastCleaningProvider(aggressive=False)
        return UltraCleaningProvider(
            llm_client=llm_client,
            value_batch_size=24,
            max_concurrency=6,
            cache_db=".cache/clean_cache.sqlite",
            use_clustering=False,      # 安装 rapidfuzz 后可切 True
            cluster_threshold=90,
            llm_budget_ratio=0.005,    # 仅Top 0.5%唯一值进LLM
            version_tag="v1",
        )

    # 快速模式：纯规则引擎（最快，10-20倍速度）
    if mode == "fast":
        return FastCleaningProvider(aggressive=False)

    # 混合模式：规则 + 批量LLM
    elif mode == "hybrid":
        if not llm_client:
            print("[WARN] Hybrid mode works best with LLM, falling back to fast cleaning")
            return FastCleaningProvider(aggressive=False)
        return HybridCleaningProvider(
            llm_client=llm_client,
            fast_first=True,
            llm_batch_size=8,
            max_sample_rows=15,
            skip_clean_columns=True
        )

    # 批量LLM模式：纯LLM但批量优化（3-5倍速度）
    elif mode == "batch":
        if not llm_client:
            print("[WARN] Batch mode requires llm_client, falling back to type cleaning")
            return TypeCleaningProvider()
        return BatchLLMCleaningProvider(
            llm_client=llm_client,
            batch_size=8,
            max_sample_rows=20,
            max_concurrent_batches=3
        )

    # 综合模式
    elif mode == "comprehensive":
        providers = []
        if llm_client:
            providers.append(LLMCleaningProvider(llm_client))
        providers.append(TypeCleaningProvider())

        if len(providers) > 1:
            return CompositeCleaningProvider(providers)
        elif len(providers) == 1:
            return providers[0]
        else:
            return TypeCleaningProvider()

    # 仅LLM清洗（最慢）
    elif mode == "llm":
        if not llm_client:
            print("[WARN] LLM mode requires llm_client, falling back to type cleaning")
            return TypeCleaningProvider()
        return LLMCleaningProvider(llm_client)

    # 仅类型清洗
    elif mode in ("type", "minimal"):
        return TypeCleaningProvider()

    else:
        raise ValueError(f"Unknown cleaning mode: {mode}")


async def clean_data(
        df,  # pl.DataFrame
        primary_keys: list,
        llm_client=None,
        mode: str = "ultra",  # 默认 ultra
        rules: dict = None
):
    """
    便捷函数：一站式数据清洗

    Args:
        df: 输入数据（Polars DataFrame）
        primary_keys: 主键列
        llm_client: LLM客户端（可选；ultra/hybrid/batch/comprehensive/llm 需要）
        mode: 清洗模式（默认 "ultra"，速度与质量平衡最佳）
        rules: 清洗规则（可选）

    Returns:
        (cleaned_df, report): 清洗后的数据和报告
    """
    cleaner = get_cleaning_provider(mode, llm_client)
    cleaned_df, report = await cleaner.clean(df, primary_keys, rules)
    return cleaned_df, report
