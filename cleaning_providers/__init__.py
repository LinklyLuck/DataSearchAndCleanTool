# cleaning_providers/__init__.py
"""
Data Cleaning Providers

专注于数据清洗，不做缺失值填补（由MVI负责）和实体解析（由ER负责）

使用示例：
    from cleaning_providers import get_cleaning_provider, clean_data

    # 获取清洗器
    cleaner = get_cleaning_provider("comprehensive", llm_client=client)

    # 清洗数据
    cleaned_df, report = await cleaner.clean(df, primary_keys=["id"])

    # 或使用便捷函数
    cleaned_df, report = await clean_data(
        df=data,
        primary_keys=["id"],
        llm_client=client,
        mode="comprehensive"
    )
"""

from .base import CleaningProvider, CompositeCleaningProvider, CleaningReport
from .llm_cleaner import LLMCleaningProvider
from .type_cleaner import TypeCleaningProvider

__all__ = [
    "CleaningProvider",
    "CompositeCleaningProvider",
    "CleaningReport",
    "LLMCleaningProvider",
    "TypeCleaningProvider",
    "get_cleaning_provider",
    "clean_data",
]


def get_cleaning_provider(
        mode: str = "comprehensive",
        llm_client=None
) -> CleaningProvider:
    """
    获取数据清洗Provider

    Args:
        mode: 清洗模式
            - "comprehensive": 综合清洗（LLM + Type）
            - "llm": 仅LLM智能清洗
            - "type": 仅类型清洗
            - "minimal": 最小清洗（仅类型）
        llm_client: LLM客户端（LLM模式需要）

    Returns:
        CleaningProvider实例
    """
    if mode == "comprehensive":
        # 综合清洗：LLM + Type
        providers = []

        if llm_client:
            providers.append(LLMCleaningProvider(llm_client))

        providers.append(TypeCleaningProvider())

        if len(providers) > 1:
            return CompositeCleaningProvider(providers)
        elif len(providers) == 1:
            return providers[0]
        else:
            # 回退到类型清洗
            return TypeCleaningProvider()

    elif mode == "llm":
        # 仅LLM清洗
        if not llm_client:
            print("[WARN] LLM mode requires llm_client, falling back to type cleaning")
            return TypeCleaningProvider()
        return LLMCleaningProvider(llm_client)

    elif mode == "type" or mode == "minimal":
        # 仅类型清洗
        return TypeCleaningProvider()

    else:
        raise ValueError(f"Unknown cleaning mode: {mode}")


async def clean_data(
        df,  # pl.DataFrame
        primary_keys: list,
        llm_client=None,
        mode: str = "comprehensive",
        rules: dict = None
):
    """
    便捷函数：一站式数据清洗

    Args:
        df: 输入数据（Polars DataFrame）
        primary_keys: 主键列
        llm_client: LLM客户端（可选）
        mode: 清洗模式（"comprehensive", "llm", "type", "minimal"）
        rules: 清洗规则（可选）

    Returns:
        (cleaned_df, report): 清洗后的数据和报告

    使用示例：
        cleaned_df, report = await clean_data(
            df=data,
            primary_keys=["id"],
            llm_client=client,
            mode="comprehensive"
        )

        print(f"Cleaned {report.cleaned_cells} cells")
    """
    # 获取清洗器
    cleaner = get_cleaning_provider(mode, llm_client)

    # 执行清洗
    cleaned_df, report = await cleaner.clean(df, primary_keys, rules)

    return cleaned_df, report
