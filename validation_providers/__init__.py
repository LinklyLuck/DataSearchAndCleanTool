# validation_providers/__init__.py
"""
Data Validation Providers

专注于数据验证，使用外部权威API验证数据准确性

使用示例：
    from validation_providers import get_validation_provider, validate_data

    # 获取验证器
    validator = get_validation_provider("wikipedia")

    # 验证数据
    validated_df, report = await validator.validate(df, primary_keys=["id"])

    # 或使用便捷函数
    validated_df, report = await validate_data(
        df=data,
        primary_keys=["id"],
        source="wikipedia"
    )
"""

from .base import ValidationProvider, CompositeValidationProvider, ValidationReport
from .wikipedia_validator import WikipediaValidationProvider

__all__ = [
    "ValidationProvider",
    "CompositeValidationProvider",
    "ValidationReport",
    "WikipediaValidationProvider",
    "get_validation_provider",
    "validate_data",
]


def get_validation_provider(
        source: str = "wikipedia",
        api_keys: dict = None,
        llm_client=None,
) -> ValidationProvider:
    """
    获取数据验证Provider

    Args:
        source: 验证数据源
            - "wikipedia": Wikipedia验证（无需API key）
            - "tmdb": TMDb验证（电影数据，需要API key）
            - "omdb": OMDb验证（电影数据，需要API key）
            - "comprehensive": 综合验证（多个源）
        api_keys: API密钥字典，如 {"tmdb": "xxx", "omdb": "yyy"}

    Returns:
        ValidationProvider实例
    """
    api_keys = api_keys or {}

    if source == "wikipedia":
        # Wikipedia验证（最常用，无需API key）
        return WikipediaValidationProvider(llm_client=llm_client)

    elif source == "tmdb":
        # TMDb验证（电影数据）
        from .tmdb_validator import TMDbValidationProvider

        tmdb_key = api_keys.get("tmdb")
        if not tmdb_key:
            print("[WARN] TMDb validation requires API key, falling back to Wikipedia")
            return WikipediaValidationProvider()

        return TMDbValidationProvider(api_key=tmdb_key)

    elif source == "omdb":
        # OMDb验证（电影数据）
        from .omdb_validator import OMDbValidationProvider

        omdb_key = api_keys.get("omdb")
        if not omdb_key:
            print("[WARN] OMDb validation requires API key, falling back to Wikipedia")
            return WikipediaValidationProvider()

        return OMDbValidationProvider(api_key=omdb_key)

    elif source == "comprehensive":
        # 综合验证（多个源）
        providers = []

        # 始终包含Wikipedia（兜底）
        providers.append(WikipediaValidationProvider(llm_client=llm_client))

        # 如果有TMDb key，加入TMDb
        if api_keys.get("tmdb"):
            try:
                from .tmdb_validator import TMDbValidationProvider
                providers.append(TMDbValidationProvider(api_key=api_keys["tmdb"]))
            except ImportError:
                pass

        # 如果有OMDb key，加入OMDb
        if api_keys.get("omdb"):
            try:
                from .omdb_validator import OMDbValidationProvider
                providers.append(OMDbValidationProvider(api_key=api_keys["omdb"]))
            except ImportError:
                pass

        if len(providers) > 1:
            return CompositeValidationProvider(providers)
        else:
            return providers[0]

    else:
        raise ValueError(f"Unknown validation source: {source}")


async def validate_data(
        df,  # pl.DataFrame
        primary_keys: list,
        source: str = "wikipedia",
        api_keys: dict = None,
        validate_columns: list = None,
        llm_client=None
):
    """
    便捷函数：一站式数据验证

    Args:
        df: 输入数据（Polars DataFrame）
        primary_keys: 主键列（用于查询外部API）
        source: 验证数据源（"wikipedia", "tmdb", "omdb", "comprehensive"）
        api_keys: API密钥字典（可选）
        validate_columns: 要验证的列（None表示所有列）

    Returns:
        (validated_df, report): 验证后的数据和报告

    使用示例：
        # 使用Wikipedia验证
        validated_df, report = await validate_data(
            df=data,
            primary_keys=["title"],
            source="wikipedia"
        )

        print(f"Corrected {report.corrected_cells} cells")
        print(f"Accuracy improved from {report.accuracy_before:.2%} to {report.accuracy_after:.2%}")

        # 使用多个源综合验证
        validated_df, report = await validate_data(
            df=data,
            primary_keys=["title"],
            source="comprehensive",
            api_keys={"tmdb": "xxx", "omdb": "yyy"}
        )
    """
    # 获取验证器
    validator = get_validation_provider(source, api_keys, llm_client=llm_client)

    try:
        # 执行验证
        validated_df, report = await validator.validate(df, primary_keys, validate_columns)

        return validated_df, report
    finally:
        # 关闭HTTP客户端
        await validator.aclose()