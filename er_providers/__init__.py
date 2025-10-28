# er_providers/__init__.py
"""
ER Providers - Entity Resolution

实体解析模块，使用LLM识别并合并重复实体
"""

from .enrichment_providers import (
    get_provider_for_auto_domain,
    detect_domain_from_query_or_columns,
    ERProvider,
    LLMERProvider,
)

__all__ = [
    "get_provider_for_auto_domain",
    "detect_domain_from_query_or_columns",
    "ERProvider",
    "LLMERProvider",
]