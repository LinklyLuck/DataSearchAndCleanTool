# er_providers/__init__.py
from .wikipedia_kaggle import WikipediaKaggleProvider
from .wikipedia_kaggle import detect_domain_from_query  # 暴露统一的域识别函数

# 映射若干常见领域到同一个 Provider（此 Provider 已做跨领域处理）
DOMAIN_PROVIDER_MAP = {
    "movie": WikipediaKaggleProvider,
    "film": WikipediaKaggleProvider,
    "cinema": WikipediaKaggleProvider,
    "stock": WikipediaKaggleProvider,
    "equity": WikipediaKaggleProvider,
    "ticker": WikipediaKaggleProvider,
    "ecommerce": WikipediaKaggleProvider,
    "commerce": WikipediaKaggleProvider,
    "retail": WikipediaKaggleProvider,
    "generic": WikipediaKaggleProvider,
}

def get_provider(domain_hint: str = ""):
    s = (domain_hint or "").lower()
    for k, cls in DOMAIN_PROVIDER_MAP.items():
        if k in s:
            return cls()
    return WikipediaKaggleProvider()
