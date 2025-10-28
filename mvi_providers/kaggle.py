# mvi_providers/kaggle.py
"""
Kaggle MVI Provider

功能：使用Kaggle Datasets API填补缺失值
- 查询相关数据集
- 提取元数据信息

需要环境变量：
- KAGGLE_USERNAME
- KAGGLE_KEY

不做实体解析（ER由LLM负责）
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
from .base import MVIProvider


class KaggleMVIProvider(MVIProvider):
    """
    使用Kaggle填补缺失值

    支持的字段：
    - kaggle_dataset: 相关数据集名称
    - kaggle_description: 数据集描述
    - tags: 相关标签

    限制：
    - 需要安装 kaggle 包
    - 需要配置 API 密钥
    """

    name = "Kaggle"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kaggle_available = self._check_kaggle_availability()

        if not self._kaggle_available:
            print("[WARN] Kaggle API not available (missing credentials or package)")

    def _check_kaggle_availability(self) -> bool:
        """检查Kaggle API是否可用"""
        # 检查环境变量
        if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
            return False

        # 检查kaggle包
        try:
            import kaggle
            return True
        except ImportError:
            return False

    async def impute_missing(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            allowed_update_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """填补缺失值"""
        if not self._kaggle_available:
            # Kaggle不可用，直接返回原数据
            return rows

        if not rows:
            return rows

        enriched_rows = []

        for row in rows:
            # 提取实体名称
            entity_name = self._extract_entity_name(row, primary_keys)
            if not entity_name:
                enriched_rows.append(row)
                continue

            # 检查是否有需要填补的列
            needs_imputation = any(
                col in allowed_update_cols and self._is_missing(row.get(col))
                for col in allowed_update_cols
            )

            if not needs_imputation:
                enriched_rows.append(row)
                continue

            # 查询Kaggle
            kaggle_data = await self._query_kaggle(entity_name)

            if not kaggle_data:
                enriched_rows.append(row)
                continue

            # 合并数据
            enriched = self._merge_with_original(
                row,
                kaggle_data,
                allowed_update_cols,
                primary_keys
            )
            enriched_rows.append(enriched)

        return enriched_rows

    async def _query_kaggle(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        查询Kaggle数据集

        Returns:
            包含Kaggle元数据的字典
        """
        # 缓存检查
        cache_key = f"kaggle:{entity_name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Kaggle API是同步的，需要在线程池中运行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._search_datasets, entity_name)

            if result:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            return None

    def _search_datasets(self, query: str) -> Optional[Dict[str, Any]]:
        """
        搜索Kaggle数据集（同步函数）

        Returns:
            提取的元数据
        """
        try:
            import kaggle

            # 认证
            kaggle.api.authenticate()

            # 搜索数据集（最多3个）
            datasets = kaggle.api.datasets_list(search=query, page_size=3)

            if not datasets:
                return None

            # 提取前3个数据集的信息
            result = {
                "kaggle_datasets": [],
                "kaggle_tags": set(),
            }

            for ds in datasets[:3]:
                dataset_info = {
                    "ref": ds.ref,
                    "title": ds.title,
                    "size": getattr(ds, 'size', None),
                    "last_updated": str(getattr(ds, 'lastUpdated', '')),
                }
                result["kaggle_datasets"].append(dataset_info)

                # 收集标签
                if hasattr(ds, 'tags') and ds.tags:
                    result["kaggle_tags"].update(ds.tags)

            # 转换为更友好的格式
            if result["kaggle_datasets"]:
                # 使用第一个数据集作为主要信息源
                first_ds = result["kaggle_datasets"][0]
                result["kaggle_dataset"] = first_ds["title"]
                result["kaggle_ref"] = first_ds["ref"]

            # 标签转为字符串
            if result["kaggle_tags"]:
                result["tags"] = ", ".join(sorted(result["kaggle_tags"]))

            return result

        except Exception as e:
            return None