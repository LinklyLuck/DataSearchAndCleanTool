# mvi_providers/wikipedia.py
"""
Wikipedia MVI Provider

功能：使用Wikipedia API填补缺失值
- 搜索实体名称
- 获取summary/description
- 提取相关信息（年份、类别等）

不做实体解析（ER由LLM负责）
"""
import re
from typing import List, Dict, Any, Optional
from .base import MVIProvider


class WikipediaMVIProvider(MVIProvider):
    """
    使用Wikipedia填补缺失值

    支持的字段：
    - description/summary: 从Wikipedia摘要提取
    - year/release_year: 从摘要中提取年份
    - category/type: 从description推断
    """

    name = "Wikipedia"

    async def impute_missing(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            allowed_update_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """填补缺失值"""
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

            # 查询Wikipedia
            wiki_data = await self._query_wikipedia(entity_name)

            if not wiki_data:
                enriched_rows.append(row)
                continue

            # 合并数据
            enriched = self._merge_with_original(
                row,
                wiki_data,
                allowed_update_cols,
                primary_keys
            )
            enriched_rows.append(enriched)

        return enriched_rows

    async def _query_wikipedia(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        查询Wikipedia并提取有用信息

        Returns:
            包含可用字段的字典，如：
            {
                "description": "...",
                "summary": "...",
                "wiki_title": "...",
                "year": 2020,
                "category": "..."
            }
        """
        # 缓存检查
        cache_key = f"wiki:{entity_name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Step 1: 搜索Wikipedia标题
            title = await self._wiki_search(entity_name)
            if not title:
                return None

            # Step 2: 获取页面摘要
            summary_data = await self._wiki_summary(title)
            if not summary_data:
                return None

            # Step 3: 提取结构化信息
            result = self._extract_fields(summary_data)

            # 缓存结果
            self._cache[cache_key] = result
            return result

        except Exception as e:
            return None

    async def _wiki_search(self, query: str) -> Optional[str]:
        """
        搜索Wikipedia，返回最佳匹配的标题

        API: https://en.wikipedia.org/w/api.php?action=opensearch
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "namespace": 0,
            "format": "json"
        }

        response = await self._make_request("GET", url, params=params)
        if not response:
            return None

        try:
            data = response.json()
            # opensearch返回格式: [query, [titles], [descriptions], [urls]]
            if isinstance(data, list) and len(data) >= 2 and data[1]:
                return data[1][0]  # 第一个结果的标题
        except Exception:
            pass

        return None

    async def _wiki_summary(self, title: str) -> Optional[Dict[str, Any]]:
        """
        获取Wikipedia页面摘要

        API: https://en.wikipedia.org/api/rest_v1/page/summary/{title}
        """
        # URL编码标题（空格→下划线）
        safe_title = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

        response = await self._make_request("GET", url)
        if not response:
            return None

        try:
            return response.json()
        except Exception:
            return None

    def _extract_fields(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从Wikipedia摘要提取结构化字段

        Args:
            summary_data: Wikipedia REST API返回的摘要数据

        Returns:
            提取的字段字典
        """
        result = {}

        # 基础字段
        if "title" in summary_data:
            result["wiki_title"] = summary_data["title"]

        if "description" in summary_data:
            result["description"] = summary_data["description"]

        if "extract" in summary_data:
            result["summary"] = summary_data["extract"]
            result["wiki_summary"] = summary_data["extract"]

        # 提取年份（从摘要文本中）
        text = summary_data.get("extract", "")
        year = self._extract_year(text)
        if year:
            result["year"] = year
            result["release_year"] = year

        # 提取类别/类型（从description）
        description = summary_data.get("description", "")
        category = self._extract_category(description)
        if category:
            result["category"] = category
            result["type"] = category

        return result

    def _extract_year(self, text: str) -> Optional[int]:
        """
        从文本中提取年份

        匹配模式：
        - "in 2020"
        - "(2020)"
        - "2020年"
        - "released 2020"
        """
        if not text:
            return None

        # 匹配4位数年份（1900-2099）
        patterns = [
            r'\b(19\d{2}|20\d{2})\b',  # 独立的年份
            r'\((\d{4})\)',  # 括号中的年份
            r'in (\d{4})',  # "in 2020"
            r'released (\d{4})',  # "released 2020"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    year = int(match.group(1))
                    # 合理性检查
                    if 1900 <= year <= 2099:
                        return year
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_category(self, description: str) -> Optional[str]:
        """
        从description提取类别

        例如：
        - "American action film" → "action"
        - "2020 science fiction movie" → "science fiction"
        - "Swedish company" → "company"
        """
        if not description:
            return None

        desc_lower = description.lower()

        # 常见类别模式
        category_patterns = {
            "action": ["action"],
            "comedy": ["comedy"],
            "drama": ["drama"],
            "thriller": ["thriller"],
            "horror": ["horror"],
            "romance": ["romance"],
            "sci-fi": ["science fiction", "sci-fi", "scifi"],
            "documentary": ["documentary"],
            "animation": ["animated", "animation"],
            "company": ["company", "corporation", "firm"],
            "movie": ["film", "movie"],
            "series": ["television series", "tv series", "series"],
        }

        for category, keywords in category_patterns.items():
            if any(kw in desc_lower for kw in keywords):
                return category

        return None