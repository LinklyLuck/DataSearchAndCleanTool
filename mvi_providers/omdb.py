# mvi_providers/omdb.py
"""
OMDb (Open Movie Database) MVI Provider

功能：使用OMDb API填补电影缺失值
- 搜索电影
- 获取详细信息（IMDb评分、导演、演员等）

需要环境变量：
- OMDB_API_KEY

官方文档：http://www.omdbapi.com/

不做实体解析（ER由LLM负责）
"""
import os
from typing import List, Dict, Any, Optional
from .base import MVIProvider


class OMDbMVIProvider(MVIProvider):
    """
    使用OMDb填补电影缺失值

    支持的字段：
    - title: 标题
    - year: 年份
    - director: 导演
    - actors/cast: 演员
    - genre: 类型
    - runtime: 时长
    - imdb_rating: IMDb评分
    - plot/description: 剧情简介
    - rated: 分级
    - box_office: 票房
    """

    name = "OMDb"
    BASE_URL = "http://www.omdbapi.com/"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("OMDB_API_KEY")
        super().__init__(api_key=api_key, **kwargs)

        if not self.api_key:
            print("[WARN] OMDb API key not found (set OMDB_API_KEY)")

    async def impute_missing(
            self,
            rows: List[Dict[str, Any]],
            primary_keys: List[str],
            allowed_update_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """填补缺失值"""
        if not self.api_key:
            return rows

        if not rows:
            return rows

        enriched_rows = []

        for row in rows:
            # 提取电影标题
            entity_name = self._extract_entity_name(row, primary_keys)
            if not entity_name:
                enriched_rows.append(row)
                continue

            # 检查是否需要填补
            needs_imputation = any(
                col in allowed_update_cols and self._is_missing(row.get(col))
                for col in allowed_update_cols
            )

            if not needs_imputation:
                enriched_rows.append(row)
                continue

            # 查询OMDb
            omdb_data = await self._query_omdb(entity_name, row.get("year"))

            if not omdb_data:
                enriched_rows.append(row)
                continue

            # 合并数据
            enriched = self._merge_with_original(
                row,
                omdb_data,
                allowed_update_cols,
                primary_keys
            )
            enriched_rows.append(enriched)

        return enriched_rows

    async def _query_omdb(
            self,
            title: str,
            year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        查询OMDb电影信息

        Args:
            title: 电影标题
            year: 年份（可选）

        Returns:
            电影详细信息
        """
        # 缓存检查
        cache_key = f"omdb:{title.lower()}:{year or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # 查询电影
            params = {
                "apikey": self.api_key,
                "t": title,  # title search
                "type": "movie",
                "plot": "full"  # 完整剧情
            }

            if year:
                params["y"] = year

            response = await self._make_request("GET", self.BASE_URL, params=params)
            if not response:
                return None

            data = response.json()

            # 检查是否成功
            if data.get("Response") != "True":
                return None

            # 提取字段
            result = self._extract_fields(data)

            # 缓存
            self._cache[cache_key] = result
            return result

        except Exception as e:
            return None

    def _extract_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从OMDb响应中提取结构化字段
        """
        result = {}

        # 标题
        if "Title" in data:
            result["title"] = data["Title"]
            result["movie_title"] = data["Title"]

        # 年份
        if "Year" in data:
            try:
                # 处理 "2020–2021" 这样的格式
                year_str = data["Year"].split("–")[0]
                year = int(year_str)
                result["year"] = year
                result["release_year"] = year
            except (ValueError, IndexError):
                pass

        # 评分
        if "Rated" in data and data["Rated"] != "N/A":
            result["rated"] = data["Rated"]

        # 上映日期
        if "Released" in data and data["Released"] != "N/A":
            result["release_date"] = data["Released"]

        # 时长
        if "Runtime" in data and data["Runtime"] != "N/A":
            # "142 min" → 142
            try:
                runtime_str = data["Runtime"].replace(" min", "")
                result["runtime"] = int(runtime_str)
            except ValueError:
                pass

        # 类型
        if "Genre" in data and data["Genre"] != "N/A":
            result["genre"] = data["Genre"]
            result["genres"] = [g.strip() for g in data["Genre"].split(",")]

        # 导演
        if "Director" in data and data["Director"] != "N/A":
            result["director"] = data["Director"]
            result["directors"] = [d.strip() for d in data["Director"].split(",")]

        # 编剧
        if "Writer" in data and data["Writer"] != "N/A":
            result["writer"] = data["Writer"]

        # 演员
        if "Actors" in data and data["Actors"] != "N/A":
            result["actors"] = data["Actors"]
            result["cast"] = data["Actors"]

        # 剧情
        if "Plot" in data and data["Plot"] != "N/A":
            result["plot"] = data["Plot"]
            result["description"] = data["Plot"]
            result["summary"] = data["Plot"]

        # 语言
        if "Language" in data and data["Language"] != "N/A":
            result["language"] = data["Language"]

        # 国家
        if "Country" in data and data["Country"] != "N/A":
            result["country"] = data["Country"]

        # 奖项
        if "Awards" in data and data["Awards"] != "N/A":
            result["awards"] = data["Awards"]

        # IMDb评分
        if "imdbRating" in data and data["imdbRating"] != "N/A":
            try:
                result["imdb_rating"] = float(data["imdbRating"])
            except ValueError:
                pass

        # IMDb ID
        if "imdbID" in data and data["imdbID"] != "N/A":
            result["imdb_id"] = data["imdbID"]

        # 票房
        if "BoxOffice" in data and data["BoxOffice"] != "N/A":
            result["box_office"] = data["BoxOffice"]
            # 尝试转换为数字（"$123,456,789" → 123456789）
            try:
                bo_str = data["BoxOffice"].replace("$", "").replace(",", "")
                result["box_office_value"] = int(bo_str)
            except ValueError:
                pass

        # Metascore
        if "Metascore" in data and data["Metascore"] != "N/A":
            try:
                result["metascore"] = int(data["Metascore"])
            except ValueError:
                pass

        return result