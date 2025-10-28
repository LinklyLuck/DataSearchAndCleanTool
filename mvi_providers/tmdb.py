# mvi_providers/tmdb.py
"""
TMDb (The Movie Database) MVI Provider

功能：使用TMDb API填补电影/电视剧缺失值
- 搜索电影/电视剧
- 获取详细信息（导演、演员、类型、预算等）

需要环境变量：
- TMDB_API_KEY

官方文档：https://developers.themoviedb.org/3

不做实体解析（ER由LLM负责）
"""
import os
from typing import List, Dict, Any, Optional
from .base import MVIProvider


class TMDbMVIProvider(MVIProvider):
    """
    使用TMDb填补电影/电视剧缺失值

    支持的字段：
    - title: 标准化标题
    - year/release_year: 上映年份
    - director: 导演
    - genre: 类型
    - budget: 预算
    - revenue: 票房
    - runtime: 时长
    - imdb_id: IMDb ID
    - overview/description: 简介
    """

    name = "TMDb"
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("TMDB_API_KEY")
        super().__init__(api_key=api_key, **kwargs)

        if not self.api_key:
            print("[WARN] TMDb API key not found (set TMDB_API_KEY)")

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

            # 查询TMDb
            tmdb_data = await self._query_tmdb(entity_name, row.get("year"))

            if not tmdb_data:
                enriched_rows.append(row)
                continue

            # 合并数据
            enriched = self._merge_with_original(
                row,
                tmdb_data,
                allowed_update_cols,
                primary_keys
            )
            enriched_rows.append(enriched)

        return enriched_rows

    async def _query_tmdb(
            self,
            movie_title: str,
            year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        查询TMDb电影信息

        Args:
            movie_title: 电影标题
            year: 年份（可选，用于更精确匹配）

        Returns:
            电影详细信息
        """
        # 缓存检查
        cache_key = f"tmdb:{movie_title.lower()}:{year or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Step 1: 搜索电影
            movie_id = await self._search_movie(movie_title, year)
            if not movie_id:
                return None

            # Step 2: 获取详细信息
            details = await self._get_movie_details(movie_id)
            if not details:
                return None

            # Step 3: 获取演职人员
            credits = await self._get_movie_credits(movie_id)

            # Step 4: 合并信息
            result = self._extract_fields(details, credits)

            # 缓存
            self._cache[cache_key] = result
            return result

        except Exception as e:
            return None

    async def _search_movie(
            self,
            title: str,
            year: Optional[int] = None
    ) -> Optional[int]:
        """
        搜索电影，返回最佳匹配的movie_id

        API: /search/movie
        """
        url = f"{self.BASE_URL}/search/movie"
        params = {
            "api_key": self.api_key,
            "query": title,
            "language": "en-US"
        }

        if year:
            params["year"] = year

        response = await self._make_request("GET", url, params=params)
        if not response:
            return None

        try:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return None

            # 返回第一个结果的ID
            return results[0]["id"]

        except Exception:
            return None

    async def _get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        获取电影详细信息

        API: /movie/{movie_id}
        """
        url = f"{self.BASE_URL}/movie/{movie_id}"
        params = {
            "api_key": self.api_key,
            "language": "en-US"
        }

        response = await self._make_request("GET", url, params=params)
        if not response:
            return None

        try:
            return response.json()
        except Exception:
            return None

    async def _get_movie_credits(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        获取演职人员信息

        API: /movie/{movie_id}/credits
        """
        url = f"{self.BASE_URL}/movie/{movie_id}/credits"
        params = {
            "api_key": self.api_key
        }

        response = await self._make_request("GET", url, params=params)
        if not response:
            return None

        try:
            return response.json()
        except Exception:
            return None

    def _extract_fields(
            self,
            details: Dict[str, Any],
            credits: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从TMDb响应中提取结构化字段
        """
        result = {}

        # 基础信息
        if "title" in details:
            result["title"] = details["title"]
            result["movie_title"] = details["title"]

        if "release_date" in details and details["release_date"]:
            try:
                year = int(details["release_date"][:4])
                result["year"] = year
                result["release_year"] = year
            except (ValueError, IndexError):
                pass

        if "overview" in details:
            result["overview"] = details["overview"]
            result["description"] = details["overview"]
            result["summary"] = details["overview"]

        # 类型
        if "genres" in details and details["genres"]:
            genres = [g["name"] for g in details["genres"]]
            result["genre"] = ", ".join(genres)
            result["genres"] = genres

        # 预算和票房
        if "budget" in details and details["budget"]:
            result["budget"] = details["budget"]

        if "revenue" in details and details["revenue"]:
            result["revenue"] = details["revenue"]

        # 时长
        if "runtime" in details and details["runtime"]:
            result["runtime"] = details["runtime"]

        # IMDb ID
        if "imdb_id" in details and details["imdb_id"]:
            result["imdb_id"] = details["imdb_id"]

        # 评分
        if "vote_average" in details:
            result["tmdb_rating"] = details["vote_average"]

        # 演职人员
        if credits:
            # 导演
            crew = credits.get("crew", [])
            directors = [
                person["name"] for person in crew
                if person.get("job") == "Director"
            ]
            if directors:
                result["director"] = directors[0]  # 第一个导演
                result["directors"] = directors

            # 演员（前5名）
            cast = credits.get("cast", [])
            if cast:
                actors = [person["name"] for person in cast[:5]]
                result["cast"] = ", ".join(actors)
                result["actors"] = actors

        return result