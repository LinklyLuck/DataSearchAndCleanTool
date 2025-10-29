# validation_providers/tmdb_validator.py
"""
TMDb (The Movie Database) Validation Provider

使用TMDb API验证电影/电视剧数据准确性
需要环境变量：TMDB_API_KEY

官方文档：https://developers.themoviedb.org/3
"""
import os
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import ValidationProvider, ValidationReport


class TMDbValidationProvider(ValidationProvider):
    """
    TMDb数据验证

    功能：
    - 验证电影/电视剧标题
    - 验证年份
    - 验证导演
    - 验证类型
    - 验证预算/票房等
    """

    name = "TMDb"
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("TMDB_API_KEY")
        super().__init__(api_key=api_key, **kwargs)

        if not self.api_key:
            print("[WARN] TMDb API key not found (set TMDB_API_KEY)")

    async def validate(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            validate_columns: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, ValidationReport]:
        """验证数据准确性"""
        if not self.api_key:
            return df, self._create_report([], 0, 0)

        if df.height == 0:
            return df, self._create_report([], 0, 0)

        # 确定要验证的列
        if validate_columns is None:
            validate_columns = [c for c in df.columns if c not in primary_keys]

        print(f"[TMDb-Validation] Validating {len(validate_columns)} columns...")

        corrections = []
        validated_cells = 0
        validated_df = df.clone()

        rows = df.to_dicts()

        for i, row in enumerate(rows):
            # 提取电影标题
            entity_name = self._extract_entity_name(row, primary_keys)
            if not entity_name:
                continue

            # 查询TMDb
            tmdb_data = await self._query_tmdb(entity_name, row.get("year"))
            if not tmdb_data:
                continue

            # 验证每个列
            for col in validate_columns:
                if col not in row:
                    continue

                current_value = row[col]
                validated_cells += 1

                # 从TMDb数据中查找对应字段
                tmdb_value = self._find_matching_field(col, tmdb_data)

                if tmdb_value is not None and not self._values_match(current_value, tmdb_value):
                    pk_info = {k: row[k] for k in primary_keys if k in row}
                    corrections.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "old_value": current_value,
                        "new_value": tmdb_value,
                        "source": "TMDb",
                        "reason": f"Corrected from TMDb: {entity_name}"
                    })

        # 应用修正
        if corrections:
            validated_df = self._apply_corrections(validated_df, corrections)

        report = self._create_report(corrections, validated_cells, len(df))

        if report.corrected_cells > 0:
            print(f"[TMDb-Validation] ✓ Corrected {report.corrected_cells} cells")

        return validated_df, report

    async def _query_tmdb(
            self,
            movie_title: str,
            year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """查询TMDb电影信息"""
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

            self._cache[cache_key] = result
            return result

        except Exception:
            return None

    async def _search_movie(
            self,
            title: str,
            year: Optional[int] = None
    ) -> Optional[int]:
        """搜索电影，返回movie_id"""
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
            if results:
                return results[0]["id"]
        except:
            pass

        return None

    async def _get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """获取电影详细信息"""
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
        except:
            return None

    async def _get_movie_credits(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """获取演职人员信息"""
        url = f"{self.BASE_URL}/movie/{movie_id}/credits"
        params = {"api_key": self.api_key}

        response = await self._make_request("GET", url, params=params)
        if not response:
            return None

        try:
            return response.json()
        except:
            return None

    def _extract_fields(
            self,
            details: Dict[str, Any],
            credits: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """从TMDb响应中提取字段"""
        result = {}

        # 基础信息
        if "title" in details:
            result["title"] = details["title"]

        if "release_date" in details and details["release_date"]:
            try:
                year = int(details["release_date"][:4])
                result["year"] = year
            except:
                pass

        if "overview" in details:
            result["description"] = details["overview"]

        # 类型
        if "genres" in details and details["genres"]:
            genres = [g["name"] for g in details["genres"]]
            result["genre"] = ", ".join(genres)

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
            result["rating"] = details["vote_average"]

        # 演职人员
        if credits:
            crew = credits.get("crew", [])
            directors = [p["name"] for p in crew if p.get("job") == "Director"]
            if directors:
                result["director"] = directors[0]

            cast = credits.get("cast", [])
            if cast:
                actors = [p["name"] for p in cast[:5]]
                result["cast"] = ", ".join(actors)

        return result

    def _find_matching_field(
            self,
            column_name: str,
            tmdb_data: Dict[str, Any]
    ) -> Optional[Any]:
        """在TMDb数据中查找匹配字段"""
        col_lower = column_name.lower()

        # 直接匹配
        if column_name in tmdb_data:
            return tmdb_data[column_name]

        # 模糊匹配
        field_mappings = {
            "year": ["year", "release_year"],
            "title": ["title", "name"],
            "director": ["director"],
            "genre": ["genre", "genres"],
            "budget": ["budget"],
            "revenue": ["revenue", "box_office"],
            "runtime": ["runtime", "duration"],
            "rating": ["rating", "score"],
            "description": ["description", "overview", "summary"],
        }

        for tmdb_key, aliases in field_mappings.items():
            if any(alias in col_lower for alias in aliases):
                if tmdb_key in tmdb_data:
                    return tmdb_data[tmdb_key]

        return None

    def _values_match(self, value1: Any, value2: Any) -> bool:
        """判断两个值是否匹配"""
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False

        str1 = str(value1).strip().lower()
        str2 = str(value2).strip().lower()

        if str1 == str2:
            return True

        # 数值比较
        try:
            num1 = float(str1.replace(",", ""))
            num2 = float(str2.replace(",", ""))
            return abs(num1 - num2) < 0.01
        except:
            pass

        return False

    def _apply_corrections(
            self,
            df: pl.DataFrame,
            corrections: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用验证修正"""
        if not corrections:
            return df

        corrections_by_col = {}
        for c in corrections:
            col = c["column"]
            if col not in corrections_by_col:
                corrections_by_col[col] = {}
            corrections_by_col[col][c["row_index"]] = c["new_value"]

        for col, change_map in corrections_by_col.items():
            if col not in df.columns:
                continue

            new_values = []
            for i in range(len(df)):
                if i in change_map:
                    new_values.append(change_map[i])
                else:
                    new_values.append(df[col][i])

            df = df.with_columns([pl.Series(col, new_values)])

        return df