# validation_providers/omdb_validator.py
"""
OMDb (Open Movie Database) Validation Provider

使用OMDb API验证电影数据准确性
需要环境变量：OMDB_API_KEY

官方文档：http://www.omdbapi.com/
"""
import os
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import ValidationProvider, ValidationReport


class OMDbValidationProvider(ValidationProvider):
    """
    OMDb数据验证

    功能：
    - 验证电影标题
    - 验证年份
    - 验证导演
    - 验证演员
    - 验证类型
    - 验证IMDb评分
    """

    name = "OMDb"
    BASE_URL = "http://www.omdbapi.com/"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.getenv("OMDB_API_KEY")
        super().__init__(api_key=api_key, **kwargs)

        if not self.api_key:
            print("[WARN] OMDb API key not found (set OMDB_API_KEY)")

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

        if validate_columns is None:
            validate_columns = [c for c in df.columns if c not in primary_keys]

        print(f"[OMDb-Validation] Validating {len(validate_columns)} columns...")

        corrections = []
        validated_cells = 0
        validated_df = df.clone()

        rows = df.to_dicts()

        for i, row in enumerate(rows):
            entity_name = self._extract_entity_name(row, primary_keys)
            if not entity_name:
                continue

            omdb_data = await self._query_omdb(entity_name, row.get("year"))
            if not omdb_data:
                continue

            for col in validate_columns:
                if col not in row:
                    continue

                current_value = row[col]
                validated_cells += 1

                omdb_value = self._find_matching_field(col, omdb_data)

                if omdb_value is not None and not self._values_match(current_value, omdb_value):
                    pk_info = {k: row[k] for k in primary_keys if k in row}
                    corrections.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "old_value": current_value,
                        "new_value": omdb_value,
                        "source": "OMDb",
                        "reason": f"Corrected from OMDb: {entity_name}"
                    })

        if corrections:
            validated_df = self._apply_corrections(validated_df, corrections)

        report = self._create_report(corrections, validated_cells, len(df))

        if report.corrected_cells > 0:
            print(f"[OMDb-Validation] ✓ Corrected {report.corrected_cells} cells")

        return validated_df, report

    async def _query_omdb(
            self,
            title: str,
            year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """查询OMDb电影信息"""
        cache_key = f"omdb:{title.lower()}:{year or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            params = {
                "apikey": self.api_key,
                "t": title,
                "type": "movie",
                "plot": "full"
            }

            if year:
                params["y"] = year

            response = await self._make_request("GET", self.BASE_URL, params=params)
            if not response:
                return None

            data = response.json()

            if data.get("Response") != "True":
                return None

            result = self._extract_fields(data)
            self._cache[cache_key] = result
            return result

        except Exception:
            return None

    def _extract_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从OMDb响应中提取字段"""
        result = {}

        if "Title" in data:
            result["title"] = data["Title"]

        if "Year" in data:
            try:
                year_str = data["Year"].split("–")[0]
                result["year"] = int(year_str)
            except:
                pass

        if "Director" in data and data["Director"] != "N/A":
            result["director"] = data["Director"]

        if "Actors" in data and data["Actors"] != "N/A":
            result["cast"] = data["Actors"]

        if "Genre" in data and data["Genre"] != "N/A":
            result["genre"] = data["Genre"]

        if "Runtime" in data and data["Runtime"] != "N/A":
            try:
                runtime_str = data["Runtime"].replace(" min", "")
                result["runtime"] = int(runtime_str)
            except:
                pass

        if "Plot" in data and data["Plot"] != "N/A":
            result["description"] = data["Plot"]

        if "imdbRating" in data and data["imdbRating"] != "N/A":
            try:
                result["imdb_rating"] = float(data["imdbRating"])
            except:
                pass

        if "imdbID" in data and data["imdbID"] != "N/A":
            result["imdb_id"] = data["imdbID"]

        if "BoxOffice" in data and data["BoxOffice"] != "N/A":
            result["box_office"] = data["BoxOffice"]

        return result

    def _find_matching_field(
            self,
            column_name: str,
            omdb_data: Dict[str, Any]
    ) -> Optional[Any]:
        """在OMDb数据中查找匹配字段"""
        col_lower = column_name.lower()

        if column_name in omdb_data:
            return omdb_data[column_name]

        field_mappings = {
            "year": ["year"],
            "title": ["title"],
            "director": ["director"],
            "cast": ["cast", "actors"],
            "genre": ["genre"],
            "runtime": ["runtime", "duration"],
            "imdb_rating": ["imdb_rating", "rating"],
            "imdb_id": ["imdb_id"],
            "description": ["description", "plot"],
            "box_office": ["box_office", "revenue"],
        }

        for omdb_key, aliases in field_mappings.items():
            if any(alias in col_lower for alias in aliases):
                if omdb_key in omdb_data:
                    return omdb_data[omdb_key]

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
            num1 = float(str1.replace(",", "").replace("$", ""))
            num2 = float(str2.replace(",", "").replace("$", ""))
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