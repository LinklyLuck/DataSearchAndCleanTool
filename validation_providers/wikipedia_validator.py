# validation_providers/wikipedia_validator.py
"""
Wikipedia Validation Provider

使用Wikipedia作为权威数据源验证数据准确性
"""
import re
import json
from typing import List, Dict, Optional, Any, Tuple
import polars as pl
from .base import ValidationProvider, ValidationReport


class WikipediaValidationProvider(ValidationProvider):
    """
    Wikipedia数据验证

    功能：
    - 查询Wikipedia获取权威数据
    - 对比现有数据
    - 修正不一致的数据

    支持验证的字段类型：
    - 年份/日期
    - 描述/摘要
    - 类别/类型
    """

    name = "Wikipedia"

    def __init__(self, llm_client=None, **kwargs):
        super().__init__(llm_client=llm_client, **kwargs)


    async def validate(
            self,
            df: pl.DataFrame,
            primary_keys: List[str],
            validate_columns: Optional[List[str]] = None
    ) -> Tuple[pl.DataFrame, ValidationReport]:
        """验证数据准确性"""
        if df.height == 0:
            return df, self._create_report([], 0, 0)

        # 确定要验证的列
        if validate_columns is None:
            validate_columns = [c for c in df.columns if c not in primary_keys]

        print(f"[Wikipedia-Validation] Validating {len(validate_columns)} columns...")

        corrections = []
        validated_cells = 0
        validated_df = df.clone()

        # 转换为行字典
        rows = df.to_dicts()

        for i, row in enumerate(rows):
            # 提取实体名称
            entity_name = self._extract_entity_name(row, primary_keys)
            if not entity_name:
                continue

            # 查询Wikipedia
            wiki_data = await self._query_wikipedia(entity_name)
            if not wiki_data:
                continue

            # 验证每个列
            for col in validate_columns:
                if col not in row:
                    continue

                current_value = row[col]
                validated_cells += 1

                # 从Wikipedia数据中查找对应字段
                wiki_value = self._find_matching_field(col, wiki_data)

                if wiki_value is not None and not self._values_match(current_value, wiki_value):
                    # 值不匹配，使用Wikipedia的值
                    pk_info = {k: row[k] for k in primary_keys if k in row}
                    corrections.append({
                        "row_index": i,
                        **pk_info,
                        "column": col,
                        "old_value": current_value,
                        "new_value": wiki_value,
                        "source": "Wikipedia",
                        "reason": f"Corrected from Wikipedia: {entity_name}"
                    })

        # 应用修正
        if corrections:
            validated_df = self._apply_corrections(validated_df, corrections)

        report = self._create_report(corrections, validated_cells, len(df))

        if report.corrected_cells > 0:
            print(f"[Wikipedia-Validation] ✓ Corrected {report.corrected_cells} cells")

        return validated_df, report

    async def _query_wikipedia(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        查询Wikipedia获取实体信息

        Returns:
            包含实体信息的字典，或None
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
            result = await self._extract_fields(entity_name, summary_data)

            # 缓存结果
            self._cache[cache_key] = result
            return result

        except Exception:
            return None

    async def _wiki_search(self, query: str) -> Optional[str]:
        """搜索Wikipedia，返回最佳匹配的标题"""
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
            if isinstance(data, list) and len(data) >= 2 and data[1]:
                return data[1][0]
        except:
            pass

        return None

    async def _wiki_summary(self, title: str) -> Optional[Dict[str, Any]]:
        """获取Wikipedia页面摘要"""
        safe_title = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"

        response = await self._make_request("GET", url)
        if not response:
            return None

        try:
            return response.json()
        except:
            return None

    async def _extract_fields(self, entity_name: str, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """从Wikipedia摘要提取结构化字段"""
        result: Dict[str, Any] = {}

        # 标题
        if "title" in summary_data:
            result["title"] = summary_data["title"]
            result["name"] = summary_data["title"]

        # 描述
        if "description" in summary_data:
            result["description"] = summary_data["description"]

        # 摘要
        if "extract" in summary_data:
            result["summary"] = summary_data["extract"]

        # 提取年份
        text = summary_data.get("extract", "")
        year = self._extract_year(text)
        if year:
            result["year"] = year
            result["release_year"] = year

        # 提取类别（优先使用LLM判断）
        description = summary_data.get("description", "")
        category = await self._extract_category(entity_name, description, summary_data.get("extract", ""))
        if category:
            result["category"] = category
            result["type"] = category

        return result

    def _extract_year(self, text: str) -> Optional[int]:
        """从文本中提取年份"""
        if not text:
            return None

        patterns = [
            r'\b(19\d{2}|20\d{2})\b',
            r'\((\d{4})\)',
            r'in (\d{4})',
            r'released (\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    year = int(match.group(1))
                    if 1900 <= year <= 2099:
                        return year
                except:
                    continue

        return None

    async def _extract_category(
            self,
            entity_name: str,
            description: str,
            extract_text: str
    ) -> Optional[str]:
        """使用LLM（可选）或通用规则提取类别/类型"""
        combined = " ".join([description or "", extract_text or ""]).strip()
        if not combined:
            return None

            # 优先使用LLM做判断
            if getattr(self, "llm_client", None) is not None:
                prompt = (
                    "You are a data quality analyst. Based on the description below, "
                    "identify the most concise high-level category or type for the entity. "
                    "Return STRICT JSON with schema {\"category\": string|null}. "
                    "Use lowercase words separated by spaces or hyphen, avoid extra commentary. "
                    "If you are unsure, return null.\n\n"
                    f"Entity: {entity_name}\n"
                    f"Description: {description or 'N/A'}\n"
                    f"Summary: {extract_text or 'N/A'}"
                )
                try:
                    response = await self.llm_client.chat("gpt-4o-mini", [
                        {"role": "user", "content": prompt}
                    ])
                    data = json.loads(response.strip())
                    category = data.get("category") if isinstance(data, dict) else None
                    if category:
                        return str(category).strip()
                except Exception:
                    pass

            # 回退到通用正则：提取 "is a ..." 片段
            simple = self._simple_category(combined)
            return simple

    def _simple_category(self, text: str) -> Optional[str]:
        """使用通用语法模式提取类别"""
        patterns = [
            r"\bis a[n]?\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,60})",
            r"\bwas a[n]?\s+([a-zA-Z][a-zA-Z0-9\-\s]{2,60})",
        ]
        lowered = text.strip()
        for pat in patterns:
            match = re.search(pat, lowered, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().lower()
                # 清理尾部标点
                candidate = re.sub(r"[^a-z0-9\-\s]$", "", candidate)
                if candidate:
                    return candidate

        return None

    def _find_matching_field(
            self,
            column_name: str,
            wiki_data: Dict[str, Any]
    ) -> Optional[Any]:
        """在Wikipedia数据中查找与列名匹配的字段"""
        col_lower = column_name.lower()

        # 直接匹配
        if column_name in wiki_data:
            return wiki_data[column_name]

        # 模糊匹配
        for wiki_key, wiki_value in wiki_data.items():
            wiki_key_lower = wiki_key.lower()

            # 年份类字段
            if "year" in col_lower and "year" in wiki_key_lower:
                return wiki_value

            # 描述类字段
            if any(kw in col_lower for kw in ["description", "summary", "overview"]):
                if any(kw in wiki_key_lower for kw in ["description", "summary", "extract"]):
                    return wiki_value

            # 名称类字段
            if any(kw in col_lower for kw in ["name", "title"]):
                if any(kw in wiki_key_lower for kw in ["name", "title"]):
                    return wiki_value

            # 类别类字段
            if any(kw in col_lower for kw in ["category", "type", "genre"]):
                if any(kw in wiki_key_lower for kw in ["category", "type"]):
                    return wiki_value

        return None

    def _values_match(self, value1: Any, value2: Any) -> bool:
        """判断两个值是否匹配"""
        if value1 is None and value2 is None:
            return True

        if value1 is None or value2 is None:
            return False

        # 转换为字符串比较
        str1 = str(value1).strip().lower()
        str2 = str(value2).strip().lower()

        # 完全相同
        if str1 == str2:
            return True

        # 年份特殊处理：提取数字
        if self._extract_year(str1) and self._extract_year(str2):
            year1 = self._extract_year(str1)
            year2 = self._extract_year(str2)
            return year1 == year2

        # 描述文本：如果一个是另一个的子串，认为匹配
        if len(str1) > 20 and len(str2) > 20:
            if str1 in str2 or str2 in str1:
                return True

        return False

    def _apply_corrections(
            self,
            df: pl.DataFrame,
            corrections: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """应用验证修正"""
        if not corrections:
            return df

        # 按列分组修正
        corrections_by_col = {}
        for c in corrections:
            col = c["column"]
            if col not in corrections_by_col:
                corrections_by_col[col] = {}
            corrections_by_col[col][c["row_index"]] = c["new_value"]

        # 应用修正
        for col, change_map in corrections_by_col.items():
            if col not in df.columns:
                continue

            new_values = []
            for i in range(len(df)):
                if i in change_map:
                    new_values.append(change_map[i])
                else:
                    new_values.append(df[col][i])

            df = df.with_columns([
                pl.Series(col, new_values)
            ])

        return df