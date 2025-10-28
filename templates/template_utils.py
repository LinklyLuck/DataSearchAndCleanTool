# template_utils.py
"""
Template驱动的智能Schema映射工具

核心功能:
1. 动态生成领域模板（无需预定义domain）
2. 用alias_rules指导LLM做准确的字段映射
3. 缓存和复用生成的模板
"""

from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class DynamicTemplate:
    """动态生成的领域模板"""
    domain: str  # 领域名称（由LLM生成）
    keywords: List[str]  # 关键词
    feature_sets: List[List[str]]  # 特征集
    join_keys: List[str]  # 连接键
    canonical_order: List[str]  # 规范列顺序
    alias_rules: Dict[str, str]  # 字段别名规则（核心！）
    policy: Dict[str, Any]  # 策略配置

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DynamicTemplate:
        return cls(**d)


class TemplateManager:
    """模板管理器 - 动态生成和应用模板"""

    def __init__(self, llm_client, cache_path: Path = Path("templates/dynamic_templates.json")):
        self.llm = llm_client
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, DynamicTemplate]:
        """加载已缓存的模板"""
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return {k: DynamicTemplate.from_dict(v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_cache(self):
        """保存模板到缓存"""
        data = {k: v.to_dict() for k, v in self._cache.items()}
        self.cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _query_hash(self, query: str) -> str:
        """生成query的hash作为缓存key"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]

    async def generate_template(
            self,
            query: str,
            discovered_datasets: List[Dict[str, Any]]
    ) -> DynamicTemplate:
        """
        核心方法：根据query和发现的数据集，动态生成template

        Args:
            query: 自然语言查询
            discovered_datasets: 已发现的数据集元信息 [{"path": ..., "columns": [...], "sample": {...}}]

        Returns:
            DynamicTemplate
        """
        # 1. 检查缓存
        cache_key = self._query_hash(query)
        if cache_key in self._cache:
            print(f"[TemplateManager] 使用缓存的template: {self._cache[cache_key].domain}")
            return self._cache[cache_key]

        # 2. LLM生成template
        print(f"[TemplateManager] 为query动态生成template...")

        # 准备数据集信息
        datasets_info = []
        for ds in discovered_datasets[:5]:  # 限制数量避免prompt过长
            datasets_info.append({
                "path": ds.get("path", "unknown"),
                "columns": ds.get("columns", [])[:20],  # 限制列数
                "sample": ds.get("sample", {})
            })

        prompt = self._build_template_generation_prompt(query, datasets_info)

        try:
            response = await self.llm.chat("gpt-4o", [
                {"role": "system", "content": "You are a data schema expert. Generate structured templates in JSON."},
                {"role": "user", "content": prompt}
            ])

            # 解析LLM返回的JSON
            template_dict = self._parse_llm_response(response)
            template = DynamicTemplate(**template_dict)

            # 3. 缓存
            self._cache[cache_key] = template
            self._save_cache()

            print(f"[TemplateManager] 生成template: {template.domain}")
            print(f"  - join_keys: {template.join_keys}")
            print(f"  - feature_sets: {len(template.feature_sets)} sets")
            print(f"  - alias_rules: {len(template.alias_rules)} rules")

            return template

        except Exception as e:
            print(f"[TemplateManager] 生成template失败: {e}, 使用fallback")
            return self._fallback_template(query, discovered_datasets)

    def _build_template_generation_prompt(self, query: str, datasets_info: List[Dict]) -> str:
        """构建生成template的prompt"""

        datasets_str = json.dumps(datasets_info, indent=2, ensure_ascii=False)

        prompt = f"""Analyze this natural language query and the discovered datasets, then generate a comprehensive domain template.

**Query:**
{query}

**Discovered Datasets:**
{datasets_str}

**Your Task:**
Generate a JSON template with these fields:

1. **domain**: A short domain name (e.g., "movie", "nba", "stock", "ecommerce")
2. **keywords**: List of 5-10 keywords related to this domain
3. **feature_sets**: 1-3 alternative feature lists that would answer the query. Each list should have 6-12 column names.
4. **join_keys**: 2-4 columns that uniquely identify entities (for joining datasets)
5. **canonical_order**: The best order for columns in the final dataset
6. **alias_rules**: CRITICAL! A dictionary mapping regex patterns to canonical column names.
   - This prevents LLM from mapping "title" → "movie_name", "film_title", etc.
   - Format: {{"^(pattern1|pattern2)$": "canonical_name"}}
   - Cover all common variations for each important field
7. **policy**: {{"strong_keys": [...], "join_min_keys": 2, "same_origin": true, "max_blowup_ratio": 1.8}}

**Alias Rules Example (VERY IMPORTANT):**
```json
{{
  "^(title|movie_title|film_title|name)$": "title",
  "^(year|release_year|pub_year|date)$": "year",
  "^(director|director_name)$": "director_name",
  "^(team|team_name|franchise)$": "team_name",
  "^(player|player_name|athlete)$": "player_name"
}}
```

**Output Format:**
Return ONLY valid JSON, no markdown blocks:
{{
  "domain": "...",
  "keywords": [...],
  "feature_sets": [[...], [...]],
  "join_keys": [...],
  "canonical_order": [...],
  "alias_rules": {{}},
  "policy": {{}}
}}"""

        return prompt

    def _parse_llm_response(self, response: str) -> dict:
        """解析LLM返回的JSON"""
        # 去除可能的markdown标记
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```(?:json)?\n', '', response)
            response = re.sub(r'\n```$', '', response)

        return json.loads(response)

    def _fallback_template(self, query: str, datasets: List[Dict]) -> DynamicTemplate:
        """兜底：生成一个通用template"""
        # 从数据集中提取所有列名
        all_cols = set()
        for ds in datasets:
            all_cols.update(ds.get("columns", []))

        all_cols_list = sorted(list(all_cols))[:15]

        return DynamicTemplate(
            domain="generic",
            keywords=query.lower().split()[:5],
            feature_sets=[all_cols_list],
            join_keys=all_cols_list[:2] if len(all_cols_list) >= 2 else all_cols_list,
            canonical_order=all_cols_list,
            alias_rules={
                "^(name|title|entity|id)$": "entity_name",
                "^(date|time|timestamp)$": "date",
                "^(value|score|rating)$": "value"
            },
            policy={
                "strong_keys": all_cols_list[:2] if len(all_cols_list) >= 2 else all_cols_list,
                "join_min_keys": 1,
                "same_origin": True,
                "max_blowup_ratio": 2.0
            }
        )

    def apply_alias_rules(
            self,
            column_name: str,
            template: DynamicTemplate
    ) -> str:
        """
        应用alias_rules，将列名映射到规范名称

        这是解决"LLM把title映射成各种名字"问题的核心方法！
        """
        col_lower = column_name.lower().strip()

        for pattern, canonical in template.alias_rules.items():
            if re.match(pattern, col_lower, re.IGNORECASE):
                return canonical

        # 无匹配则返回原名
        return column_name

    def get_mapping_guidance(self, template: DynamicTemplate) -> str:
        """
        生成一个给LLM的映射指导文本

        在pipeline的schema映射步骤中，把这段文本加到prompt里，
        让LLM按照alias_rules来映射字段名
        """
        rules_text = "\n".join([
            f"  - Fields matching '{pattern}' → use canonical name '{canonical}'"
            for pattern, canonical in template.alias_rules.items()
        ])

        guidance = f"""**CRITICAL SCHEMA MAPPING RULES:**
You MUST follow these alias rules to ensure consistent column naming:

{rules_text}

**Example:**
- If you see "movie_title", "film_title", "title" → ALL map to "title"
- If you see "release_year", "year", "pub_year" → ALL map to "year"

DO NOT create new column names that are variations of these canonical names.
ALWAYS use the exact canonical name from the rules above.
"""
        return guidance

    def validate_schema_mapping(
            self,
            mapped_columns: Dict[str, str],
            template: DynamicTemplate
    ) -> Tuple[bool, List[str]]:
        """
        验证schema映射是否符合alias_rules

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        for source_col, target_col in mapped_columns.items():
            # 检查target_col是否应该被规范化
            expected = self.apply_alias_rules(target_col, template)
            if expected != target_col:
                errors.append(
                    f"Column '{source_col}' mapped to '{target_col}', "
                    f"but should be '{expected}' according to alias_rules"
                )

        return (len(errors) == 0, errors)

    def get_join_key_candidates(
            self,
            columns: List[str],
            template: DynamicTemplate
    ) -> List[str]:
        """
        根据template的join_keys和alias_rules，从列名中提取候选连接键
        """
        candidates = []

        for col in columns:
            # 规范化列名
            canonical = self.apply_alias_rules(col, template)

            # 检查是否是join_key
            if canonical in template.join_keys or col in template.join_keys:
                candidates.append(canonical)

        return list(set(candidates))


class SchemaMapper:
    """
    配合TemplateManager使用的Schema映射器

    在pipeline的map阶段，用这个类来确保LLM按照template的alias_rules映射字段
    """

    def __init__(self, llm_client, template_manager: TemplateManager):
        self.llm = llm_client
        self.tm = template_manager

    async def map_schema_with_template(
            self,
            source_columns: List[str],
            source_sample: Dict[str, Any],
            template: DynamicTemplate,
            target_schema: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        使用template的alias_rules指导LLM进行schema映射

        Args:
            source_columns: 源数据集的列名
            source_sample: 源数据集的样本行
            template: 动态生成的template
            target_schema: 目标schema（可选，默认用template的canonical_order）

        Returns:
            映射字典 {source_col: target_col}
        """
        target_schema = target_schema or template.canonical_order

        # 构建带alias_rules指导的prompt
        guidance = self.tm.get_mapping_guidance(template)

        prompt = f"""You are a data schema mapper. Map source columns to target schema.

**Source Columns:**
{json.dumps(source_columns, indent=2)}

**Sample Data:**
{json.dumps(source_sample, indent=2, ensure_ascii=False)}

**Target Schema:**
{json.dumps(target_schema, indent=2)}

{guidance}

**Instructions:**
1. Map each source column to the most appropriate target column
2. STRICTLY follow the alias rules above
3. If no good match, use null
4. Return ONLY a JSON object: {{"source_col": "target_col" or null}}

**Output (JSON only):**"""

        try:
            response = await self.llm.chat("gpt-4o-mini", [
                {"role": "user", "content": prompt}
            ])

            # 解析映射
            mapping = json.loads(response.strip())

            # 验证映射
            is_valid, errors = self.tm.validate_schema_mapping(mapping, template)
            if not is_valid:
                print(f"[SchemaMapper] Warning: Mapping validation failed:")
                for err in errors:
                    print(f"  - {err}")
                # 自动修正
                mapping = self._auto_correct_mapping(mapping, template)

            return mapping

        except Exception as e:
            print(f"[SchemaMapper] Mapping failed: {e}, using fallback")
            return self._fallback_mapping(source_columns, target_schema, template)

    def _auto_correct_mapping(
            self,
            mapping: Dict[str, str],
            template: DynamicTemplate
    ) -> Dict[str, str]:
        """自动修正不符合alias_rules的映射"""
        corrected = {}
        for src, tgt in mapping.items():
            if tgt is not None:
                corrected[src] = self.tm.apply_alias_rules(tgt, template)
            else:
                corrected[src] = tgt
        return corrected

    def _fallback_mapping(
            self,
            source_columns: List[str],
            target_schema: List[str],
            template: DynamicTemplate
    ) -> Dict[str, str]:
        """兜底映射：简单的字符串匹配"""
        from difflib import SequenceMatcher

        mapping = {}
        for src in source_columns:
            # 规范化
            src_canonical = self.tm.apply_alias_rules(src, template)

            # 在target_schema中找最相似的
            best_match = None
            best_score = 0.0
            for tgt in target_schema:
                score = SequenceMatcher(None, src_canonical.lower(), tgt.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = tgt

            mapping[src] = best_match if best_score > 0.6 else None

        return mapping


# ============== 工具函数 ==============

def load_or_generate_template(
        llm_client,
        query: str,
        discovered_datasets: List[Dict[str, Any]],
        cache_path: Path = Path("templates/dynamic_templates.json")
) -> DynamicTemplate:
    """
    便捷函数：加载或生成template

    在pipeline开始时调用一次即可
    """
    import asyncio
    tm = TemplateManager(llm_client, cache_path)
    return asyncio.run(tm.generate_template(query, discovered_datasets))


def get_schema_mapper(llm_client, template: DynamicTemplate) -> SchemaMapper:
    """
    便捷函数：获取SchemaMapper实例
    """
    tm = TemplateManager(llm_client)
    tm._cache["current"] = template  # 临时存储当前template
    return SchemaMapper(llm_client, tm)