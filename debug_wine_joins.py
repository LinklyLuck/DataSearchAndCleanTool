#!/usr/bin/env python3
"""诊断脚本：检查 wine 数据集为什么 join 失败"""
import polars as pl
from pathlib import Path

# 读取所有源文件
files = {
    "wine_acid_sugar": "datalake/wine_acid_sugar.csv",
    "wine_components": "datalake/wine_components.csv",
    "wine_quality_red_327": "datalake/wine_quality_red_327.csv",
    "winequality_red_xxx": "datalake/winequality_red_xxx.csv",
}

dfs = {}
for name, path in files.items():
    p = Path(path)
    if p.exists():
        dfs[name] = pl.read_csv(p)
        print(f"\n{'=' * 60}")
        print(f"📄 {name}")
        print(f"{'=' * 60}")
        print(f"Rows: {dfs[name].height}, Cols: {dfs[name].width}")
        print(f"Columns: {dfs[name].columns}")
        print(f"Sample row:\n{dfs[name].head(1)}")
    else:
        print(f"❌ {path} not found")

# 检查哪些表有 quality 和 residual sugar
print(f"\n{'=' * 60}")
print("🔍 关键字段分布")
print(f"{'=' * 60}")
for name, df in dfs.items():
    has_quality = "quality" in df.columns
    has_residual = "residual sugar" in df.columns
    has_wine_id = "wine_id" in df.columns
    print(f"{name:25} | quality={has_quality:5} | residual_sugar={has_residual:5} | wine_id={has_wine_id:5}")

# 检查可能的 join keys
print(f"\n{'=' * 60}")
print("🔑 可能的 Join Keys 分析")
print(f"{'=' * 60}")

base = dfs.get("wine_components")
if base:
    print(f"假设 base = wine_components (最宽的表)")
    base_cols = set(base.columns)

    for name, df in dfs.items():
        if name == "wine_components":
            continue
        common = base_cols & set(df.columns)
        print(f"\n{name}:")
        print(f"  Common columns: {list(common) if common else 'NONE! ❌'}")

        if common:
            # 检查这些列是否适合做 join key（重复度）
            for col in common:
                base_uniq = base[col].n_unique()
                df_uniq = df[col].n_unique()
                print(f"    {col}: base有{base_uniq}个唯一值, 该表有{df_uniq}个唯一值")

# 尝试 join
print(f"\n{'=' * 60}")
print("🔗 模拟 Join 过程")
print(f"{'=' * 60}")

if "wine_components" in dfs and "wine_acid_sugar" in dfs:
    base = dfs["wine_components"]
    right = dfs["wine_acid_sugar"]

    # 找共同列
    common = list(set(base.columns) & set(right.columns))
    print(f"\nwine_components + wine_acid_sugar:")
    print(f"  Join keys candidate: {common}")

    if common:
        try:
            # 尝试 inner join
            joined = base.join(right, on=common, how="inner")
            print(f"  ✅ Join成功！结果: {joined.height} rows, {joined.width} cols")
            print(f"  Columns: {joined.columns}")

            # 检查是否有 residual sugar
            if "residual sugar" in joined.columns:
                print(f"  ✅ residual sugar 保留成功！")
            else:
                print(f"  ❌ residual sugar 丢失！")
        except Exception as e:
            print(f"  ❌ Join失败: {e}")
    else:
        print(f"  ❌ 没有共同列，无法 join！")
        print(f"\n  【建议】需要找到其他连接方式，或者这两张表本来就不应该 join")

print("\n" + "=" * 60)
print("💡 诊断建议")
print("=" * 60)
print("""
如果看到 'NONE! ❌'，说明表之间没有共同列可以 join。
可能的解决方案：
1. 如果 wine_id 只在部分表中，考虑用行号或其他隐式索引
2. 如果表本身就是独立的，可能需要改用 concat 而不是 join
3. 检查是否有同义列名（如 id vs wine_id）需要提前重命名
""")