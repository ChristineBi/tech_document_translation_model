import pandas as pd
import json
import re
import collections

# ── 读取 ──────────────────────────────────────────
df = pd.read_csv('data/processed/cleaned.csv', encoding='utf-8-sig')
with open('data/glossary_reviewed.json', 'r', encoding='utf-8') as f:
    glossary = json.load(f)

print(f'替换前: {len(df)} 条')

# ── 替换函数 ──────────────────────────────────────
def replace_terms(text, lang, glossary):
    # 按key长度从长到短排序，避免短词先匹配
    sorted_terms = sorted(glossary.items(), key=lambda x: len(x[0]), reverse=True)
    for zh_term, trans in sorted_terms:
        target = trans.get(lang, '')  # 直接取，不判断note
        if target and zh_term in text:
            text = text.replace(zh_term, target)
    return text

# ── 应用替换 ──────────────────────────────────────
df['japanese'] = df['japanese'].apply(lambda x: replace_terms(x, 'ja', glossary))
df['indonesian'] = df['indonesian'].apply(lambda x: replace_terms(x, 'id', glossary))

# ── 验证替换效果 ──────────────────────────────────
def chinese_char_ratio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    return len(re.findall(r'[\u4e00-\u9fff]', text)) / len(text)

df['zh_ratio_in_id'] = df['indonesian'].apply(chinese_char_ratio)
still_mixed_id = (df['zh_ratio_in_id'] > 0.05).sum()
print(f'替换后印尼语仍有混语: {still_mixed_id} 条')
# 注意：日语混语率不作为质量指标，因为日语本身含汉字，会产生误报
# 仅用印尼语混语率验证替换效果


# 打印几条看效果
print('\n替换效果示例（前3条）:')
for _, row in df.head(3).iterrows():
    print(f'  中文: {row["chinese"]}')
    print(f'  日语: {row["japanese"]}')
    print(f'  印尼: {row["indonesian"]}')
    print()

# ── 保存 ──────────────────────────────────────────
df = df.drop(columns=['zh_ratio_in_ja', 'zh_ratio_in_id', 'has_mixed_lang'],
             errors='ignore')
df.to_csv('data/processed/final.csv', index=False, encoding='utf-8-sig')
print('已保存至 data/processed/final.csv')