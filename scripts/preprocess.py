import pandas as pd
import re
import os
import json
import time
import collections

# ── 路径 ──────────────────────────────────────────
INPUT  = 'data/raw/tech_document_translation_data.csv'
OUTPUT = 'data/processed/cleaned.csv'
os.makedirs('data/processed', exist_ok=True)

# ── 读取 ──────────────────────────────────────────
df = pd.read_csv(INPUT, encoding='utf-8-sig')
print(f'原始数据: {len(df)} 条')

# ── 去重 ───────────────────────────────────
df = df.drop_duplicates(subset=['chinese', 'japanese', 'indonesian'])
print(f'去重后: {len(df)} 条')

# ── 去除空值 ───────────────────────────────
df = df.dropna(subset=['chinese', 'japanese', 'indonesian'])
df = df[df['chinese'].str.strip() != '']
df = df[df['japanese'].str.strip() != '']
df = df[df['indonesian'].str.strip() != '']
print(f'去除空值后: {len(df)} 条')

# ── 混语标记─────────────
def chinese_char_ratio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(zh_chars) / len(text)

df['zh_ratio_in_ja'] = df['japanese'].apply(chinese_char_ratio)
df['zh_ratio_in_id'] = df['indonesian'].apply(chinese_char_ratio)
df['has_mixed_lang'] = (df['zh_ratio_in_ja'] > 0.05) | (df['zh_ratio_in_id'] > 0.05)

mixed_count = df['has_mixed_lang'].sum()
print(f'混语样本数: {mixed_count} ({mixed_count/len(df)*100:.1f}%)')

# ── 保存 ──────────────────────────────────────────
df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f'\n已保存至 {OUTPUT}')

# 提取印尼语里所有中文词汇（长度2-10），统计频率
def extract_chinese_terms(text):
    return re.findall(r'[\u4e00-\u9fff]{2,10}', text)

id_terms = collections.Counter()
for text in df['indonesian']:
    id_terms.update(extract_chinese_terms(text))

# 所有出现过的术语列表（按频率排序）
all_terms = [term for term, count in id_terms.most_common()]
print(f'共发现 {len(all_terms)} 个独立术语')
print(all_terms[:10])
