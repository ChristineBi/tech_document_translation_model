import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('data/processed/final.csv', encoding='utf-8-sig')
os.makedirs('data/splits', exist_ok=True)

# 按topic分层，保证每个领域在三个集合里都有覆盖
train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['topic'])
val, test   = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['topic'])

train.to_csv('data/splits/train.csv', index=False, encoding='utf-8-sig')
val.to_csv('data/splits/val.csv',   index=False, encoding='utf-8-sig')
test.to_csv('data/splits/test.csv',  index=False, encoding='utf-8-sig')

print(f'训练集: {len(train)} 条 ({len(train)/len(df)*100:.1f}%)')
print(f'验证集: {len(val)} 条 ({len(val)/len(df)*100:.1f}%)')
print(f'测试集: {len(test)} 条 ({len(test)/len(df)*100:.1f}%)')
print(f'\n各集合topic分布:')
for name, split in [('train', train), ('val', val), ('test', test)]:
    print(f'\n{name}:')
    print(split['topic'].value_counts().to_string())