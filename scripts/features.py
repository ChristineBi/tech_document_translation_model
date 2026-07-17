"""特征工程，仅用训练集拟合以避免数据泄漏。

产出四类特征:
1) TF-IDF 术语特征（含术语表标记）→ data/features/term_features.csv
2) Word2Vec 词向量（dim=128，对齐 d_model 可初始化 embedding）→ w2v_{lang}.kv
3) 模板骨架（术语替换为 <T>，量化句式多样性）→ template_features.csv
4) 上下文特征: topic 前缀 token（见 vocab.py），训练时拼在源句句首

skeleton() 供复用: from features import skeleton
"""
import os

import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from vocab import GLOSSARY_TERMS, LANG_COL, TOKENIZERS, tokenize_zh

FEATURE_DIR = 'data/features'
PUNCT = set('，。、；：？！（）“”,.;:?!()')

# ── 句式结构特征：模板骨架（模块级，供复用）──────
_terms_by_len = sorted(GLOSSARY_TERMS, key=len, reverse=True)


def skeleton(text):
    """把句中术语替换为 <T>，得到句式骨架（长词优先，避免短词打断长词）"""
    for t in _terms_by_len:
        text = text.replace(t, '<T>')
    return text


def main():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    train = pd.read_csv('data/splits/train.csv', encoding='utf-8-sig')
    print(f'训练集: {len(train)} 条')
    print()

    # ── 1. TF-IDF 术语特征 ────────────────────────
    vectorizer = TfidfVectorizer(tokenizer=tokenize_zh, token_pattern=None, lowercase=False)
    tfidf = vectorizer.fit_transform(train['chinese'])
    vocab_terms = vectorizer.get_feature_names_out()

    # 术语特征表：文档频次、平均TF-IDF、是否在术语表中
    tfidf_mean = tfidf.mean(axis=0).A1
    doc_freq = (tfidf > 0).sum(axis=0).A1
    term_rows = []
    for i, term in enumerate(vocab_terms):
        if len(term) < 2 or all(ch in PUNCT for ch in term):
            continue  # 过滤标点和单字虚词，保留候选术语
        term_rows.append({
            'term': term,
            'doc_freq': int(doc_freq[i]),
            'mean_tfidf': round(float(tfidf_mean[i]), 5),
            'in_glossary': term in GLOSSARY_TERMS,
        })
    term_df = pd.DataFrame(term_rows).sort_values('mean_tfidf', ascending=False)
    term_df.to_csv(f'{FEATURE_DIR}/term_features.csv', index=False, encoding='utf-8-sig')

    n_gloss = term_df['in_glossary'].sum()
    print(f'[TF-IDF] 候选术语 {len(term_df)} 个，其中 {n_gloss} 个在第一周术语表中')
    print('全局 Top 10（按平均TF-IDF）:')
    print(term_df.head(10).to_string(index=False))
    print()

    # 按 topic 提取关键术语（该 topic 内平均TF-IDF最高的词）
    term_index = {t: i for i, t in enumerate(vocab_terms)}
    keep_idx = [term_index[t] for t in term_df['term']]
    print('各 topic 关键术语 Top 5:')
    for topic, group_idx in train.groupby('topic').groups.items():
        topic_mean = tfidf[list(group_idx)][:, keep_idx].mean(axis=0).A1
        top5 = term_df['term'].iloc[topic_mean.argsort()[::-1][:5]].tolist()
        print(f'  {topic}: {top5}')
    print()

    # ── 2. Word2Vec 语义向量 ──────────────────────
    # skip-gram + 固定 seed + 单线程，保证结果可复现
    for lang, col in LANG_COL.items():
        token_lists = [TOKENIZERS[lang](s) for s in train[col]]
        w2v = Word2Vec(token_lists, vector_size=128, window=5, min_count=1,
                       sg=1, epochs=100, seed=42, workers=1)
        w2v.wv.save(f'{FEATURE_DIR}/w2v_{lang}.kv')
        print(f'[Word2Vec] {lang}: {len(w2v.wv)} 个向量 (dim=128) → w2v_{lang}.kv')

    # 近邻抽查（ja 字符级，不做词义抽查）
    probes = {'zh': ['故障恢复', '快照', '资源调度'], 'id': ['snapshot', 'backup', 'node']}
    for lang, words in probes.items():
        wv = KeyedVectors.load(f'{FEATURE_DIR}/w2v_{lang}.kv')
        for w in words:
            if w in wv:
                neighbors = [f'{t}({s:.2f})' for t, s in wv.most_similar(w, topn=3)]
                print(f'  {lang} "{w}" 近邻: {neighbors}')
    print()

    # ── 3. 句式结构特征：模板骨架统计 ─────────────
    train['skeleton'] = train['chinese'].apply(skeleton)
    skel_counts = train['skeleton'].value_counts()
    skel_df = skel_counts.rename_axis('skeleton').reset_index(name='count')
    skel_df.to_csv(f'{FEATURE_DIR}/template_features.csv', index=False, encoding='utf-8-sig')

    print(f'[句式结构] 训练集 {len(train)} 句 → 仅 {len(skel_counts)} 种模板骨架 '
          f'(多样性 {len(skel_counts)/len(train)*100:.1f}%)')
    print('最高频模板 Top 5:')
    for skel, cnt in skel_counts.head(5).items():
        print(f'  [{cnt:4d}次] {skel}')
    print()
    print('各 topic 模板数（对照第一周重复率）:')
    print(train.groupby('topic')['skeleton'].nunique().sort_values().to_string())


if __name__ == '__main__':
    main()
