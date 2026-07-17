"""分词与词表构建

分词方案: 中文 jieba 词级（术语整词）、日语字符级、印尼语小写+空格。
词表只用训练集构建（避免数据泄漏），未登录词回退 <unk>。
可作模块复用: from vocab import Vocab, TOKENIZERS
"""
import collections
import json
import os

import jieba
import pandas as pd

# ── 常量 ──────────────────────────────────────────
SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']
PAD, BOS, EOS, UNK = 0, 1, 2, 3

LANG_COL = {'zh': 'chinese', 'ja': 'japanese', 'id': 'indonesian'}
VOCAB_DIR = 'data/vocab'
GLOSSARY_PATH = 'data/glossary_reviewed.json'


# ── 分词器 ────────────────────────────────────────
# 术语注入 jieba 用户词典，让多词术语（如"故障恢复"）保持整词
def _load_glossary_terms():
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH, encoding='utf-8') as f:
            terms = list(json.load(f))
        for term in terms:
            jieba.add_word(term)
        return terms
    print(f'警告: 未找到术语表 {GLOSSARY_PATH}，中文分词不含术语整词')
    return []


GLOSSARY_TERMS = _load_glossary_terms()


def tokenize_zh(text):
    return jieba.lcut(text)


def tokenize_ja(text):
    return list(text)


def tokenize_id(text):
    return text.lower().split()


TOKENIZERS = {'zh': tokenize_zh, 'ja': tokenize_ja, 'id': tokenize_id}


# ── 词表 ──────────────────────────────────────────
class Vocab:
    """token <-> id 映射；encode 自动加 <bos>/<eos>，未登录词回退 <unk>"""

    def __init__(self, itos):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, extra_tokens=()):
        counter = collections.Counter()
        for toks in token_lists:
            counter.update(toks)
        # sorted 保证词表构建可复现
        return cls(SPECIALS + list(extra_tokens) + sorted(counter))

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=1)

    def encode(self, tokens):
        return [BOS] + [self.stoi.get(t, UNK) for t in tokens] + [EOS]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i not in (PAD, BOS, EOS)]

    def __len__(self):
        return len(self.itos)


# ── 构建入口 ──────────────────────────────────────
def main():
    os.makedirs(VOCAB_DIR, exist_ok=True)
    splits = {name: pd.read_csv(f'data/splits/{name}.csv', encoding='utf-8-sig')
              for name in ['train', 'val', 'test']}
    train = splits['train']

    # topic 前缀 token（上下文特征），先入中文词表
    topic_tokens = [f'<topic_{t}>' for t in sorted(train['topic'].unique())]

    vocabs = {}
    for lang, col in LANG_COL.items():
        token_lists = [TOKENIZERS[lang](s) for s in train[col]]
        extra = topic_tokens if lang == 'zh' else ()
        vocab = Vocab.build(token_lists, extra_tokens=extra)
        vocab.save(f'{VOCAB_DIR}/vocab_{lang}.json')
        vocabs[lang] = vocab

        # 统计词表规模、序列长度、未登录词率
        lens = [len(toks) + 2 for toks in token_lists]
        print(f'[{lang}] 词表 {len(vocab)}（含 {len(SPECIALS) + len(extra)} 个特殊token） '
              f'序列长度 avg={sum(lens)/len(lens):.1f} max={max(lens)}')
        for name in ['val', 'test']:
            total = unk = 0
            for s in splits[name][col]:
                ids = vocab.encode(TOKENIZERS[lang](s))
                total += len(ids)
                unk += ids.count(UNK)
            print(f'     {name} 未登录词率: {unk}/{total} ({unk/total*100:.2f}%)')

    # 编码-解码往返自检
    sample = train['chinese'].iloc[0]
    ids = vocabs['zh'].encode(tokenize_zh(sample))
    restored = ''.join(vocabs['zh'].decode(ids))
    print()
    print(f'往返自检(zh): {"通过" if restored == sample else "失败"}')
    print(f'  原句: {sample}')
    print(f'  编码: {ids}')


if __name__ == '__main__':
    main()
