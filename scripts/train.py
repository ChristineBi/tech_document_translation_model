"""基线模型训练

用法: python3 scripts/train.py --tgt ja  （或 --tgt id）
训练用 teacher forcing + label smoothing，每轮监控 train/val loss 并早停。
产出: models/baseline_{tgt}.pt、logs/train_log_{tgt}.csv、logs/train_curve_{tgt}.png
"""
import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import Seq2SeqTransformer, greedy_decode
from vocab import LANG_COL, PAD, TOKENIZERS, Vocab

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ── 数据 ──────────────────────────────────────────
class PairDataset(Dataset):
    """源端 = <topic_xxx> 前缀 + 中文分词；目标端 = 目标语言分词"""

    def __init__(self, df, src_vocab, tgt_vocab, tgt_lang):
        tok_zh, tok_tgt = TOKENIZERS['zh'], TOKENIZERS[tgt_lang]
        self.pairs = []
        for _, row in df.iterrows():
            src_tokens = [f'<topic_{row["topic"]}>'] + tok_zh(row['chinese'])
            tgt_tokens = tok_tgt(row[LANG_COL[tgt_lang]])
            self.pairs.append((torch.tensor(src_vocab.encode(src_tokens)),
                               torch.tensor(tgt_vocab.encode(tgt_tokens))))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def collate(batch):
    src, tgt = zip(*batch)
    return (pad_sequence(src, batch_first=True, padding_value=PAD),
            pad_sequence(tgt, batch_first=True, padding_value=PAD))


def init_embedding_from_w2v(embedding, vocab, kv_path):
    """用第2步训练的 Word2Vec 向量初始化 embedding（特殊token保持随机）"""
    kv = KeyedVectors.load(kv_path)
    hit = 0
    with torch.no_grad():
        for i, tok in enumerate(vocab.itos):
            if tok in kv:
                embedding.weight[i] = torch.from_numpy(kv[tok].copy())
                hit += 1
    return hit, len(vocab)


# ── 评估 ──────────────────────────────────────────
@torch.no_grad()
def evaluate_loss(model, loader, criterion):
    model.eval()
    total, count = 0.0, 0
    for src, tgt in loader:
        logits = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        total += loss.item() * src.size(0)
        count += src.size(0)
    return total / count


@torch.no_grad()
def translate_corpus(model, loader, tgt_vocab, tgt_lang):
    """批量贪心解码，返回字符串列表"""
    joiner = '' if tgt_lang == 'ja' else ' '
    hyps = []
    for src, _ in loader:
        out = greedy_decode(model, src, max_len=100)
        for row in out:
            hyps.append(joiner.join(tgt_vocab.decode(row.tolist())))
    return hyps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt', choices=['ja', 'id'], default='ja')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=8, help='早停轮数')
    parser.add_argument('--no-w2v', action='store_true', help='不用Word2Vec初始化embedding')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs('models', exist_ok=True)

    # 数据与词表
    src_vocab = Vocab.load('data/vocab/vocab_zh.json')
    tgt_vocab = Vocab.load(f'data/vocab/vocab_{args.tgt}.json')
    train_df = pd.read_csv('data/splits/train.csv', encoding='utf-8-sig')
    val_df = pd.read_csv('data/splits/val.csv', encoding='utf-8-sig')

    train_loader = DataLoader(PairDataset(train_df, src_vocab, tgt_vocab, args.tgt),
                              batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_df, src_vocab, tgt_vocab, args.tgt),
                            batch_size=128, shuffle=False, collate_fn=collate)

    # 模型（参数配置见 model.py 说明）
    model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'方向: zh→{args.tgt}  模型参数量: {n_params/1e6:.2f}M  设备: CPU')

    if not args.no_w2v:
        hit_s, n_s = init_embedding_from_w2v(model.src_emb, src_vocab, 'data/features/w2v_zh.kv')
        hit_t, n_t = init_embedding_from_w2v(model.tgt_emb, tgt_vocab, f'data/features/w2v_{args.tgt}.kv')
        print(f'Word2Vec 初始化: 源端 {hit_s}/{n_s}，目标端 {hit_t}/{n_t}（其余为随机初始化）')

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 训练循环：监控 loss，早停防过拟合 ──
    history = []
    best_val, best_state, bad_epochs = float('inf'), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, count = 0.0, 0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防梯度爆炸
            optimizer.step()
            total += loss.item() * src.size(0)
            count += src.size(0)
        train_loss = total / count
        val_loss = evaluate_loss(model, val_loader, criterion)
        history.append({'epoch': epoch, 'train_loss': round(train_loss, 4),
                        'val_loss': round(val_loss, 4)})
        marker = ''
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = ' *'
        else:
            bad_epochs += 1
        print(f'epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  '
              f'({time.time()-t0:.1f}s){marker}')
        if bad_epochs >= args.patience:
            print(f'val loss 连续 {args.patience} 轮未下降，早停')
            break

    # ── 恢复最优权重，保存 ──
    model.load_state_dict(best_state)
    torch.save({'state_dict': best_state, 'tgt': args.tgt,
                'val_loss': best_val, 'args': vars(args)},
               f'models/baseline_{args.tgt}.pt')
    pd.DataFrame(history).to_csv(f'logs/train_log_{args.tgt}.csv', index=False)

    # loss 曲线
    hist_df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df['epoch'], hist_df['train_loss'], color='steelblue', label='train')
    plt.plot(hist_df['epoch'], hist_df['val_loss'], color='indianred', label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'zh→{args.tgt} 基线模型训练曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'logs/train_curve_{args.tgt}.png', dpi=120)

    # ── val 集 BLEU + 翻译样例 ──
    refs = val_df[LANG_COL[args.tgt]].tolist()
    hyps = translate_corpus(model, val_loader, tgt_vocab, args.tgt)
    if args.tgt == 'ja':
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize='char')
        note = '字符级BLEU'
    else:
        bleu = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
        note = '词级BLEU(忽略大小写)'
    print()
    print(f'最优 val loss: {best_val:.4f}   val BLEU: {bleu.score:.2f} ({note})')

    print()
    print('翻译样例（val 前3条）:')
    for i in range(3):
        print(f'  源: {val_df["chinese"].iloc[i]}')
        print(f'  参: {refs[i]}')
        print(f'  译: {hyps[i]}')
        print()


if __name__ == '__main__':
    main()
