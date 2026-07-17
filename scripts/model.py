"""基线翻译模型

PyTorch nn.Transformer 的 encoder-decoder。参数按数据规模取小值防过拟合：
d_model=128（对齐 Word2Vec 维度以便词向量初始化）、nhead=4、
encoder/decoder 各 2 层、FFN=512、dropout=0.1，共 0.99M 参数。
"""
import math

import torch
import torch.nn as nn

from vocab import PAD, BOS, EOS


class PositionalEncoding(nn.Module):
    """正弦位置编码（Vaswani et al. 2017）"""

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4,
                 num_layers=2, dim_ff=512, dropout=0.1, max_len=200):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def _embed_src(self, src):
        return self.pos(self.src_emb(src) * math.sqrt(self.d_model))

    def _embed_tgt(self, tgt):
        return self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))

    def forward(self, src, tgt_in):
        """src: (B, S)  tgt_in: (B, T)  →  logits: (B, T, tgt_vocab)"""
        src_pad_mask = src == PAD
        tgt_pad_mask = tgt_in == PAD
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device)
        out = self.transformer(
            self._embed_src(src), self._embed_tgt(tgt_in),
            tgt_mask=causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask)
        return self.generator(out)

    def encode(self, src, src_pad_mask):
        return self.transformer.encoder(self._embed_src(src),
                                        src_key_padding_mask=src_pad_mask)

    def decode_step(self, ys, memory, src_pad_mask):
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            ys.size(1), device=ys.device)
        out = self.transformer.decoder(self._embed_tgt(ys), memory,
                                       tgt_mask=causal_mask,
                                       memory_key_padding_mask=src_pad_mask)
        return self.generator(out)


@torch.no_grad()
def greedy_decode(model, src, max_len=100):
    """批量贪心解码。src: (B, S) → (B, <=max_len) 生成序列（含 BOS/EOS）"""
    model.eval()
    src_pad_mask = src == PAD
    memory = model.encode(src, src_pad_mask)
    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        logits = model.decode_step(ys, memory, src_pad_mask)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        next_tok[finished] = PAD  # 已结束的句子只补 pad
        ys = torch.cat([ys, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == EOS)
        if bool(finished.all()):
            break
    return ys
