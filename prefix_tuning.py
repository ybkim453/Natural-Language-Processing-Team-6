'''
Prefix Tuning for GPT-2 sentiment classification.

GPT-2 backbone은 완전히 frozen하고, 각 attention 레이어의 K/V 앞에
학습 가능한 prefix 벡터를 prepend해서 학습한다.

실행:
  python prefix_tuning.py --use_gpu --prefix_length 20
'''

import csv
import json
import math
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from types import SimpleNamespace
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gpt2 import GPT2Model
from utils import get_extended_attention_mask
from classifier import SentimentDataset, load_data, model_eval, save_model
from optimizer import AdamW

TQDM_DISABLE = False
RESULTS_DIR = 'predictions'


class PrefixTuningGPT2(nn.Module):
    '''
    GPT2Model을 wrapping하여 각 attention 레이어에 학습 가능한 prefix K/V를 주입한다.
    backbone의 모든 파라미터는 frozen 상태를 유지한다.
    '''
    def __init__(self, base_model: GPT2Model, prefix_length: int = 20):
        super().__init__()
        self.gpt = base_model
        for p in self.gpt.parameters():
            p.requires_grad_(False)

        config = base_model.config
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        head_dim = config.hidden_size // n_heads

        self.prefix_length = prefix_length
        # 각 레이어별로 독립적인 prefix K/V: (n_layers, n_heads, prefix_length, head_dim)
        self.prefix_keys = nn.Parameter(
            torch.randn(n_layers, n_heads, prefix_length, head_dim) * 0.01
        )
        self.prefix_values = nn.Parameter(
            torch.randn(n_layers, n_heads, prefix_length, head_dim) * 0.01
        )

    def forward(self, input_ids, attention_mask):
        hidden_states = self.gpt.embed(input_ids=input_ids)
        extended_mask = get_extended_attention_mask(attention_mask, hidden_states.dtype)

        for i, layer in enumerate(self.gpt.gpt_layers):
            hidden_states = self._layer_forward(layer, hidden_states, extended_mask, i)

        hidden_states = self.gpt.final_layer_norm(hidden_states)
        last_non_pad_idx = attention_mask.sum(dim=1) - 1
        last_token = hidden_states[torch.arange(hidden_states.shape[0]), last_non_pad_idx]
        return {'last_hidden_state': hidden_states, 'last_token': last_token}

    def _layer_forward(self, layer, hidden_states, attention_mask, layer_idx):
        attn = layer.self_attention
        B, T, _ = hidden_states.shape
        P = self.prefix_length

        normed = layer.attention_layer_norm(hidden_states)

        # Q: (B, H, T, D) / K, V: prefix concat 후 (B, H, P+T, D)
        query = attn.transform(normed, attn.query)
        key   = attn.transform(normed, attn.key)
        value = attn.transform(normed, attn.value)

        prefix_k = self.prefix_keys[layer_idx].unsqueeze(0).expand(B, -1, -1, -1)
        prefix_v = self.prefix_values[layer_idx].unsqueeze(0).expand(B, -1, -1, -1)
        key   = torch.cat([prefix_k, key],   dim=2)  # (B, H, P+T, D)
        value = torch.cat([prefix_v, value], dim=2)

        # attention scores: (B, H, T, P+T)
        scores = query @ key.transpose(-2, -1) / math.sqrt(attn.attention_head_size)

        # causal mask: prefix 위치는 항상 visible, 토큰 간에만 upper-triangular 적용
        token_causal = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
        prefix_causal = torch.zeros(T, P, device=scores.device, dtype=torch.bool)
        causal_mask = torch.cat([prefix_causal, token_causal], dim=1)  # (T, P+T)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -10000.0)

        # padding mask: prefix 위치는 0(visible), 토큰 위치는 기존 extended_mask 사용
        prefix_bias = torch.zeros(B, 1, 1, P, device=scores.device, dtype=scores.dtype)
        full_mask = torch.cat([prefix_bias, attention_mask], dim=-1)  # (B, 1, 1, P+T)
        scores = scores + full_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn.dropout(attn_weights)
        context = rearrange(attn_weights @ value, 'b h t d -> b t (h d)')

        hidden_states = layer.add(hidden_states, context, layer.attention_dense, layer.attention_dropout)
        ffn_out = layer.interm_af(layer.interm_dense(layer.out_layer_norm(hidden_states)))
        hidden_states = layer.add(hidden_states, ffn_out, layer.out_dense, layer.out_dropout)
        return hidden_states

    def hidden_state_to_token(self, hidden_state):
        return self.gpt.hidden_state_to_token(hidden_state)


class PrefixSentimentClassifier(nn.Module):
    def __init__(self, config, prefix_length: int = 20):
        super().__init__()
        self.num_labels = config.num_labels
        base_model = GPT2Model.from_pretrained()
        self.prefix_gpt = PrefixTuningGPT2(base_model, prefix_length)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.prefix_gpt(input_ids, attention_mask)
        last_token = self.dropout(output['last_token'])
        return self.classifier(last_token)


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                            collate_fn=dev_dataset.collate_fn)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        data_dir='.',
    )
    model = PrefixSentimentClassifier(config, prefix_length=args.prefix_length)
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = count_trainable_params(model)
    print(f"전체 파라미터: {total:,} | 학습 파라미터: {trainable:,} ({100*trainable/total:.2f}%)")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    best_dev_acc = 0
    epoch_logs = []

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.filepath))[0]
    log_path    = os.path.join(RESULTS_DIR, f'{stem}_log.csv')
    result_path = os.path.join(RESULTS_DIR, f'{stem}_result.json')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_loader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        dev_acc, dev_f1, *_ = model_eval(dev_loader, model, device)
        epoch_logs.append({'epoch': epoch, 'train_loss': round(train_loss, 4),
                           'dev_acc': round(dev_acc, 4), 'dev_f1': round(dev_f1, 4)})

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: loss={train_loss:.3f}, dev_acc={dev_acc:.3f}")

    # 에포크별 학습 로그 CSV
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'dev_acc', 'dev_f1'])
        writer.writeheader()
        writer.writerows(epoch_logs)

    # 최종 요약 JSON
    result = {
        'method': 'prefix_tuning',
        'prefix_length': args.prefix_length,
        'total_params': total,
        'trainable_params': trainable,
        'trainable_ratio_pct': round(100 * trainable / total, 4),
        'best_dev_acc': round(best_dev_acc, 4),
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
    }
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nBest dev acc: {best_dev_acc:.3f}")
    print(f"학습 로그 저장: {log_path}")
    print(f"결과 요약 저장: {result_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--prefix_length", type=int, default=20, help="prefix 토큰 수")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.train = 'data/ids-sst-train.csv'
    args.dev = 'data/ids-sst-dev.csv'
    args.filepath = f'prefix-len{args.prefix_length}-sst.pt'
    seed_everything(args.seed)
    train(args)
