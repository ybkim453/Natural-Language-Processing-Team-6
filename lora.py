'''
LoRA (Low-Rank Adaptation) for GPT-2 sentiment classification.

실행:
  python lora.py --use_gpu --lora_r 8 --lora_alpha 16
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
from types import SimpleNamespace
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gpt2 import GPT2Model
from classifier import SentimentDataset, load_data, model_eval, save_model
from optimizer import AdamW

TQDM_DISABLE = False
RESULTS_DIR = 'predictions'


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.scaling = alpha / r
        d_in, d_out = linear.in_features, linear.out_features

        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora(model: GPT2Model, r: int = 8, alpha: float = 16.0) -> GPT2Model:
    '''모든 파라미터를 freeze하고 각 GPT 레이어의 query/value에 LoRA를 삽입한다.'''
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in model.gpt_layers:
        attn = layer.self_attention
        attn.query = LoRALinear(attn.query, r, alpha)
        attn.value = LoRALinear(attn.value, r, alpha)
    return model


class LoRASentimentClassifier(nn.Module):
    def __init__(self, config, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2Model.from_pretrained()
        apply_lora(self.gpt, r=r, alpha=alpha)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.gpt(input_ids, attention_mask)
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
    model = LoRASentimentClassifier(config, r=args.lora_r, alpha=args.lora_alpha)
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
        'method': 'lora',
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
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
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling alpha")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.train = 'data/ids-sst-train.csv'
    args.dev = 'data/ids-sst-dev.csv'
    args.filepath = f'lora-r{args.lora_r}-sst.pt'
    seed_everything(args.seed)
    train(args)
