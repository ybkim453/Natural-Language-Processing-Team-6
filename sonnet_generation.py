'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from utils import format_parameter_count_table, parameter_count_rows

TQDM_DISABLE = False
TRAINABLE_GPT_LAYERS = 4
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-3
GENERATION_TOP_K = 50
REPETITION_PENALTY = 1.1
NUM_QUALITATIVE_SAMPLES = 3


def get_torch_device(torch_module, use_gpu):
  if not use_gpu:
    return torch_module.device("cpu")

  if torch_module.cuda.is_available():
    return torch_module.device("cuda")

  mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
  if mps_backend is not None and mps_backend.is_available():
    return torch_module.device("mps")

  return torch_module.device("cpu")


# 재현성을 위한 random seed 고정
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    fine_tune_mode = getattr(args, 'fine_tune_mode', 'partial-model')
    if fine_tune_mode == 'full-model':
      for param in self.gpt.parameters():
        param.requires_grad = True
    elif fine_tune_mode == 'partial-model':
      # 작은 소넷 데이터셋에서는 하위 레이어를 고정해 과적합과 GPU 메모리 사용을 줄인다.
      for param in self.gpt.parameters():
        param.requires_grad = False

      trainable_layers = min(TRAINABLE_GPT_LAYERS, len(self.gpt.gpt_layers))
      for layer in self.gpt.gpt_layers[-trainable_layers:]:
        for param in layer.parameters():
          param.requires_grad = True

      for param in self.gpt.final_layer_norm.parameters():
        param.requires_grad = True
    elif fine_tune_mode == 'lora':
      self.gpt.enable_lora(
        rank=getattr(args, 'lora_rank', 8),
        alpha=getattr(args, 'lora_alpha', 16),
        dropout=getattr(args, 'lora_dropout', 0.05),
      )
      self.gpt.mark_only_lora_as_trainable()
    else:
      raise ValueError(f'Unsupported fine_tune_mode: {fine_tune_mode}')

  def forward(self, input_ids, attention_mask):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    output = self.gpt(input_ids, attention_mask)
    return self.gpt.hidden_state_to_token(output['last_hidden_state'])


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=180):
    """
    temperature, top-k, top-p sampling을 함께 사용하여 소넷을 생성한다.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :].clone()

      # 반복을 줄이고, top-k/top-p sampling으로 다음 토큰 후보를 제한
      logits_last_token = self._apply_repetition_penalty(logits_last_token, token_ids)
      logits_last_token = logits_last_token / max(temperature, 1e-5)
      logits_last_token = self._filter_logits(logits_last_token, top_k=GENERATION_TOP_K, top_p=top_p)

      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)
      sampled_token = torch.multinomial(probs, 1)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().tolist(), skip_special_tokens=True)
    return token_ids, generated_output

  def _apply_repetition_penalty(self, logits, token_ids):
    if REPETITION_PENALTY == 1.0:
      return logits

    for batch_idx in range(logits.shape[0]):
      used_token_ids = set(token_ids[batch_idx].tolist())
      for token_id in used_token_ids:
        if logits[batch_idx, token_id] < 0:
          logits[batch_idx, token_id] *= REPETITION_PENALTY
        else:
          logits[batch_idx, token_id] /= REPETITION_PENALTY
    return logits

  def _filter_logits(self, logits, top_k=0, top_p=1.0):
    if top_k > 0:
      top_k = min(top_k, logits.size(-1))
      kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
      logits = logits.masked_fill(logits < kth_values, -float('inf'))

    if top_p < 1.0:
      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

      sorted_indices_to_remove = cumulative_probs > top_p
      sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = False

      indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
      logits = logits.masked_fill(indices_to_remove, -float('inf'))

    return logits


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def save_sonnet_result(best_dev_loss, result_path='predictions/sonnet_result.json'):
  import json
  import math
  import os

  loss_value = float(best_dev_loss)
  result = {
    'method': 'sonnet_generation',
    'metric': 'dev_loss',
    'lower_is_better': True,
    'best_dev_loss': round(loss_value, 4) if math.isfinite(loss_value) else None,
  }

  os.makedirs(os.path.dirname(result_path), exist_ok=True)
  with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)
  print(f"결과 저장: {result_path}")


def get_dev_sonnet_path(args, filename):
  # dev 파일이 같은 data 폴더에 있으면 검증용으로 사용
  candidate = os.path.join(os.path.dirname(args.sonnet_path), filename)
  return candidate if os.path.exists(candidate) else None


def language_modeling_loss(logits, token_ids, attention_mask):
  # padding 위치를 제외하고 다음 토큰 예측 손실만 계산
  logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
  labels = token_ids[:, 1:].contiguous().flatten()
  label_mask = attention_mask[:, 1:].contiguous().flatten().bool()
  return F.cross_entropy(logits[label_mask], labels[label_mask], reduction='mean')


@torch.no_grad()
def evaluate_sonnet_loss(model, dataloader, device):
  model.eval()
  total_loss = 0
  num_batches = 0

  for batch in tqdm(dataloader, desc='dev', disable=TQDM_DISABLE):
    b_ids, b_mask = batch['token_ids'], batch['attention_mask']
    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)

    logits = model(b_ids, b_mask)
    loss = language_modeling_loss(logits, b_ids, b_mask)
    total_loss += loss.item()
    num_batches += 1

  return total_loss / max(num_batches, 1)


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련."""
  device = get_torch_device(torch, args.use_gpu)
  print(f"Using device: {device}")
  # 데이터, 해당 데이터셋 및 데이터로드 생성
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # dev 정답 파일이 있으면 loss를 기준으로 best checkpoint를 선택
  dev_prompt_path = get_dev_sonnet_path(args, 'sonnets_held_out_dev.txt')
  dev_target_path = get_dev_sonnet_path(args, 'TRUE_sonnets_held_out_dev.txt')
  held_out_sonnet_dataset = SonnetsDataset(dev_prompt_path or args.held_out_sonnet_path)
  dev_dataloader = None
  if dev_target_path is not None:
    dev_sonnet_dataset = SonnetsDataset(dev_target_path)
    dev_dataloader = DataLoader(dev_sonnet_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_sonnet_dataset.collate_fn)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)
  print(format_parameter_count_table(parameter_count_rows(model, args.fine_tune_mode)))

  lr = args.lr
  # 선택한 fine-tuning 모드에서 freeze하지 않은 파라미터만 학습
  trainable_params = [param for param in model.parameters() if param.requires_grad]
  optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

  best_dev_loss = float('inf')
  epochs_without_improvement = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = language_modeling_loss(logits, b_ids, b_mask)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    dev_loss = evaluate_sonnet_loss(model, dev_dataloader, device) if dev_dataloader is not None else train_loss
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    for sample_idx, batch in enumerate(held_out_sonnet_dataset):
      if sample_idx >= NUM_QUALITATIVE_SAMPLES:
        break
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{output[1]}\n\n')

    # dev loss가 좋아질 때만 저장하고, 개선이 없으면 조기 종료
    if dev_loss + EARLY_STOPPING_MIN_DELTA < best_dev_loss:
      best_dev_loss = dev_loss
      epochs_without_improvement = 0
      save_model(model, optimizer, args, args.filepath)
    else:
      epochs_without_improvement += 1
      print(f"dev loss did not improve for {epochs_without_improvement} epoch(s).")

    if dev_dataloader is not None and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
      print(f"Early stopping at epoch {epoch}; best dev loss :: {best_dev_loss :.3f}.")
      break

  save_sonnet_result(best_dev_loss)


@torch.no_grad()
def generate_submission_sonnets(args):
  device = get_torch_device(torch, args.use_gpu)
  print(f"Using device: {device}")
  checkpoint_path = args.filepath if os.path.exists(args.filepath) else f'{args.epochs-1}_{args.filepath}'
  saved = torch.load(checkpoint_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    # generate가 이미 디코딩된 문자열을 반환
    decoded_output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[1]
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  parser.add_argument("--fine_tune_mode", "--fine-tune-mode", dest="fine_tune_mode", type=str,
                      choices=['full-model', 'partial-model', 'lora'], default='lora')
  parser.add_argument("--lora_rank", type=int, default=8)
  parser.add_argument("--lora_alpha", type=int, default=16)
  parser.add_argument("--lora_dropout", type=float, default=0.05)

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-{args.fine_tune_mode}-sonnet.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  generate_submission_sonnets(args)
