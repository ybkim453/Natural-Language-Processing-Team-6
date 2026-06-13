import math

import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn


class LoRALinear(nn.Module):
  def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.0):
    super().__init__()
    if rank <= 0:
      raise ValueError("LoRA rank must be positive.")

    self.linear = linear_layer
    self.rank = rank
    self.alpha = alpha
    self.scaling = alpha / rank
    self.lora_dropout = nn.Dropout(dropout)
    self.lora_A = nn.Linear(linear_layer.in_features, rank, bias=False)
    self.lora_B = nn.Linear(rank, linear_layer.out_features, bias=False)

    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)

  def forward(self, x):
    lora_update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
    return self.linear(x) + lora_update

  def lora_parameters(self):
    yield self.lora_A.weight
    yield self.lora_B.weight

  def mark_only_lora_as_trainable(self):
    for param in self.linear.parameters():
      param.requires_grad = False
    for param in self.lora_parameters():
      param.requires_grad = True


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def enable_lora(self, rank=8, alpha=16, dropout=0.0):
    self.query = self._wrap_with_lora(self.query, rank, alpha, dropout)
    self.value = self._wrap_with_lora(self.value, rank, alpha, dropout)

  def _wrap_with_lora(self, linear_layer, rank, alpha, dropout):
    if isinstance(linear_layer, LoRALinear):
      return linear_layer
    return LoRALinear(linear_layer, rank=rank, alpha=alpha, dropout=dropout)

  def mark_only_lora_as_trainable(self):
    for linear_layer in (self.query, self.value):
      if isinstance(linear_layer, LoRALinear):
        linear_layer.mark_only_lora_as_trainable()

  def transform(self, x, linear_layer):
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    proj = linear_layer(x)
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    T = query.size(2)

    scores = query @ key.transpose(-2, -1) / math.sqrt(self.attention_head_size)

    causal_mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, -10000.0)

    scores = scores + attention_mask

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)

    context = attn_weights @ value
    return rearrange(context, 'b h t d -> b t (h d)')


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
