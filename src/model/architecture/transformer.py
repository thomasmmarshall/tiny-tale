# src/model/architecture/transformer.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class TransformerConfig:
    """Configuration for a compact modern decoder-only language model."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int

    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    use_rope: bool = True
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    bias: bool = False
    tie_word_embeddings: bool = True
    num_key_value_heads: Optional[int] = None
    use_sdpa: bool = True

    config_version: str = "2.0.0"

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                f"intermediate_size {self.intermediate_size} should be larger than "
                f"hidden_size {self.hidden_size}"
            )

        if not 0 <= self.hidden_dropout_prob <= 1:
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1, got {self.hidden_dropout_prob}")

        if not 0 <= self.attention_dropout_prob <= 1:
            raise ValueError(f"attention_dropout_prob must be between 0 and 1, got {self.attention_dropout_prob}")

    def save(self, save_path: str):
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_json(cls, config_path: str) -> "TransformerConfig":
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        loaded_version = config_dict.pop("config_version", "0.0.0")
        if loaded_version != cls.config_version:
            print(
                f"Warning: Loading config with version {loaded_version}, "
                f"current version is {cls.config_version}"
            )
        return cls(**config_dict)

    def get_num_parameters(self) -> int:
        embedding_params = self.vocab_size * self.hidden_size
        position_params = 0 if self.use_rope else self.max_position_embeddings * self.hidden_size
        head_dim = self.hidden_size // self.num_attention_heads
        kv_size = self.num_key_value_heads * head_dim

        attention_params = (
            self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * kv_size
            + self.hidden_size * self.hidden_size
        )
        if self.bias:
            attention_params += self.hidden_size + 2 * kv_size + self.hidden_size

        swiglu_params = 3 * self.hidden_size * self.intermediate_size
        if self.bias:
            swiglu_params += 2 * self.intermediate_size + self.hidden_size

        norm_params = 2 * self.hidden_size
        layer_params = attention_params + swiglu_params + norm_params
        output_params = 0 if self.tie_word_embeddings else self.hidden_size * self.vocab_size
        return embedding_params + position_params + output_params + layer_params * self.num_hidden_layers


class RMSNorm(nn.Module):
    """RMSNorm is cheaper than LayerNorm and standard in modern small LMs."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, seq_length: int, device: torch.device, dtype: torch.dtype):
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum {self.max_position_embeddings}"
            )
        positions = torch.arange(seq_length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, heads * n_rep, seq_len, head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.use_rope = config.use_rope
        self.use_sdpa = config.use_sdpa
        self.attention_dropout_prob = config.attention_dropout_prob

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

    def _shape(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states), self.num_heads)
        key_states = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value_states = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        if self.use_rope:
            cos, sin = self.rotary_emb(seq_length, hidden_states.device, query_states.dtype)
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = torch.ones(
            (seq_length, seq_length),
            device=hidden_states.device,
            dtype=torch.bool,
        ).tril()
        attn_mask = attention_mask[:, None, None, :].bool() & causal_mask[None, None, :, :]

        dropout_p = self.attention_dropout_prob if self.training else 0.0
        if self.use_sdpa:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
            )
        else:
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
            attention_scores = attention_scores * (self.head_dim ** -0.5)
            attention_scores = attention_scores.masked_fill(~attn_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attention_probs = F.dropout(attention_probs, p=dropout_p, training=self.training)
            attn_output = torch.matmul(attention_probs, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.resid_dropout(self.o_proj(attn_output))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.dropout(self.down_proj(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.input_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.input_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.feed_forward(self.post_attention_norm(hidden_states))
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = None
        if not config.use_rope:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.word_embeddings

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        if seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum {self.config.max_position_embeddings}"
            )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        attention_mask = attention_mask.to(device=input_ids.device)

        hidden_states = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            hidden_states = hidden_states + self.position_embeddings(position_ids)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, attention_mask, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        if self.lm_head is None:
            logits = F.linear(hidden_states, self.word_embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)

        if labels is None:
            return logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype