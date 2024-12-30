# src/model/architecture/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import warnings

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    warnings.warn("Flash Attention not available. Install with: pip install flash-attn")

class RotaryEmbedding(nn.Module):
    """Implements Rotary Position Embedding (RoPE) with caching."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Initialize inverse frequency buffer
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Initialize cache
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, dtype: torch.dtype, device: torch.device, seq_len: int):
        """Update cos and sin cache for given sequence length."""
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        
        self._cos_cached = emb.cos()[None, None, :, :]
        self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
            
        if seq_len != self._seq_len_cached:
            self._update_cos_sin_cache(q.dtype, q.device, seq_len)
            
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)
        )

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings to input tensor."""
    # Reshape for rotation
    x_rot, x_pass = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    # Apply rotation using einsum for better efficiency
    x_rot = torch.stack([x_rot[..., ::2], x_rot[..., 1::2]], dim=-1)
    x_rot = torch.stack([
        x_rot[..., 0] * cos - x_rot[..., 1] * sin,
        x_rot[..., 1] * cos + x_rot[..., 0] * sin,
    ], dim=-1).flatten(-2)
    
    return torch.cat((x_rot, x_pass), dim=-1)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional flash attention and rotary embeddings."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_dropout_prob
        self.scale = self.head_dim ** -0.5
        
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        
        # Linear layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        
        # Initialize rotary embeddings if enabled
        self.use_rope = config.use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_seq_len=config.max_position_embeddings
            )
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Flash attention config
        self.use_flash_attn = FLASH_ATTENTION_AVAILABLE and not config.get('disable_flash_attn', False)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split heads and transpose for attention computation."""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
        
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads and transpose back."""
        x = x.permute(0, 2, 1, 3)  # (batch, seq_len, heads, head_dim)
        new_shape = x.size()[:-2] + (self.hidden_size,)
        return x.reshape(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self._split_heads(self.q_proj(hidden_states))
        key_states = self._split_heads(self.k_proj(hidden_states))
        value_states = self._split_heads(self.v_proj(hidden_states))
        
        # Apply rotary embeddings if enabled
        if self.use_rope:
            query_states, key_states = self.rotary_emb(
                query_states, key_states, seq_length
            )
        
        # Handle cached key/value states for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        if self.use_flash_attn and attention_mask is None:
            # Use flash attention when possible
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),  # (batch, seq_len, heads, head_dim)
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
            attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        else:
            # Traditional attention with masking
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
            attention_scores = attention_scores * self.scale
            
            # Apply causal mask if no explicit mask provided
            if attention_mask is None:
                attention_mask = torch.triu(
                    torch.ones(seq_length, seq_length, device=hidden_states.device),
                    diagonal=1
                ).bool()
                attention_scores.masked_fill_(attention_mask[None, None, :, :], float('-inf'))
            else:
                attention_scores = attention_scores + attention_mask
            
            # Compute attention probabilities
            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attention_probs = self.attn_dropout(attention_probs)
            
            # Compute attention output
            attn_output = torch.matmul(attention_probs, value_states)
            attn_output = self._merge_heads(attn_output)
        
        # Final projection and dropout
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, past_key_value
