# src/model/architecture/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    vocab_size: int = 8192
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    max_seq_length: int = 256
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    feed_forward_size: int = 1536  # 4x hidden_size
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_rope: bool = True  # Whether to use RoPE for positional embedding
    tie_word_embeddings: bool = True
    bias: bool = False  # Modern transformers often work better without biases

