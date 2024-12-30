# src/model/architecture/config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TransformerConfig:
    """Configuration class for the Transformer model."""

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        max_position_embeddings: int = 256,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        layer_norm_epsilon: float = 1e-12,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        use_rope: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        tie_word_embeddings: bool = True,
        config_version: str = "1.0.0"
    ):
        """Initialize the configuration.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Size of the hidden layers.
            num_hidden_layers: Number of hidden layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: Size of the intermediate (feed-forward) layer.
            max_position_embeddings: Maximum sequence length that this model might ever be used with.
            hidden_dropout_prob: Dropout probability for all fully connected layers.
            attention_dropout_prob: Dropout probability for attention weights.
            layer_norm_epsilon: The epsilon used by LayerNorm.
            initializer_range: The standard deviation of the truncated_normal_initializer for initializing weights.
            use_cache: Whether to use the past key/values attentions (if applicable to the model).
            use_rope: Whether to use rotary position embeddings.
            rope_scaling: Dictionary containing the scaling configuration for rotary position embeddings.
            bias: Whether to use bias in all linear layers.
            tie_word_embeddings: Whether to tie input and output embeddings.
            config_version: Version string for the configuration format.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling
        self.bias = bias
        self.tie_word_embeddings = tie_word_embeddings
        self.config_version = config_version

