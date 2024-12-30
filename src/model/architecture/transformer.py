# src/model/architecture/transformer.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration class for transformer model with validation and versioning."""
    
    # Model architecture
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    
    # Optional parameters with defaults
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    use_rope: bool = True
    rope_scaling: Optional[dict] = None
    bias: bool = True
    tie_word_embeddings: bool = True
    
    # Version tracking
    config_version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_attention_heads}"
            )
        
        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                f"intermediate_size {self.intermediate_size} should be larger than hidden_size {self.hidden_size}"
            )
            
        if not 0 <= self.hidden_dropout_prob <= 1:
            raise ValueError(f"hidden_dropout_prob must be between 0 and 1, got {self.hidden_dropout_prob}")
            
        if not 0 <= self.attention_dropout_prob <= 1:
            raise ValueError(f"attention_dropout_prob must be between 0 and 1, got {self.attention_dropout_prob}")
    
    def save(self, save_path: str):
        """Save configuration to JSON file with version tracking."""
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'TransformerConfig':
        """Load configuration from JSON file with version compatibility check."""
        with open(config_path) as f:
            config_dict = json.load(f)
            
        loaded_version = config_dict.pop('config_version', '0.0.0')
        current_version = cls.config_version
        
        if loaded_version != current_version:
            print(f"Warning: Loading config with version {loaded_version}, current version is {current_version}")
            
        return cls(**config_dict)
    
    def get_num_parameters(self) -> int:
        """Calculate total number of parameters in the model."""
        embedding_params = self.vocab_size * self.hidden_size  # Token embeddings
        position_params = self.max_position_embeddings * self.hidden_size  # Position embeddings
        
        # Parameters per layer
        attention_params = (
            4 * self.hidden_size * self.hidden_size +  # Q, K, V, O projections
            (4 * self.hidden_size if self.bias else 0)  # Biases if used
        )
        
        ffn_params = (
            2 * self.hidden_size * self.intermediate_size +  # Two linear layers
            self.hidden_size + self.intermediate_size  # Layer norm params
        )
        
        params_per_layer = attention_params + ffn_params
        total_params = embedding_params + position_params + (params_per_layer * self.num_hidden_layers)
        
        return total_params


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Create Linear layers for Q, K, V projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the tensor for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Project inputs to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # Final output projection
        output = self.output(context_layer)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(
            self.ln_1(hidden_states),  # Pre-normalization
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        # Feed-forward
        ff_output = self.feed_forward(self.ln_2(hidden_states))  # Pre-normalization
        hidden_states = hidden_states + self.dropout(ff_output)

        return hidden_states


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln_embedding = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Language modeling head
        if config.tie_word_embeddings:
            self.lm_head = lambda x: F.linear(x, self.word_embeddings.weight)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.word_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get input shape
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Convert attention mask to attention bias
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Get embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = word_embeds + position_embeds
        hidden_states = self.ln_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss

        return logits

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype