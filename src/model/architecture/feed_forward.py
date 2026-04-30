import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerConfig

class FeedForward(nn.Module):
    """SwiGLU feed-forward block compatible with the main transformer config."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(x)
        return self.w2(x)
