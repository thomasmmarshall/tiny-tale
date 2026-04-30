import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerConfig

class FeedForward(nn.Module):
    """
    Implements the Feed Forward Network of the Transformer with SwiGLU activation.
    Using SwiGLU instead of ReLU as per modern architectures (PaLM, GPT-4, etc.)
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)  # for SwiGLU
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        swish = F.silu(self.w1(x))
        gate = self.w3(x)
        x = swish * gate
        x = self.dropout(x)
        x = self.w2(x)
        return x
