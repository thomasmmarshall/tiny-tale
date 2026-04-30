# src/model/architecture/embeddings.py
import torch
import torch.nn as nn
from .transformer import TransformerConfig

class Embeddings(nn.Module):
    """Token embeddings; RoPE is applied inside attention blocks."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        return self.dropout(embeddings)