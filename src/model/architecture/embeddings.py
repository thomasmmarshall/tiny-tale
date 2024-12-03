# src/model/architecture/embeddings.py
import torch
import torch.nn as nn
import math
from config import TransformerConfig
from typing import Optional

class Embeddings(nn.Module):
    """Combines token embeddings with optional learned position embeddings."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Only use learned position embeddings if not using RoPE
        self.use_learned_position_embeddings = not config.use_rope
        if self.use_learned_position_embeddings:
            self.position_embeddings = nn.Embedding(
                config.max_seq_length,
                config.hidden_size
            )
            
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        embeddings = self.word_embeddings(input_ids)
        
        if self.use_learned_position_embeddings:
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
            
        return self.dropout(embeddings)