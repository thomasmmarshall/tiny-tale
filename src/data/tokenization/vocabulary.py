# src/data/tokenization/vocabulary.py
from typing import Dict, List, Optional, Set
from collections import Counter
import json
from pathlib import Path
import regex as re

class Vocabulary:
    """Manages the token vocabulary with special tokens support."""
    
    def __init__(
        self,
        special_tokens: Dict[str, str] = None
    ):
        self.special_tokens = special_tokens or {
            '<pad>': '[PAD]',
            '<unk>': '[UNK]',
            '<bos>': '[BOS]',
            '<eos>': '[EOS]',
        }
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        """Initialize special tokens in the vocabulary."""
        for token in self.special_tokens.values():
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its id."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str):
        """Save vocabulary to file."""
        data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(special_tokens=data['special_tokens'])
        vocab.token_to_id = data['token_to_id']
        vocab.id_to_token = {int(k): v for k, v in data['token_to_id'].items()}
        return vocab

    def __getitem__(self, token: str) -> int:
        """Get the ID for a token, return UNK token ID if not found."""
        return self.token_to_id.get(token, self.token_to_id[self.special_tokens['<unk>']])

