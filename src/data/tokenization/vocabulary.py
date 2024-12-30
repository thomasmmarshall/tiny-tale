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
        # First add the special token values (e.g. [PAD], [UNK], etc.)
        for token in self.special_tokens.values():
            self.add_token(token)
        
        # Then create reverse mapping for special token keys (e.g. <pad>, <unk>, etc.)
        for key, value in self.special_tokens.items():
            self.token_to_id[key] = self.token_to_id[value]

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its id."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def __len__(self) -> int:
        """Return number of unique tokens (not counting special token aliases)."""
        return len(self.id_to_token)

    def lookup_token(self, token_id: int) -> str:
        """Convert a token ID back to its string representation."""
        if token_id not in self.id_to_token:
            return self.special_tokens['<unk>']
        return self.id_to_token[token_id]

    def to_dict(self) -> dict:
        """Convert vocabulary to a dictionary for serialization."""
        return {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'special_tokens': self.special_tokens
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Vocabulary':
        """Create vocabulary from a dictionary representation."""
        vocab = cls(special_tokens=data['special_tokens'])
        vocab.token_to_id = {str(k): int(v) for k, v in data['token_to_id'].items()}
        vocab.id_to_token = {int(k): str(v) for k, v in data['id_to_token'].items()}
        return vocab

    def save(self, path: str):
        """Save vocabulary to file."""
        data = self.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __getitem__(self, token: str) -> int:
        """Get the ID for a token, return UNK token ID if not found."""
        return self.token_to_id.get(token, self.token_to_id[self.special_tokens['<unk>']])

