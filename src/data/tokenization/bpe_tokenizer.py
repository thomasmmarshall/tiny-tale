from typing import Dict, List, Tuple, Set, Optional, Union
from collections import Counter, defaultdict
import regex as re
from pathlib import Path
import json
import logging
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial
import pickle
from tqdm.auto import tqdm
import os
from dataclasses import dataclass, asdict
from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)

@dataclass
class BPEConfig:
    """Configuration for BPE tokenizer."""
    vocab_size: int = 8192
    min_frequency: int = 2
    special_tokens: Optional[Dict[str, str]] = None
    max_token_length: int = 100
    lowercase: bool = False
    unicode_normalizer: Optional[str] = 'NFKC'
    cache_capacity: int = 10000
    num_threads: int = os.cpu_count()

class TokenizerError(Exception):
    """Base class for tokenizer errors."""
    pass

class BPETokenizer:
    """Byte-Pair Encoding Tokenizer with caching and parallel processing."""
    
    def __init__(self, config: Union[BPEConfig, dict]):
        if isinstance(config, dict):
            config = BPEConfig(**config)
        self.config = config
        
        # Initialize vocabulary with special tokens
        self.vocab = Vocabulary(config.special_tokens)
        
        # Initialize merge rules
        self.merges: Dict[Tuple[str, str], str] = {}
        
        # Compile regex pattern for tokenization
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Initialize cache
        self._cache: Dict[str, List[str]] = {}
        self._cache_keys = []
    
    @property
    def pad_token_id(self) -> int:
        """Get the ID of the padding token."""
        return self.vocab.token_to_id[self.vocab.special_tokens['<pad>']]

    @property
    def unk_token_id(self) -> int:
        """Get the ID of the unknown token."""
        return self.vocab.token_to_id[self.vocab.special_tokens['<unk>']]

    @property
    def bos_token_id(self) -> int:
        """Get the ID of the beginning of sequence token."""
        return self.vocab.token_to_id[self.vocab.special_tokens['<bos>']]

    @property
    def eos_token_id(self) -> int:
        """Get the ID of the end of sequence token."""
        return self.vocab.token_to_id[self.vocab.special_tokens['<eos>']]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text with normalization and lowercasing."""
        if not isinstance(text, str):
            raise TokenizerError(f"Input must be string, got {type(text)}")
            
        if self.config.unicode_normalizer:
            try:
                import unicodedata
                text = unicodedata.normalize(self.config.unicode_normalizer, text)
            except Exception as e:
                logger.warning(f"Unicode normalization failed: {str(e)}")
                
        if self.config.lowercase:
            text = text.lower()
            
        return text
    
    def _update_cache(self, key: str, value: List[str]):
        """Update LRU cache with size limit."""
        if key in self._cache:
            self._cache_keys.remove(key)
        elif len(self._cache) >= self.config.cache_capacity:
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]
            
        self._cache[key] = value
        self._cache_keys.append(key)
    
    def _tokenize_worker(self, text: str) -> List[str]:
        """Worker function for parallel tokenization."""
        return [match.group() for match in self.pattern.finditer(text)]
    
    def train(self, texts: List[str]):
        """Train BPE with parallel processing and proper error handling."""
        logger.info("Starting BPE training...")
        
        # Parallel word frequency counting
        word_freqs = defaultdict(int)
        with ProcessPoolExecutor(max_workers=self.config.num_threads) as executor:
            tokenized_texts = list(tqdm(
                executor.map(self._tokenize_worker, texts),
                total=len(texts),
                desc="Tokenizing texts"
            ))
            
        for words in tokenized_texts:
            for word in words:
                word_freqs[word] += 1
        
        # Build initial character vocabulary
        logger.info("Building initial character vocabulary...")
        chars = set()
        for word, freq in word_freqs.items():
            if freq < self.config.min_frequency:
                continue
            if len(word) > self.config.max_token_length:
                logger.warning(f"Skipping word longer than {self.config.max_token_length} chars: {word[:20]}...")
                continue
            chars.update(word)
        
        for char in sorted(chars):
            self.vocab.add_token(char)
        
        # Initialize splits dictionary
        splits = {
            word: [c for c in word]
            for word, freq in word_freqs.items()
            if freq >= self.config.min_frequency and len(word) <= self.config.max_token_length
        }
        
        # Calculate number of merges
        num_merges = min(
            self.config.vocab_size - len(self.vocab),
            sum(1 for word in splits if len(splits[word]) >= 2)
        )
        
        logger.info(f"Learning {num_merges} BPE merges...")
        pbar = tqdm(total=num_merges, desc="Learning merges")
        
        try:
            while len(self.vocab) < self.config.vocab_size:
                # Count pairs
                pair_freqs = defaultdict(int)
                for word, freq in word_freqs.items():
                    if freq < self.config.min_frequency or len(word) > self.config.max_token_length:
                        continue
                    
                    split = splits.get(word)
                    if not split or len(split) < 2:
                        continue
                    
                    for i in range(len(split) - 1):
                        pair = (split[i], split[i + 1])
                        pair_freqs[pair] += freq
                
                if not pair_freqs:
                    break
                
                # Find best pair
                best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
                new_token = ''.join(best_pair)
                
                # Validate new token
                if len(new_token) > self.config.max_token_length:
                    logger.warning(f"Skipping merge that would create token longer than {self.config.max_token_length} chars")
                    continue
                
                self.vocab.add_token(new_token)
                self.merges[best_pair] = new_token
                
                # Update splits efficiently
                new_splits = {}
                for word, split in splits.items():
                    if len(split) < 2:
                        new_splits[word] = split
                        continue
                    
                    i = 0
                    new_split = []
                    while i < len(split):
                        if i < len(split) - 1 and tuple(split[i:i + 2]) == best_pair:
                            new_split.append(new_token)
                            i += 2
                        else:
                            new_split.append(split[i])
                            i += 1
                    new_splits[word] = new_split
                
                splits = new_splits
                pbar.update(1)
                
        except Exception as e:
            logger.error(f"Error during BPE training: {str(e)}")
            raise
        finally:
            pbar.close()
            
        logger.info(f"Finished training. Final vocabulary size: {len(self.vocab)}")
    
    def _encode_word(self, word: str) -> List[str]:
        """Encode a single word to subword tokens using trained merges."""
        if word in self._cache:
            return self._cache[word]
            
        splits = [c for c in word]
        while len(splits) >= 2:
            min_pair = None
            min_idx = None
            
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i + 1])
                if pair in self.merges:
                    min_pair = pair
                    min_idx = i
                    break
                    
            if min_pair is None:
                break
                
            splits[min_idx:min_idx + 2] = [self.merges[min_pair]]
        
        self._update_cache(word, splits)
        return splits
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_attention_mask: bool = False
    ) -> Dict[str, List[int]]:
        """Encode text to token ids with various options."""
        if not self.merges:
            raise TokenizerError("Tokenizer is not trained. Call train() first.")
            
        try:
            text = self._preprocess_text(text)
            words = self._tokenize_worker(text)
            tokens = []
            
            for word in words:
                tokens.extend(self._encode_word(word))
            
            # Convert tokens to ids
            token_ids = []
            for token in tokens:
                # Try to get token ID, use UNK token if not found
                token_id = self.vocab.token_to_id.get(token)
                if token_id is None:
                    token_id = self.vocab.token_to_id[self.vocab.special_tokens['<unk>']]
                token_ids.append(token_id)
            
            # Handle special tokens
            if add_special_tokens:
                bos_token = self.vocab.special_tokens.get('<bos>')
                eos_token = self.vocab.special_tokens.get('<eos>')
                if bos_token:
                    token_ids.insert(0, self.vocab.token_to_id[bos_token])
                if eos_token:
                    token_ids.append(self.vocab.token_to_id[eos_token])
            
            # Handle length constraints
            if max_length is not None:
                if truncation and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                if padding and len(token_ids) < max_length:
                    pad_id = self.vocab.token_to_id[self.vocab.special_tokens['<pad>']]
                    token_ids.extend([pad_id] * (max_length - len(token_ids)))
            
            result = {'input_ids': token_ids}
            if return_attention_mask:
                attention_mask = [1] * len(token_ids)
                if padding and max_length is not None:
                    attention_mask.extend([0] * (max_length - len(attention_mask)))
                result['attention_mask'] = attention_mask
            
            return result
            
        except Exception as e:
            raise TokenizerError(f"Failed to encode text: {str(e)}")
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text."""
        try:
            tokens = []
            for id in ids:
                token = self.vocab.lookup_token(id)
                if skip_special_tokens and token in self.vocab.special_tokens.values():
                    continue
                tokens.append(token)
            return ''.join(tokens)
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}")
            raise TokenizerError(f"Failed to decode ids: {str(e)}")
    
    def save(self, path: str):
        """Save tokenizer state to disk."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'config': asdict(self.config),
                'vocab': self.vocab.to_dict(),
                'merges': {','.join(k): v for k, v in self.merges.items()}
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Tokenizer saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving tokenizer: {str(e)}")
            raise TokenizerError(f"Failed to save tokenizer: {str(e)}")
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer state from disk."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            tokenizer = cls(state['config'])
            tokenizer.vocab = Vocabulary.from_dict(state['vocab'])
            tokenizer.merges = {
                tuple(k.split(',')): v
                for k, v in state['merges'].items()
            }
            
            logger.info(f"Tokenizer loaded from {path}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise TokenizerError(f"Failed to load tokenizer: {str(e)}")