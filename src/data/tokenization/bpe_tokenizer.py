from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
import regex as re
from pathlib import Path
import json
from .vocabulary import Vocabulary
from tqdm import tqdm

class BPETokenizer:
    """Byte-Pair Encoding Tokenizer implementation with training progress bar."""
    
    def __init__(
        self,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: Dict[str, str] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab = Vocabulary(special_tokens)
        self.merges: Dict[Tuple[str, str], str] = {}
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def train(self, texts: List[str]):
        """Train BPE on a list of texts with progress tracking."""
        print("Counting word frequencies...")
        word_freqs = defaultdict(int)
        for text in tqdm(texts, desc="Processing texts"):
            words = [match.group() for match in self.pattern.finditer(text)]
            for word in words:
                word_freqs[word] += 1

        print("Building initial character vocabulary...")
        chars = set()
        for word, freq in word_freqs.items():
            if freq < self.min_frequency:
                continue
            chars.update(word)

        for char in sorted(chars):
            self.vocab.add_token(char)

        splits: Dict[str, List[str]] = {
            word: [c for c in word]
            for word in word_freqs
            if word_freqs[word] >= self.min_frequency
        }

        num_merges = min(
            self.vocab_size - len(self.vocab),
            sum(1 for word in splits if len(splits[word]) >= 2)
        )
        
        print(f"Training BPE: learning {num_merges} merges...")
        pbar = tqdm(total=num_merges, desc="Learning merges")
        
        while len(self.vocab) < self.vocab_size:
            # Count pairs with progress tracking for large vocabularies
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                if freq < self.min_frequency:
                    continue
                    
                split = splits[word]
                if len(split) < 2:
                    continue
                    
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)
            self.vocab.add_token(new_token)
            self.merges[best_pair] = new_token

            # Update splits
            for word in splits:
                split = splits[word]
                if len(split) < 2:
                    continue

                i = 0
                while i < len(split) - 1:
                    if i < len(split) - 1 and tuple(split[i:i + 2]) == best_pair:
                        split[i:i + 2] = [new_token]
                    else:
                        i += 1
                        
            pbar.update(1)
            
        pbar.close()
        print(f"Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        tokens = []
        for match in self.pattern.finditer(text):
            word = match.group()
            current_tokens = [c for c in word]

            while len(current_tokens) >= 2:
                pairs = [(current_tokens[i], current_tokens[i + 1]) 
                        for i in range(len(current_tokens) - 1)]
                pair_found = False
                
                for pair in pairs:
                    if pair in self.merges:
                        idx = current_tokens.index(pair[0])
                        current_tokens[idx:idx + 2] = [self.merges[pair]]
                        pair_found = True
                        break
                        
                if not pair_found:
                    break

            for token in current_tokens:
                if token in self.vocab.token_to_id:
                    tokens.append(self.vocab.token_to_id[token])
                else:
                    tokens.append(self.vocab.token_to_id[self.vocab.special_tokens['<unk>']])
                    
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        tokens = [self.vocab.id_to_token[id] for id in ids]
        return ''.join(tokens).strip()

    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'merges': {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        }
        
        # Save vocabulary
        vocab_path = Path(path).with_suffix('.vocab')
        self.vocab.save(str(vocab_path))
        
        # Save tokenizer config
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        vocab_path = Path(path).with_suffix('.vocab')
        vocab = Vocabulary.load(str(vocab_path))
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            min_frequency=data['min_frequency']
        )
        tokenizer.vocab = vocab
        tokenizer.merges = {
            tuple(k.split(' ')): v 
            for k, v in data['merges'].items()
        }
        
        return tokenizer