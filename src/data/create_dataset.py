
# src/data/preprocessing/create_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Iterator, Optional, Union, Dict
import json
from pathlib import Path
import logging
from torch.utils.data.distributed import DistributedSampler

class StreamingTextDataset(IterableDataset):
    """Memory-efficient dataset that streams data from disk."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int,
        shuffle_buffer_size: int = 10000
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        
    def line_iterator(self) -> Iterator[str]:
        """Iterate through lines of all files in data directory."""
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            files = list(self.data_path.glob('*.txt'))
            
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        yield line
                        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Return iterator over tokenized sequences."""
        buffer = []
        
        for text in self.line_iterator():
            tokens = self.tokenizer.encode(text)
            
            # Split into chunks of max_length
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) == self.max_length:  # Only use complete sequences
                    buffer.append(torch.tensor(chunk))
                
                # Shuffle and yield when buffer is full
                if len(buffer) >= self.shuffle_buffer_size:
                    torch.random.shuffle(buffer)
                    yield from buffer
                    buffer = []
                    
        # Yield remaining items
        if buffer:
            torch.random.shuffle(buffer)
            yield from buffer


class TextDataCollator:
    """Collate function for batching sequences."""
    
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Pad sequences to longest in batch
        max_len = max(len(ex) for ex in examples)
        batch = torch.full((len(examples), max_len), self.pad_token_id)
        
        for i, ex in enumerate(examples):
            batch[i, :len(ex)] = ex
            
        # Create attention mask
        attention_mask = (batch != self.pad_token_id).float()
        
        return {
            'input_ids': batch,
            'attention_mask': attention_mask,
            'labels': batch.clone()  # For language modeling
        }