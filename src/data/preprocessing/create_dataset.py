from pathlib import Path
from typing import Iterable, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset


class StreamingTextDataset(IterableDataset):
    """Stream a text file and yield fixed-length token id tensors."""

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer,
        max_length: int,
        shuffle_buffer_size: int = 0,
    ):
        super().__init__()
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size

    def _encode(self, text: str) -> List[int]:
        encoded = self.tokenizer.encode(text)
        if isinstance(encoded, dict):
            token_ids = encoded["input_ids"]
        else:
            token_ids = encoded
        return list(token_ids)[: self.max_length]

    def __iter__(self) -> Iterable[torch.Tensor]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                token_ids = self._encode(text)
                if token_ids:
                    yield torch.tensor(token_ids, dtype=torch.long)


class TextDataCollator:
    """Pad variable-length language-model samples into a batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[torch.Tensor]) -> dict:
        input_ids = pad_sequence(
            batch,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = input_ids.ne(self.pad_token_id).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
