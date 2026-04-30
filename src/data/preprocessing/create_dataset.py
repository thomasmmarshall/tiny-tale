import random
from pathlib import Path
from typing import Iterable, Iterator, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info


class StreamingTextDataset(IterableDataset):
    """Stream text as packed fixed-length token blocks.

    Packing documents into full context windows is critical for budget training:
    it avoids spending steps on padding-heavy short lines and avoids truncating
    useful tokens from long lines.
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer,
        max_length: int,
        shuffle_buffer_size: int = 0,
        pack_documents: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pack_documents = pack_documents
        self.drop_last = drop_last
        self.seed = seed

    def _encode(self, text: str, add_special_tokens: bool) -> List[int]:
        try:
            encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            encoded = self.tokenizer.encode(text)
        if isinstance(encoded, dict):
            token_ids = encoded["input_ids"]
        else:
            token_ids = encoded
        return list(token_ids)

    def _iter_lines(self) -> Iterator[str]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        with open(self.path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f):
                if line_number % num_workers != worker_id:
                    continue
                text = line.strip()
                if text:
                    yield text

    def _iter_unpacked(self) -> Iterator[torch.Tensor]:
        for text in self._iter_lines():
            token_ids = self._encode(text, add_special_tokens=True)[: self.max_length]
            if token_ids:
                yield torch.tensor(token_ids, dtype=torch.long)

    def _iter_packed(self) -> Iterator[torch.Tensor]:
        buffer: List[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        for text in self._iter_lines():
            token_ids = self._encode(text, add_special_tokens=False)
            if eos_token_id is not None:
                token_ids.append(eos_token_id)
            buffer.extend(token_ids)

            while len(buffer) >= self.max_length:
                yield torch.tensor(buffer[: self.max_length], dtype=torch.long)
                del buffer[: self.max_length]

        if buffer and not self.drop_last:
            yield torch.tensor(buffer, dtype=torch.long)

    def _shuffle_buffer(self, samples: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        if self.shuffle_buffer_size <= 1:
            yield from samples
            return

        rng = random.Random(self.seed)
        buffer: List[torch.Tensor] = []
        for sample in samples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = sample

        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self) -> Iterable[torch.Tensor]:
        samples = self._iter_packed() if self.pack_documents else self._iter_unpacked()
        yield from self._shuffle_buffer(samples)


PackedTextDataset = StreamingTextDataset


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
