try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - exercised only when Lightning is absent.
    class _LightningDataModule:
        pass

    class _LightningFallback:
        LightningDataModule = _LightningDataModule

    pl = _LightningFallback()

from typing import Optional

from .create_dataset import StreamingTextDataset, TextDataCollator


class LMDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for language-model text files."""

    def __init__(
        self,
        train_path,
        val_path,
        tokenizer,
        batch_size: int = 8,
        max_length: int = 256,
        num_workers: int = 0,
        shuffle_buffer_size: int = 1000,
        prefetch_factor: Optional[int] = None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_factor = prefetch_factor
        self.collator = TextDataCollator(pad_token_id=tokenizer.pad_token_id)

    def setup(self, stage=None):
        self.train_dataset = StreamingTextDataset(
            self.train_path,
            self.tokenizer,
            max_length=self.max_length,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )
        self.val_dataset = StreamingTextDataset(
            self.val_path,
            self.tokenizer,
            max_length=self.max_length,
            shuffle_buffer_size=1,
        )

    def _loader(self, dataset, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader

        kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": self.collator,
            "num_workers": self.num_workers,
            "shuffle": shuffle and not isinstance(dataset, torch.utils.data.IterableDataset),
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            self.setup("fit")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.setup("validate")
        return self._loader(self.val_dataset, shuffle=False)
