
# src/data/preprocessing/data_module.py
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class LMDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for language modeling."""
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        tokenizer,
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        shuffle_buffer_size: int = 10000
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        
        self.collator = TextDataCollator(tokenizer.pad_token_id)
        
    def setup(self, stage: Optional[str] = None):
        """Create datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = StreamingTextDataset(
                self.train_path,
                self.tokenizer,
                self.max_length,
                self.shuffle_buffer_size
            )
            self.val_dataset = StreamingTextDataset(
                self.val_path,
                self.tokenizer,
                self.max_length,
                self.shuffle_buffer_size
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator
        )