# tests/data/test_data_module.py
import pytest
import pytorch_lightning as pl
from src.data.preprocessing.data_module import LMDataModule

def test_data_module_setup(sample_data, tokenizer):
    data_module = LMDataModule(
        train_path=sample_data,
        val_path=sample_data,  # Using same file for testing
        tokenizer=tokenizer,
        batch_size=2,
        max_length=10
    )
    
    data_module.setup()
    
    # Test train dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch
    
    # Test val dataloader
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    assert isinstance(batch, dict)
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch