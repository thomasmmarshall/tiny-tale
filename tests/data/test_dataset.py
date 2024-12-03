
# tests/data/test_dataset.py
import pytest
import torch
from src.data.preprocessing.create_dataset import StreamingTextDataset, TextDataCollator
from pathlib import Path
import tempfile

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        
    def encode(self, text):
        # Simple mock encoding: just convert to list of char codes
        return [ord(c) % 100 for c in text]

@pytest.fixture
def sample_data():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Hello world\n")
        f.write("This is a test\n")
        f.write("Another line of text\n")
        return Path(f.name)

@pytest.fixture
def tokenizer():
    return MockTokenizer()

def test_streaming_dataset(sample_data, tokenizer):
    dataset = StreamingTextDataset(
        sample_data,
        tokenizer,
        max_length=10,
        shuffle_buffer_size=2
    )
    
    # Convert iterator to list for testing
    samples = list(iter(dataset))
    assert len(samples) > 0
    assert all(isinstance(s, torch.Tensor) for s in samples)
    assert all(len(s) <= 10 for s in samples)

def test_data_collator(tokenizer):
    collator = TextDataCollator(pad_token_id=tokenizer.pad_token_id)
    
    # Create sample batch
    batch = [
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([1, 2])
    ]
    
    output = collator(batch)
    
    assert 'input_ids' in output
    assert 'attention_mask' in output
    assert 'labels' in output
    assert output['input_ids'].shape == (3, 4)  # Batch size 3, max length 4
    assert output['attention_mask'].shape == (3, 4)
    assert output['labels'].shape == (3, 4)
