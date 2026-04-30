
import pytest
import torch

from src.data.preprocessing.create_dataset import PackedTextDataset, StreamingTextDataset, TextDataCollator

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 99
        
    def encode(self, text, **kwargs):
        # Simple mock encoding: just convert to list of char codes
        return [ord(c) % 100 for c in text]

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


def test_packed_dataset_emits_full_blocks(tmp_path, tokenizer):
    path = tmp_path / "packed.txt"
    path.write_text("abc\ndefghijk\n", encoding="utf-8")

    dataset = PackedTextDataset(
        path,
        tokenizer,
        max_length=5,
        shuffle_buffer_size=0,
        drop_last=True,
    )
    samples = list(dataset)

    assert len(samples) == 2
    assert all(sample.shape == (5,) for sample in samples)
    assert samples[0].tolist() == [97, 98, 99, tokenizer.eos_token_id, 0]


def test_packed_dataset_flushes_remainder_when_requested(tmp_path, tokenizer):
    path = tmp_path / "packed.txt"
    path.write_text("abc\n", encoding="utf-8")

    dataset = PackedTextDataset(
        path,
        tokenizer,
        max_length=6,
        drop_last=False,
        shuffle_buffer_size=0,
    )

    assert [sample.tolist() for sample in dataset] == [[97, 98, 99, tokenizer.eos_token_id]]

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
    assert output['labels'][2, 2:].tolist() == [-100, -100]
