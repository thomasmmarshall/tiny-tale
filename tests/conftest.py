from pathlib import Path

import pytest


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def encode(self, text):
        return [ord(c) % 100 for c in text]


@pytest.fixture
def sample_data(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "Hello world\n"
        "This is a test\n"
        "Another line of text\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tokenizer():
    return MockTokenizer()
