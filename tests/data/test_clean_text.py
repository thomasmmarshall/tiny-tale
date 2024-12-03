# tests/data/test_clean_text.py
import pytest
from src.data.preprocessing.clean_text import TextCleaner, TextCleaningConfig

@pytest.fixture
def cleaner():
    config = TextCleaningConfig(
        remove_html=True,
        remove_special_chars=True,
        lowercase=True,
        min_length=5,
        max_length=1000
    )
    return TextCleaner(config)

def test_html_removal(cleaner):
    text = "Hello <b>World</b> <script>alert('test')</script>"
    cleaned = cleaner.clean_text(text)
    assert cleaned == "hello world alert test"

def test_special_chars(cleaner):
    text = "Hello! @#$%^& World?"
    cleaned = cleaner.clean_text(text)
    assert cleaned == "hello! world?"

def test_length_filtering(cleaner):
    short_text = "hi"
    long_text = "a" * 1001
    assert cleaner.clean_text(short_text) is None
    assert cleaner.clean_text(long_text) is None

def test_parallel_cleaning(cleaner):
    texts = ["Hello <b>World</b>", "Test @#$ Text", "Short"]
    cleaned = cleaner.clean_texts(texts)
    assert len(cleaned) == 2  # "Short" should be filtered out
    assert "hello world" in cleaned
    assert "test text" in cleaned

