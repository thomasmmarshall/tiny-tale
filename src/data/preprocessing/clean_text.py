from dataclasses import dataclass
from html import unescape
from multiprocessing import cpu_count
from typing import Iterable, List, Optional
import re


@dataclass
class TextCleaningConfig:
    remove_html: bool = True
    remove_special_chars: bool = True
    lowercase: bool = True
    min_length: int = 1
    max_length: int = 100_000


class TextCleaner:
    """Configurable text cleaner used before tokenizer training."""

    def __init__(self, config: TextCleaningConfig):
        self.config = config

    def clean_text(self, text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None

        cleaned = unescape(text).strip()
        if self.config.remove_html:
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        if self.config.remove_special_chars:
            cleaned = re.sub(r"[^A-Za-z0-9\s.,!?;:\-]", " ", cleaned)

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if self.config.lowercase:
            cleaned = cleaned.lower()

        if len(cleaned) <= self.config.min_length or len(cleaned) > self.config.max_length:
            return None

        return cleaned

    def clean_texts(self, texts: Iterable[str], num_workers: int = 1) -> List[str]:
        # Keep the default path serial; it is deterministic and avoids process overhead
        # for small local datasets and unit tests.
        if num_workers <= 1:
            cleaned = (self.clean_text(text) for text in texts)
            return [text for text in cleaned if text]

        workers = max(1, min(num_workers, cpu_count()))
        from multiprocessing import Pool

        with Pool(workers) as pool:
            cleaned = pool.map(self.clean_text, texts)
        return [text for text in cleaned if text]
