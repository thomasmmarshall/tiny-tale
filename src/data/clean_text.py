# src/data/preprocessing/clean_text.py
import re
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class TextCleaningConfig:
    remove_html: bool = True
    remove_special_chars: bool = True
    lowercase: bool = True
    min_length: int = 20
    max_length: int = 100000

class TextCleaner:
    """Handles text cleaning and normalization with configurable options."""
    
    def __init__(self, config: TextCleaningConfig):
        self.config = config
        self.html_pattern = re.compile('<.*?>')
        self.special_chars_pattern = re.compile(r'[^\w\s.,!?-]')
        
    def clean_text(self, text: str) -> Optional[str]:
        """Clean a single text entry according to configuration."""
        if not isinstance(text, str) or not text.strip():
            return None
            
        if self.config.remove_html:
            text = self.html_pattern.sub(' ', text)
            
        if self.config.remove_special_chars:
            text = self.special_chars_pattern.sub(' ', text)
            
        if self.config.lowercase:
            text = text.lower()
            
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Length filtering
        if not (self.config.min_length <= len(text) <= self.config.max_length):
            return None
            
        return text

    def clean_texts(self, texts: List[str], num_threads: int = 4) -> List[str]:
        """Clean multiple texts in parallel."""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            cleaned_texts = list(filter(None, executor.map(self.clean_text, texts)))
        return cleaned_texts



