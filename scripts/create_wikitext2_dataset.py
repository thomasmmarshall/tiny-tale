# scripts/create_wikitext_dataset.py
import os
import logging
from pathlib import Path
import json
from tqdm import tqdm
from datasets import load_dataset
from typing import Dict, Any

class WikiText2Processor:
    """Downloads and processes WikiText-2 dataset using HuggingFace datasets."""
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        use_raw: bool = True  # True for raw text, False for tokenized version
    ):
        self.output_dir = Path(output_dir)
        self.use_raw = use_raw
        self.dataset_name = "wikitext-raw-v1" if use_raw else "wikitext-2-v1"
        self.setup_logging()

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('wikitext_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_split(self, split_data: Dict[str, Any], output_path: Path) -> int:
        """Process a single data split and return number of lines."""
        self.logger.info(f"Processing split to {output_path}")
        line_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for item in tqdm(split_data, desc=f"Processing {output_path.name}"):
                text = item['text'].strip()
                # Skip empty lines and Wikipedia article markers
                if text and not text.startswith('='):
                    outfile.write(text + '\n')
                    line_count += 1
                    
        return line_count

    def create_dataset(self) -> None:
        """Create the WikiText-2 dataset."""
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Loading WikiText-2 dataset from HuggingFace")
            dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1" if self.use_raw else "wikitext-2-v1",
            )
            
            # Process each split
            dataset_stats = {}
            for split in ['train', 'validation', 'test']:
                output_path = self.output_dir / f"{split}.txt"
                
                # For validation split, dataset uses 'validation' but we want 'valid.txt'
                if split == 'validation':
                    output_path = self.output_dir / 'valid.txt'
                
                lines = self.process_split(dataset[split], output_path)
                dataset_stats[split] = {
                    'num_lines': lines,
                    'path': str(output_path)
                }
            
            # Save dataset info
            dataset_info = {
                'name': 'WikiText-2',
                'version': 'raw' if self.use_raw else 'tokenized',
                'splits': dataset_stats,
                'source': 'HuggingFace Datasets',
                'description': 'WikiText-2 dataset for language modeling (https://huggingface.co/datasets/wikitext)'
            }
            
            with open(self.output_dir / 'dataset_info.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            self.logger.info("Dataset creation completed successfully")
            self.logger.info(f"Dataset statistics: {json.dumps(dataset_stats, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error creating dataset: {e}", exc_info=True)
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create WikiText-2 dataset')
    parser.add_argument('--output_dir', default='data/raw', help='Output directory')
    parser.add_argument('--use_raw', action='store_true', help='Use raw text version instead of tokenized')
    
    args = parser.parse_args()
    
    processor = WikiText2Processor(
        output_dir=args.output_dir,
        use_raw=args.use_raw
    )
    processor.create_dataset()

if __name__ == "__main__":
    main()