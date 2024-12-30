# src/pipeline.py
import logging
import sys
import time
from pathlib import Path
from typing import List
import yaml
import torch
import psutil
import GPUtil
from datetime import datetime
import json
from itertools import islice

from data.preprocessing.clean_text import TextCleaner, TextCleaningConfig
from data.tokenization.bpe_tokenizer import BPETokenizer
from data.preprocessing.data_module import LMDataModule
from model.architecture.transformer import TransformerConfig
from model.training.trainer import TransformerLightningModule

class Pipeline:
    def __init__(self, config_path: str, experiment_name: str):
        self.experiment_name = experiment_name
        self.logger = self.setup_logging(experiment_name)
        self.config = self.load_config(config_path)
        self.experiment_dir = self.setup_experiment_dirs(experiment_name)

    def setup_logging(self, experiment_name: str) -> logging.Logger:
        """Configure logging to both file and console with timestamps."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logfile = log_dir / f"{experiment_name}_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)

    def load_config(self, config_path: str) -> dict:
        """Load and return the configuration from a YAML file."""
        self.logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.logger.info("Configuration loaded:\n" + json.dumps(config, indent=2))
        return config

    def setup_experiment_dirs(self, experiment_name: str) -> Path:
        """Create and return the experiment directory structure."""
        experiment_dir = Path("experiments") / experiment_name
        for subdir in ['checkpoints', 'tokenizer', 'processed_data']:
            dir_path = experiment_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directory created: {dir_path}")
        return experiment_dir

    def log_system_info(self):
        """Log detailed system information: CPU, memory, GPU, and PyTorch details."""
        self.logger.info("=== System Information ===")

        # CPU info
        cpu_info = {
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "CPU Usage": f"{psutil.cpu_percent()}%"
        }
        self.logger.info("CPU Information:\n" + json.dumps(cpu_info, indent=2))

        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "Total (GB)": f"{memory.total / (1024**3):.2f}",
            "Available (GB)": f"{memory.available / (1024**3):.2f}",
            "Used (GB)": f"{memory.used / (1024**3):.2f}",
            "Percentage": f"{memory.percent}%"
        }
        self.logger.info("Memory Information:\n" + json.dumps(memory_info, indent=2))

        # GPU info
        try:
            for i, gpu in enumerate(GPUtil.getGPUs()):
                gpu_info = {
                    "ID": gpu.id,
                    "Name": gpu.name,
                    "Memory Total (MB)": gpu.memoryTotal,
                    "Memory Used (MB)": gpu.memoryUsed,
                    "Memory Free (MB)": gpu.memoryFree,
                    "Memory Utilization (%)": f"{gpu.memoryUtil * 100:.2f}",
                    "GPU Utilization (%)": f"{gpu.load * 100:.2f}"
                }
                self.logger.info(f"GPU {i} Information:\n" + json.dumps(gpu_info, indent=2))
        except Exception as e:
            self.logger.warning(f"Could not retrieve GPU information: {e}")

        # PyTorch info
        torch_info = {
            "PyTorch Version": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "Number of GPUs": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        self.logger.info("PyTorch Information:\n" + json.dumps(torch_info, indent=2))

    def process_data(self) -> List[str]:
        """Clean and preprocess the raw training data."""
        self.logger.info("=== Starting Data Processing ===")
        
        cleaning_config = TextCleaningConfig(**self.config['data']['cleaning'])
        cleaner = TextCleaner(cleaning_config)
        self.logger.info(f"TextCleaner initialized with config: {cleaning_config}")

        train_path = Path(self.config['data']['train_path'])
        self.logger.info(f"Processing training data from {train_path}")

        # Log a small sample of raw data
        with open(train_path, 'r') as f:
            sample_raw = list(islice(f, 5))
        self.logger.info(f"Sample of raw training data:\n{sample_raw}")

        start_time = time.time()
        processed_texts = []
        lines_processed = 0

        with open(train_path, 'r') as f:
            for line in f:
                cleaned_line = cleaner.clean_text(line)
                if cleaned_line:
                    processed_texts.append(cleaned_line)
                lines_processed += 1
                if lines_processed % 10000 == 0:
                    self.logger.info(f"Processed {lines_processed} lines...")

        processing_time = time.time() - start_time
        total_tokens = sum(len(t.split()) for t in processed_texts)
        self.logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Total tokens: {total_tokens}")
        self.logger.info("Sample of processed texts:\n" + json.dumps(processed_texts[:5], indent=2))

        # Save processed data
        processed_train_path = self.experiment_dir / 'processed_data' / 'train.txt'
        with open(processed_train_path, 'w') as f:
            f.write("\n".join(processed_texts))
        self.logger.info(f"Processed training data saved to {processed_train_path}")

        return processed_texts

    def train_tokenizer(self, texts: List[str]) -> BPETokenizer:
        """Train and save the BPE tokenizer, or load if it already exists."""
        self.logger.info("=== Starting Tokenizer Setup ===")

        tokenizer_path = self.experiment_dir / 'tokenizer' / 'tokenizer.json'
        
        # Try to load existing tokenizer
        if tokenizer_path.exists():
            self.logger.info(f"Found existing tokenizer at {tokenizer_path}")
            try:
                tokenizer = BPETokenizer.load(str(tokenizer_path))
                self.logger.info("Successfully loaded existing tokenizer")
                
                # Log sample tokenization
                if texts:
                    sample_text = texts[0]
                    tokens = tokenizer.encode(sample_text)
                    self.logger.info("Sample tokenization with loaded tokenizer:")
                    self.logger.info("Original text: " + sample_text)
                    self.logger.info("Tokens: " + str(tokens))
                    self.logger.info("Decoded text: " + tokenizer.decode(tokens['input_ids']))
                
                return tokenizer
            except Exception as e:
                self.logger.warning(f"Failed to load existing tokenizer: {e}. Training new one.")

        # Train new tokenizer if loading failed or none exists
        tokenizer_config = {
            'vocab_size': self.config['model']['vocab_size'],
            'min_frequency': self.config['tokenizer']['min_frequency'],
            'special_tokens': self.config['tokenizer']['special_tokens']
        }
        tokenizer = BPETokenizer(tokenizer_config)

        start_time = time.time()
        tokenizer.train(texts)
        training_time = time.time() - start_time

        self.logger.info(f"Tokenizer training completed in {training_time:.2f} seconds")
        self.logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")

        # Log sample tokenization
        if texts:
            sample_text = texts[0]
            tokens = tokenizer.encode(sample_text)
            self.logger.info("Sample tokenization:")
            self.logger.info("Original text: " + sample_text)
            self.logger.info("Tokens: " + str(tokens))
            self.logger.info("Decoded text: " + tokenizer.decode(tokens['input_ids']))

        tokenizer.save(str(tokenizer_path))
        self.logger.info(f"Tokenizer saved to {tokenizer_path}")
        return tokenizer

    def setup_data_module(self, tokenizer: BPETokenizer) -> LMDataModule:
        """Initialize the data module for training."""
        data_config = self.config['data']
        data_module = LMDataModule(
            train_path=data_config['train_path'],
            val_path=data_config['val_path'],
            tokenizer=tokenizer,
            **data_config['dataloader']
        )
        return data_module

    def setup_model(self) -> TransformerLightningModule:
        """Initialize and log statistics about the transformer model."""
        self.logger.info("=== Setting up Model ===")
        model_config = TransformerConfig(**self.config['model'])
        self.logger.info("Model Configuration:\n" + json.dumps(model_config.__dict__, indent=2))

        # Extract only the parameters needed for the model
        model_training_params = {
            'learning_rate': self.config['training']['learning_rate'],
            'weight_decay': self.config['training']['weight_decay'],
            'warmup_steps': self.config['training']['warmup_steps'],
            'max_steps': self.config['training']['max_steps'],
            'grad_clip_val': self.config['training']['grad_clip_val']
        }
        model = TransformerLightningModule(config=model_config, **model_training_params)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_stats = {
            "Total Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Model Size (MB)": f"{total_params * 4 / (1024**2):.2f}",
            "Architecture": str(model)
        }
        self.logger.info("Model Statistics:\n" + json.dumps(model_stats, indent=2))
        return model

    def run(self):
        """Execute the full training pipeline."""
        self.logger.info("=== Starting Training Pipeline ===")
        self.log_system_info()

        try:
            processed_texts = self.process_data()
            tokenizer = self.train_tokenizer(processed_texts)
            data_module = self.setup_data_module(tokenizer)
            model = self.setup_model()

            self.logger.info("=== Starting Model Training ===")
            model.train()  # Actual training logic should be called here

            self.logger.info("=== Pipeline Completed Successfully ===")
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise

def main():
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <config_path> <experiment_name>")
        sys.exit(1)

    config_path = sys.argv[1]
    experiment_name = sys.argv[2]

    pipeline = Pipeline(config_path, experiment_name)
    pipeline.run()

if __name__ == "__main__":
    main()
