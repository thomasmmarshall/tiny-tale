# src/pipeline.py
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
import yaml
import torch
import psutil
import GPUtil
from datetime import datetime
import json

from data.preprocessing.clean_text import TextCleaner, TextCleaningConfig
from data.tokenization.bpe_tokenizer import BPETokenizer
from data.preprocessing.data_module import LMDataModule
from model.architecture.transformer import TransformerConfig
from model.training.trainer import TransformerLightningModule

class Pipeline:
    def __init__(self, config_path: str, experiment_name: str):
        self.setup_logging(experiment_name)
        self.load_config(config_path)
        self.setup_directories(experiment_name)
        self.logger = logging.getLogger(__name__)

    def setup_logging(self, experiment_name: str):
        """Configure detailed logging to both file and console."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f"logs/{experiment_name}_{timestamp}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def load_config(self, config_path: str):
        """Load and validate configuration."""
        self.logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.logger.info(f"Configuration loaded: {json.dumps(self.config, indent=2)}")

    def setup_directories(self, experiment_name: str):
        """Create necessary directories for the experiment."""
        dirs = ['checkpoints', 'tokenizer', 'processed_data']
        self.experiment_dir = Path(f"experiments/{experiment_name}")
        
        for dir_name in dirs:
            dir_path = self.experiment_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def log_system_info(self):
        """Log detailed system information."""
        self.logger.info("=== System Information ===")
        
        # CPU Info
        cpu_info = {
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            "CPU Usage": f"{psutil.cpu_percent()}%"
        }
        self.logger.info(f"CPU Information: {json.dumps(cpu_info, indent=2)}")
        
        # Memory Info
        memory = psutil.virtual_memory()
        memory_info = {
            "Total": f"{memory.total / (1024**3):.2f}GB",
            "Available": f"{memory.available / (1024**3):.2f}GB",
            "Used": f"{memory.used / (1024**3):.2f}GB",
            "Percentage": f"{memory.percent}%"
        }
        self.logger.info(f"Memory Information: {json.dumps(memory_info, indent=2)}")
        
        # GPU Info
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info = {
                    "ID": gpu.id,
                    "Name": gpu.name,
                    "Memory Total": f"{gpu.memoryTotal}MB",
                    "Memory Used": f"{gpu.memoryUsed}MB",
                    "Memory Free": f"{gpu.memoryFree}MB",
                    "Memory Utilization": f"{gpu.memoryUtil * 100}%",
                    "GPU Utilization": f"{gpu.load * 100}%"
                }
                self.logger.info(f"GPU {i} Information: {json.dumps(gpu_info, indent=2)}")
        except Exception as e:
            self.logger.warning(f"Could not get GPU information: {e}")

        # PyTorch Info
        torch_info = {
            "Version": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "Number of GPUs": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        self.logger.info(f"PyTorch Information: {json.dumps(torch_info, indent=2)}")

    def process_data(self):
        """Clean and preprocess the raw data."""
        self.logger.info("=== Starting Data Processing ===")
        
        # Initialize text cleaner
        cleaning_config = TextCleaningConfig(**self.config['data']['cleaning'])
        cleaner = TextCleaner(cleaning_config)
        self.logger.info(f"Initialized text cleaner with config: {cleaning_config}")
        
        # Process training data
        train_path = Path(self.config['data']['train_path'])
        self.logger.info(f"Processing training data from {train_path}")
        
        # Log sample of raw data
        with open(train_path, 'r') as f:
            sample_raw = list(islice(f, 5))
        self.logger.info(f"Sample of raw training data: {sample_raw}")
        
        # Clean and save processed data
        processed_train_path = self.experiment_dir / 'processed_data' / 'train.txt'
        self.logger.info(f"Saving processed training data to {processed_train_path}")
        
        processed_texts = []
        total_tokens = 0
        start_time = time.time()
        
        with open(train_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    self.logger.info(f"Processed {i} lines...")
                processed = cleaner.clean_text(line)
                if processed:
                    processed_texts.append(processed)
                    total_tokens += len(processed.split())
        
        processing_time = time.time() - start_time
        self.logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Total tokens: {total_tokens}")
        self.logger.info(f"Sample of processed texts: {processed_texts[:5]}")

        return processed_texts

    def train_tokenizer(self, texts):
        """Train the BPE tokenizer on processed data."""
        self.logger.info("=== Starting Tokenizer Training ===")
        
        tokenizer = BPETokenizer(
            vocab_size=self.config['model']['vocab_size'],
            min_frequency=self.config['tokenizer']['min_frequency']
        )
        
        start_time = time.time()
        self.logger.info("Training tokenizer...")
        tokenizer.train(texts)
        training_time = time.time() - start_time
        
        # Log tokenizer statistics
        self.logger.info(f"Tokenizer training completed in {training_time:.2f} seconds")
        self.logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
        
        # Log sample tokenization
        sample_text = texts[0]
        tokens = tokenizer.encode(sample_text)
        self.logger.info(f"Sample tokenization:")
        self.logger.info(f"Original text: {sample_text}")
        self.logger.info(f"Tokens: {tokens}")
        self.logger.info(f"Decoded text: {tokenizer.decode(tokens)}")
        
        # Save tokenizer
        tokenizer_path = self.experiment_dir / 'tokenizer' / 'tokenizer.json'
        tokenizer.save(str(tokenizer_path))
        self.logger.info(f"Saved tokenizer to {tokenizer_path}")
        
        return tokenizer

    def setup_model(self):
        """Initialize the transformer model."""
        self.logger.info("=== Setting up Model ===")
        
        # Create model config
        model_config = TransformerConfig(**self.config['model'])
        self.logger.info(f"Model configuration: {json.dumps(model_config.__dict__, indent=2)}")
        
        # Initialize model
        model = TransformerLightningModule(
            config=model_config,
            **self.config['training']
        )
        
        # Log model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_stats = {
            "Total Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Model Size (MB)": f"{total_params * 4 / (1024**2):.2f}",
            "Architecture": str(model)
        }
        self.logger.info(f"Model Statistics: {json.dumps(model_stats, indent=2)}")
        
        return model

    def run(self):
        """Execute the full training pipeline."""
        self.logger.info("=== Starting Training Pipeline ===")
        self.log_system_info()
        
        try:
            # Process data
            processed_texts = self.process_data()
            
            # Train tokenizer
            tokenizer = self.train_tokenizer(processed_texts)
            
            # Setup data module
            data_module = LMDataModule(
                train_path=self.config['data']['train_path'],
                val_path=self.config['data']['val_path'],
                tokenizer=tokenizer,
                **self.config['data']['dataloader']
            )
            
            # Setup model
            model = self.setup_model()
            
            # Train model
            self.logger.info("=== Starting Model Training ===")
            model.train()
            
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