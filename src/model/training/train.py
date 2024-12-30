# src/model/training/train.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    StochasticWeightAveraging,
    GradientAccumulationScheduler
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
import torch
import argparse
from pathlib import Path
import os
import json
from typing import Optional, Dict, Any

from ..architecture.transformer import TransformerConfig
from .trainer import TransformerLightningModule
from ...data.preprocessing.data_module import LMDataModule

def setup_training_environment(args):
    """Setup training environment with proper logging and device configuration."""
    # Set PyTorch environment variables for performance
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if args.compile_model:
        if not hasattr(torch, 'compile'):
            print("Warning: PyTorch 2.0 compile not available, falling back to eager mode")
            args.compile_model = False
    
    # Create experiment directory
    experiment_dir = Path(args.output_dir) / args.run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full config
    config_path = experiment_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return experiment_dir

def create_callbacks(args, experiment_dir: Path) -> list:
    """Create training callbacks for monitoring and checkpointing."""
    callbacks = []
    
    # Checkpoint saving
    callbacks.append(ModelCheckpoint(
        dirpath=experiment_dir / 'checkpoints',
        filename='{epoch}-{step}-{val_loss:.3f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_train_steps=args.save_every_n_steps
    ))
    
    # Learning rate monitoring
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Early stopping
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            mode='min'
        ))
    
    # Gradient accumulation scheduling
    if args.gradient_accumulation_steps_schedule is not None:
        accumulation_schedule = dict(enumerate(args.gradient_accumulation_steps_schedule))
        callbacks.append(GradientAccumulationScheduler(scheduling=accumulation_schedule))
    
    # Stochastic Weight Averaging
    if args.use_swa:
        callbacks.append(StochasticWeightAveraging(
            swa_epoch_start=0.8,
            swa_lrs=args.learning_rate * 0.1
        ))
    
    return callbacks

def train(args):
    """Main training function with enhanced configuration and monitoring."""
    experiment_dir = setup_training_environment(args)
    
    # Initialize config
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_length,
        hidden_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        use_rope=not args.disable_rope,
        bias=not args.disable_bias
    )
    
    # Save model config
    config.save(experiment_dir / 'model_config.json')
    
    # Initialize data module with proper error handling
    try:
        data_module = LMDataModule(
            train_path=args.train_path,
            val_path=args.val_path,
            tokenizer=None,  # Will be initialized in data module
            batch_size=args.batch_size,
            max_length=args.max_seq_length,
            num_workers=args.num_workers,
            prefetch_factor=2
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize data module: {str(e)}")
    
    # Initialize model with gradient checkpointing if enabled
    model = TransformerLightningModule(
        config=config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        use_gradient_checkpointing=args.gradient_checkpointing
    )
    
    # Compile model if enabled (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.run_name,
        log_model=True,
        save_dir=str(experiment_dir)
    )
    
    # Create training callbacks
    callbacks = create_callbacks(args, experiment_dir)
    
    # Initialize training strategy
    if args.use_deepspeed:
        strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=args.offload_optimizer,
            offload_parameters=args.offload_parameters,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
        )
    else:
        strategy = 'ddp' if args.num_gpus > 1 else 'auto'
    
    # Initialize trainer with all configurations
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=args.num_gpus if args.num_gpus > 0 else 'auto',
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.grad_clip_val,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=args.detect_anomaly,
    )
    
    # Train
    try:
        trainer.fit(
            model,
            data_module,
            ckpt_path=args.resume_from_checkpoint
        )
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--hidden_dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--disable_rope', action='store_true')
    parser.add_argument('--disable_bias', action='store_true')
    
    # Training arguments
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--grad_clip_val', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps_schedule', type=int, nargs='+')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=-1)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--resume_from_checkpoint', type=str)
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--save_every_n_steps', type=int, default=1000)
    parser.add_argument('--val_check_interval', type=float, default=0.5)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    
    # Advanced training options
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--compile_model', action='store_true')
    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--offload_optimizer', action='store_true')
    parser.add_argument('--offload_parameters', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=0)
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--detect_anomaly', action='store_true')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()