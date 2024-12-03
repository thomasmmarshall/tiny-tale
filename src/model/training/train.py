
# src/model/training/train.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import argparse
from pathlib import Path

from ..architecture.transformer import TransformerConfig
from .trainer import TransformerLightningModule
from ...data.preprocessing.data_module import LMDataModule

def train(args):
    # Initialize config
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_length,
    )
    
    # Initialize data module
    data_module = LMDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        tokenizer=None,  # You'll need to pass your tokenizer here
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = TransformerLightningModule(
        config=config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps
    )
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.run_name,
        log_model=True
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator='auto',
        devices='auto',
        strategy='ddp' if args.num_gpus > 1 else None,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.grad_clip_val,
        val_check_interval=args.val_check_interval,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, data_module)


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    
    # Training arguments
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--grad_clip_val', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_check_interval', type=int, default=1000)
    
    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default='transformer-lm')
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()