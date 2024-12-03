# src/model/training/trainer.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from ..architecture.transformer import Transformer, TransformerConfig

class TransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: TransformerConfig,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        grad_clip_val: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize transformer model
        self.model = Transformer(config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip_val = grad_clip_val

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch)
        
        # Log training metrics
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        
        # Calculate and log perplexity
        perplexity = torch.exp(loss)
        self.log('train_perplexity', perplexity, on_step=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch)
        
        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate and log perplexity
        perplexity = torch.exp(loss)
        self.log('val_perplexity', perplexity, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        # Separate weight decay and non-weight decay parameters
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

