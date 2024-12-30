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
        # Debug info
        print(f"Training step {batch_idx}, device: {next(self.parameters()).device}")
        print(f"Batch size: {batch['input_ids'].shape}, device: {batch['input_ids'].device}")
        
        # Ensure all tensors are on the correct device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass with gradient computation
        with torch.set_grad_enabled(True):
            loss = self.model(**batch)
            
            # Check if loss is valid and requires grad
            if not torch.isfinite(loss):
                print(f"Warning: Loss is not finite: {loss}")
                loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
            
            # Ensure loss requires grad
            if not loss.requires_grad:
                print("Warning: Loss does not require grad")
                loss = loss.requires_grad_(True)
        
        # Log training metrics
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Calculate and log perplexity
        perplexity = torch.exp(loss)
        self.log('train_perplexity', perplexity.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        self.log('learning_rate', opt.param_groups[0]['lr'], on_step=True, prog_bar=True)
        
        # Debug gradients
        if batch_idx % 10 == 0:  # Every 10 steps
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient norm: {total_norm}")
            
            # Print parameter statistics
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    print(f"{name}: grad_norm={grad_norm:.3f}, param_norm={param_norm:.3f}")
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Debug info
        print(f"Validation step {batch_idx}, device: {next(self.parameters()).device}")
        
        # Ensure all tensors are on the correct device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        loss = self.model(**batch)
        
        # Log validation metrics
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Calculate and log perplexity
        perplexity = torch.exp(loss)
        self.log('val_perplexity', perplexity.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        # Debug info
        print("Configuring optimizer...")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Model device: {next(self.parameters()).device}")
        
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
        # Debug gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)
        if not torch.isfinite(total_norm):
            print(f"Warning: Gradient norm is not finite: {total_norm}")
        else:
            print(f"Clipped gradient norm: {total_norm}")

