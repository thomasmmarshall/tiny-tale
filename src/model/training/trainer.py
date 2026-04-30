# src/model/training/trainer.py
import math

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
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
        grad_clip_val: float = 1.0,
        use_gradient_checkpointing: bool = False,
        min_learning_rate: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Transformer(config)
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip_val = grad_clip_val
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.min_learning_rate = min_learning_rate or learning_rate * 0.1

    def forward(self, batch):
        return self.model(**batch)

    def _model_step(self, batch):
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        loss = self._model_step(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_idx}: {loss}")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        perplexity = torch.exp(loss.detach().clamp(max=20))
        self.log('train_perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        self.log('learning_rate', opt.param_groups[0]['lr'], on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._model_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        perplexity = torch.exp(loss.detach().clamp(max=20))
        self.log('val_perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'norm', 'embedding']
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

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step + 1) / max(1, self.warmup_steps)
            progress = (current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            min_ratio = self.min_learning_rate / self.learning_rate
            return max(min_ratio, 0.5 * (1.0 + math.cos(progress * math.pi)))

        scheduler = LambdaLR(optimizer, lr_lambda)

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
        # else:
        #     print(f"Clipped gradient norm: {total_norm}")

