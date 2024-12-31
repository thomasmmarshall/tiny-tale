import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Union, Dict
import json

from .architecture.transformer import Transformer, TransformerConfig
from .training.trainer import TransformerLightningModule

class ModelInference:
    """A wrapper class for easy model inference."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the inference wrapper.
        
        Args:
            model_path: Path to the saved model checkpoint directory
            device: Device to run inference on ('cpu' or 'mps' on Apple Silicon)
            dtype: Data type for model weights (float32 for MPS compatibility)
        """
        self.device = device
        self.dtype = dtype
        self.model_path = Path(model_path)
        
        # Load model configuration
        config_path = self.model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        
        # Create model configuration
        self.config = TransformerConfig(
            vocab_size=config_dict.get("vocab_size", 50257),
            hidden_size=config_dict.get("hidden_size", 768),
            num_hidden_layers=config_dict.get("num_layers", 12),
            num_attention_heads=config_dict.get("num_heads", 12),
            intermediate_size=config_dict.get("intermediate_size", 3072),
            max_position_embeddings=config_dict.get("max_seq_length", 1024),
            hidden_dropout_prob=config_dict.get("hidden_dropout", 0.1),
            attention_dropout_prob=config_dict.get("attention_dropout", 0.1),
            use_rope=not config_dict.get("disable_rope", False),
            bias=not config_dict.get("disable_bias", False)
        )
        
        # Initialize model
        self.model = TransformerLightningModule.load_from_checkpoint(
            checkpoint_path=str(self.model_path / "checkpoints" / "final_model.ckpt"),
            config=self.config
        )
        
        # Prepare model for inference
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        
        # Disable dropout for inference
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        min_length: int = 20,
        repetition_penalty: float = 1.5,
        no_repeat_ngram_size: int = 4,
    ) -> List[List[int]]:
        """Generate text using the model with enhanced repetition prevention."""
        # Move input to device and convert to correct dtype
        input_ids = input_ids.to(device=self.device)
        
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]
        
        # Expand input for multiple sequences per sample
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            
        # Initialize generation output with input sequence
        generated = input_ids
        
        # Create attention mask for the current sequence
        attention_mask = torch.ones_like(generated, dtype=torch.long, device=self.device)
        
        # Keep track of which sequences have finished
        unfinished_sequences = torch.ones(generated.shape[0], dtype=torch.long, device=self.device)
        
        # Create a set to store generated n-grams for each sequence
        if no_repeat_ngram_size > 0:
            banned_ngrams = [set() for _ in range(generated.shape[0])]
        
        while (generated.shape[1] < max_length) and (unfinished_sequences.sum() > 0):
            # Get model predictions
            model_outputs = self.model.model(
                input_ids=generated,
                attention_mask=attention_mask,
            )
            
            # Get next token logits
            next_token_logits = model_outputs[:, -1, :].float()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(generated[i].tolist()):
                        # Apply exponential penalty based on frequency
                        freq = (generated[i] == previous_token).sum().item()
                        next_token_logits[i, previous_token] /= (repetition_penalty ** freq)
            
            # Apply n-gram blocking
            if no_repeat_ngram_size > 0:
                for batch_idx in range(generated.shape[0]):
                    # Get the last n-1 tokens
                    prev_tokens = generated[batch_idx, -(no_repeat_ngram_size-1):].tolist()
                    
                    # For each potential next token
                    for token_idx in range(next_token_logits.shape[-1]):
                        # Check if completing this n-gram would create a repeat
                        ngram = tuple(prev_tokens + [token_idx])
                        if ngram in banned_ngrams[batch_idx]:
                            next_token_logits[batch_idx, token_idx] = -float('inf')
            
            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            else:
                # For temperature = 0, use greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for batch_idx in range(next_token_logits.shape[0]):
                    indices_to_remove = sorted_indices_to_remove[batch_idx].scatter(
                        0, sorted_indices[batch_idx], sorted_indices_to_remove[batch_idx]
                    )
                    next_token_logits[batch_idx][indices_to_remove] = -float('inf')
            
            # Ensure at least one token has a valid probability
            if (next_token_logits == -float('inf')).all(dim=-1).any():
                problematic_batches = (next_token_logits == -float('inf')).all(dim=-1)
                next_token_logits[problematic_batches, :] = 0
                # Allow only the top token for problematic batches
                next_token_logits[problematic_batches, :] = -float('inf')
                top_tokens = torch.topk(model_outputs[problematic_batches, -1, :], k=1)[1]
                next_token_logits[problematic_batches, top_tokens] = 0
            
            # Sample next tokens
            probs = F.softmax(next_token_logits, dim=-1)
            if temperature == 0:
                next_tokens = torch.argmax(probs, dim=-1)
            else:
                next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Update n-gram sets
            if no_repeat_ngram_size > 0:
                for batch_idx in range(generated.shape[0]):
                    prev_tokens = generated[batch_idx, -no_repeat_ngram_size+1:].tolist()
                    ngram = tuple(prev_tokens + [next_tokens[batch_idx].item()])
                    banned_ngrams[batch_idx].add(ngram)
            
            # Only replace token in unfinished sequences
            tokens_to_add = next_tokens * unfinished_sequences.unsqueeze(-1)
            
            # Append next tokens to generated sequence
            generated = torch.cat((generated, tokens_to_add), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(tokens_to_add)), dim=1)
            
            # Update which sequences are finished
            if eos_token_id is not None and generated.shape[1] >= min_length:
                unfinished_sequences = unfinished_sequences.mul(
                    (tokens_to_add.squeeze(-1) != eos_token_id).long()
                )
                
                # Early stopping if all sequences are finished
                if unfinished_sequences.max() == 0:
                    break
        
        return generated.tolist()
    
    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get raw logits from the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            Model logits [batch_size, seq_len, vocab_size]
        """
        input_ids = input_ids.to(device=self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
            
        return self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    def to(self, device: str) -> 'ModelInference':
        """Move model to specified device.
        
        Args:
            device: Device to move model to ('cpu' or 'cuda')
        
        Returns:
            Self for chaining
        """
        self.device = device
        self.model.to(device)
        return self
    
    def half(self) -> 'ModelInference':
        """Convert model to half precision (float16).
        
        Returns:
            Self for chaining
        """
        self.dtype = torch.float16
        self.model.half()
        return self 