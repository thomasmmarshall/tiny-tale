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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        """Initialize the inference wrapper.
        
        Args:
            model_path: Path to the saved model checkpoint directory
            device: Device to run inference on ('cpu' or 'cuda')
            dtype: Data type for model weights (float16 for efficiency on GPU)
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
    ) -> List[List[int]]:
        """Generate text using the model."""
        # Move input to device and convert to correct dtype
        input_ids = input_ids.to(device=self.device)
        print(f"\nGeneration details:")
        print(f"Input device: {input_ids.device}, Input dtype: {input_ids.dtype}")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]
        
        # Expand input for multiple sequences per sample
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            print(f"Expanded input shape for {num_return_sequences} sequences: {input_ids.shape}")
            
        # Initialize generation output with input sequence
        generated = input_ids
        
        # Create attention mask for the current sequence
        attention_mask = torch.ones_like(generated, dtype=torch.long, device=self.device)
        
        # Keep track of which sequences have finished
        unfinished_sequences = torch.ones(generated.shape[0], dtype=torch.long, device=self.device)
        
        # Track repetition
        repetition_penalty = 1.2
        
        # Generate tokens one at a time
        print(f"\nStarting generation loop from length {input_length} to {max_length}")
        
        while (generated.shape[1] < max_length) and (unfinished_sequences.sum() > 0):
            # Get model predictions
            model_outputs = self.model.model(
                input_ids=generated,
                attention_mask=attention_mask,
            )
            
            # Get next token logits
            next_token_logits = model_outputs[:, -1, :]
            
            # Create mask for special tokens
            special_tokens_mask = torch.zeros((next_token_logits.shape[-1],), device=self.device)
            if generated.shape[1] < min_length:
                special_tokens_mask[eos_token_id] = -float('inf')  # Prevent EOS before min_length
            
            # Never generate BOS, PAD, or repeated EOS tokens
            special_tokens_mask[[pad_token_id, 2]] = -float('inf')  # 2 is BOS token ID
            if generated.shape[1] > 0 and generated[:, -1].item() == eos_token_id:
                special_tokens_mask[eos_token_id] = -float('inf')  # Prevent multiple EOS tokens
            
            next_token_logits = next_token_logits + special_tokens_mask
            
            # Enhanced repetition prevention with n-gram blocking
            for i in range(batch_size):
                # Get sequence history for n-gram checking
                seq_history = generated[i].tolist()
                
                # Check for repeating n-grams (n=3,4,5)
                for n in [3, 4, 5]:
                    if len(seq_history) >= n * 2:
                        # Get the last n tokens
                        last_ngram = seq_history[-n:]
                        # Check if this n-gram appears in the recent history
                        for j in range(len(seq_history) - n):
                            if seq_history[j:j + n] == last_ngram:
                                # If we find a repeat, heavily penalize the next token that would continue the pattern
                                if j + n < len(seq_history):
                                    next_token_logits[i, seq_history[j + n]] = -float('inf')
                
                # Get the last few tokens to check for repetition
                last_tokens = seq_history[-15:]  # Increased window to 15 tokens
                for previous_token in set(last_tokens):
                    count = last_tokens.count(previous_token)
                    # Completely prevent immediate repetition
                    if previous_token == seq_history[-1]:
                        next_token_logits[i, previous_token] = -float('inf')
                    # Apply exponential penalty for frequent tokens
                    elif count > 1:
                        penalty = repetition_penalty ** count
                        next_token_logits[i, previous_token] /= penalty
            
            # Apply temperature
            if temperature == 0:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Apply temperature scaling
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Ensure valid probabilities and sample
                if torch.max(next_token_logits) == -float('inf'):
                    # If all tokens are masked, allow the top 10 tokens from the original logits
                    top_logits, top_indices = torch.topk(model_outputs[:, -1, :], k=10)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_indices, F.softmax(top_logits / temperature, dim=-1))
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Only replace token in unfinished sequences
            tokens_to_add = next_tokens * unfinished_sequences.unsqueeze(-1)
            
            # Append next tokens to generated sequence
            generated = torch.cat((generated, tokens_to_add), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(tokens_to_add)), dim=1)
            
            # Update which sequences are finished
            if eos_token_id is not None and generated.shape[1] >= min_length:
                unfinished_sequences = unfinished_sequences.mul((tokens_to_add.squeeze(-1) != eos_token_id).long())
                
                # Early stopping if all sequences are finished
                if unfinished_sequences.max() == 0:
                    break
        
        print(f"\nGeneration finished at length {generated.shape[1]}")
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