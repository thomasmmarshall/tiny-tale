# tiny-tale large language models

A repository for my own in depth investigation into large language models and all that surround them. The name of this repository should be read as a bilingual hyphenation, but also works in English only. This repository contains an educational implementation of a Large Language Model (LLM) training pipeline. The project aims to provide a clear understanding of modern LLM architectures while being practical enough to run on consumer hardware.

## Technical Overview

### Core Architecture Components

#### Position Embeddings

Position embeddings are crucial for transformers to understand token ordering since self-attention is permutation-invariant. We implement both absolute positional encodings (sinusoidal or learned) and relative positional embeddings. Modern architectures often use RoPE (Rotary Position Embeddings), which encodes positional information through rotation matrices in the complex plane, allowing better extrapolation to unseen sequence lengths.

#### Attention Mechanisms

Our implementation includes standard scaled dot-product attention with optional optimizations. Flash Attention reduces memory usage from O(n²) to O(n) by computing attention in tiles that fit in fast memory (GPU SRAM), making it crucial for training with limited resources. We also implement multi-head attention with parallel attention heads that can learn different relationship patterns in the data.

#### Normalization and Regularization

Layer normalization is applied before the self-attention and feed-forward layers (pre-norm) following modern architectures, which provides more stable training than post-norm. We implement both traditional LayerNorm and RMSNorm (used in LLaMA), which only normalizes by standard deviation. Dropout is applied to attention weights and feed-forward layers.

### Training Infrastructure

#### Mixed Precision Training

To maximize our limited computational resources, we implement mixed precision training using fp16 or bf16 formats. This not only reduces memory usage but also leverages modern hardware acceleration. We maintain a master copy of weights in fp32 for stability while performing forward and backward passes in lower precision.

#### Gradient Checkpointing

This technique trades compute for memory by discarding intermediate activations during the forward pass and recomputing them during backpropagation. While this increases training time by ~20%, it significantly reduces memory usage, making it possible to train larger models or use bigger batch sizes on limited hardware.

#### Memory-Efficient Optimizations

We implement several memory optimizations:

- Selective activation caching for frequently needed intermediate results
- Gradient accumulation for effective larger batch sizes
- Efficient attention implementations that avoid materializing the full attention matrix
- Layer parameter sharing options for reducing model size

### Tokenization

#### Byte-Pair Encoding (BPE)

Our BPE implementation starts with a character vocabulary and iteratively merges the most frequent adjacent pairs. We include both regular BPE and GPT-style BPE with byte-level fallback to handle any Unicode text without unknown tokens. The implementation includes:

- Efficient token merge operations
- Vocabulary management with special tokens
- Fast tokenization using a prefix tree (trie) data structure
- Detokenization with byte fallback

### Fine-tuning Approaches

#### RLHF (Reinforcement Learning from Human Feedback)

Our RLHF implementation follows the InstructGPT approach:

1. Supervised fine-tuning on instruction data
2. Reward model training on human preferences
3. PPO optimization against the reward model

The PPO implementation includes:

- Value function for advantage estimation
- KL penalty to prevent divergence from reference model
- Adaptive KL coefficient balancing

#### Parameter-Efficient Fine-tuning

We implement several efficient fine-tuning methods:

- LoRA (Low-Rank Adaptation): Adds trainable low-rank matrices to frozen model weights
- Prompt Tuning: Learns continuous prompt embeddings while keeping the model frozen
- QLoRA: Quantized LoRA for even more memory efficiency

### Training Data Management

The data pipeline includes:

- Efficient streaming dataset implementation to handle large corpora
- Dynamic batching with length-based binning
- Preprocessing with configurable cleaning steps
- Data mixing for multiple sources
- Validation split management

For initial training, we recommend using WikiText-2 or TinyStories, which provide enough complexity for meaningful learning while fitting in memory. The pipeline can scale to larger datasets through streaming and efficient data loading.

### Recommended Model Configurations

For 8GB RAM constraints:

```python
model_config = {
    'vocab_size': 8192,        # Smaller vocabulary for efficiency
    'hidden_size': 384,        # Balanced for expressivity vs memory
    'num_layers': 6,           # Enough depth for non-trivial learning
    'num_heads': 6,            # Multiple attention patterns
    'max_seq_length': 256,     # Manageable context size
    'batch_size': 8,           # Adjusted based on memory
    'learning_rate': 3e-4,     # Standard for transformer training
    'warmup_steps': 1000,      # Gradual learning rate warmup
    'gradient_checkpointing': True,  # Memory optimization
    'mixed_precision': 'fp16'   # Memory and speed optimization
}
```

## Getting Started

[Installation and setup instructions to be added]

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# For M2 Mac, ensure you're using the correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

```bash
# Create datasets
python scripts/create_wikitext2_dataset.py
```

## References

- Attention Is All You Need (Vaswani et al., 2017)
- GPT-3: Language Models are Few-Shot Learners (Brown et al., 2020)
- InstructGPT: Training Language Models to Follow Instructions (Ouyang et al., 2022)
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
