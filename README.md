# tiny-tale large language models

_Note_: this is a livingscratchpad, and I am generally not that interested in following any Git or any other best practices here. For transparency, this repository is produced with assistance from various AI tooling (personally I quite like Cursor and Claude Sonnet 3.5).

## Latest Update Notes

- 2025-01-01: Decided it is not possible to get sensible outputs from a 10M parameter model training on ~2M tokens.
- 2025-01-04: Heuristics suggest good results can be obtained with 20 tokens per trainable parameter. So I'm trying to use wikitext103 as a dataset. However, it is becoming increasingly clear why scaling is so hard and is such a big deal in the LLM world.
- 2025-01-10: I've decided that this it is likely not possible to get workable results with a M2 Macbook Air. Might update this again later if something changes. I did however have a lot of fun using Weights & Biases to conduct this testing. 

## Project Overview

A repository for my own in depth investigation into large language models and all that surround them; and this has to run on my base model 8GB M2 Macbook Air. The name of this repository should be read as a bilingual hyphenation, but also works in English only. This repository contains an educational implementation of a Large Language Model (LLM) training pipeline. The project aims to provide a clear understanding of modern LLM architectures while being practical enough to run on consumer hardware.

In the days before the onslaught of ChatGPT on the world, I was a masters student at Stellenbosch University and I had just taken a class about advanced NLP. Back in those days, large language models were these quirky little guys who just shouted random stuff at you. At the time, I was more interested in applying other areas of NLP theory to accomplish natural language processing tasks. As a student I started a little project to do LLMs from scratch; obviously the landscape looked very different back then. Alas I was not able to finish it. I am now able to finish it, and I am starting again and sharing it with the world. This is a living project, and I will be updating it as I learn more and have time to work on it.

## On the Use of AI Tooling

On the use of AI tooling, over the past 12 months I have seen various LLM-based tools come and go. At first, GPT-3.5 was really not that useful for most things, especially coding. Then came Claude Sonnet 3.5, which still largely knocks it out of the park. During the year we also got Cursor, which has just become more interesting over time. Cursor + Claude Sonnet 3.5 has become my go-to tool for most things. However, I do have some concerns. I do not contest the usefulness of these tools when used properly; they are _immensely_ useful. What does concern me is the possible induction of a slop cycle feedback loop and the blunting of a knife many of us have spent years sharpening. I suppose only time will tell what fresh hell is to be unleashed at scale by widespread use of AI tooling. It is in the interest of these concerns that I am sharing the fact that I am using AI tooling to write this project. It saves me a lot of time, but requires that I be more careful about what I am doing because ultimately the goal of this project is to learn. We must not allow tools of great power to prevent us from learning.

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

````bash
# Create datasets
python scripts/create_wikitext2_dataset.py
```

# Run training pipeline
```bash
python src/pipeline.py configs/pipeline_config.yaml experiment_name
```

# Run inference pipeline
```bash
python src/inference_pipeline.py
	--model_path experiments/experiment_name
	--tokenizer_path experiments/experiment_name/tokenizer/tokenizer.json
	--input_text "What is your name?"
	--max_length 100 --temperature 0.9 --top_k 50 --top_p 0.92
```

## References
...
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- ... etc, will get to it.
````
