import argparse
import torch
from pathlib import Path
from typing import List, Optional
import json

from model.inference import ModelInference
from data.tokenization.bpe_tokenizer import BPETokenizer

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run inference with a trained transformer model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model checkpoint directory'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to the tokenizer directory containing the saved tokenizer'
    )
    
    # Input handling
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_text',
        type=str,
        help='Text input for generation'
    )
    input_group.add_argument(
        '--input_file',
        type=str,
        help='Path to file containing input text (one prompt per line)'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of generated sequence'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Number of highest probability tokens to keep for top-k sampling'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Cumulative probability for nucleus sampling'
    )
    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=1,
        help='Number of sequences to generate per input'
    )
    
    # Output handling
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to save generated text (if not provided, prints to stdout)'
    )
    
    # Hardware options
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--use_half_precision',
        action='store_true',
        help='Use half precision (float16) for faster inference on GPU'
    )
    
    return parser

def load_inputs(args) -> List[str]:
    """Load input prompts from either text or file."""
    if args.input_text:
        return [args.input_text]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def save_outputs(generated_texts: List[str], args):
    """Save or print generated texts."""
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in generated_texts:
                f.write(text + '\n')
        print(f"Generated texts saved to {output_path}")
    else:
        for i, text in enumerate(generated_texts, 1):
            print(f"\nGeneration {i}:")
            print("=" * 40)
            print(text)
            print("=" * 40)

def main():
    args = setup_argument_parser().parse_args()
    
    print(f"Loading model from {args.model_path}")
    model = ModelInference(args.model_path, device=args.device)
    
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = BPETokenizer.load(args.tokenizer_path)
    
    input_texts = load_inputs(args)
    print(f"Loaded {len(input_texts)} input prompts\n")
    
    generated_texts = []
    for i, text in enumerate(input_texts, 1):
        print(f"\nProcessing input {i}/{len(input_texts)}")
        print(f"Input text: {text!r}")
        
        # Convert to lowercase since our tokenizer is trained on lowercase text
        text = text.lower()
        print(f"Preprocessed text: {text!r}")
        
        # Encode input text with special tokens
        print("\nTokenization step:")
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=None,  # Don't truncate or pad the input
            truncation=False,
            padding=False,
            return_attention_mask=True
        )
        
        # Debug tokenization
        input_tokens = [tokenizer.vocab.lookup_token(id) for id in encoded['input_ids']]
        print(f"Tokenized input (with special tokens):")
        print(f"Token IDs: {encoded['input_ids']}")
        print(f"Tokens: {input_tokens}")
        print(f"Attention mask: {encoded['attention_mask']}")
        
        input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long)
        attention_mask = torch.tensor([encoded['attention_mask']], dtype=torch.long)
        
        print("\nGeneration step:")
        print(f"Input shape: {input_ids.shape}")
        print(f"Using temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        
        # Generate
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            min_length=20,  # Force at least some generation
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print(f"\nGenerated sequences: {len(output_ids)}")
        
        # Decode all sequences
        for j, sequence in enumerate(output_ids, 1):
            print(f"\nSequence {j} details:")
            print(f"Output IDs: {sequence}")
            output_tokens = [tokenizer.vocab.lookup_token(id) for id in sequence]
            print(f"Output tokens: {output_tokens}")
            
            decoded_text = tokenizer.decode(sequence[len(encoded['input_ids']):], skip_special_tokens=True)
            print(f"\nGeneration {j}:")
            print("=" * 40)
            print(decoded_text.strip())
            print("=" * 40)
            generated_texts.append(decoded_text)
    
    if args.output_file:
        save_outputs(generated_texts, args)

if __name__ == "__main__":
    main() 