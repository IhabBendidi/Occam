#!/usr/bin/env python3
"""
Main inference script for Occam with h5ad data support

Usage:
    python occam_embed.py --model_dir ./trained_model --input_data data.h5ad --output results

Example with multiple files:
    python occam_embed.py \
        --model_dir ./trained_model \
        --input_data file1.h5ad file2.h5ad file3.h5ad \
        --output_dir ./inference_results \
        --batch_size 1024 \
        --gene_col gene_symbol
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from occam.inference.embed import load_trained_model, infer_from_h5ad, batch_inference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with trained Occam model on h5ad data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model and preprocessor')
    parser.add_argument('--checkpoint', type=str,
                        help='Optional checkpoint path or filename (absolute or relative to model_dir). If omitted, defaults to best_model.pth, then final_model.pth, then model_checkpoint.pth')
    parser.add_argument('--preprocessor_path', type=str,
                        help='Optional preprocessor path or filename (absolute or relative to model_dir). If omitted, loader will search next to checkpoint, parent, then cache, then debug fallback')
    
    # Data arguments
    parser.add_argument('--input_data', type=str, nargs='+', required=True,
                        help='Path(s) to input h5ad file(s)')
    parser.add_argument('--gene_col', type=str, default='auto',
                        help='Gene name column in adata.var; use "auto" to pick best-overlap column')
    
    # Output arguments
    parser.add_argument('--output', type=str, 
                        help='Output path prefix (for single file) or directory (for multiple files)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory (alternative to --output)')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for inference')
    parser.add_argument('--save_processed', action='store_true',
                        help='Save processed expression data along with embeddings')
    parser.add_argument('--eval-after-inference', action='store_true',
                        help='If set, run evaluation on inferred embeddings using training metrics')
    parser.add_argument('--eval-frequency', type=int, default=0,
                        help='For multi-file inference, run evaluation every N files (0 means only once at the end)')
    parser.add_argument('--eval-seed', type=int, default=42,
                        help='Random seed for perturbation consistency null distribution sampling')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Check model directory exists
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    # Check for model files / specified checkpoint
    if getattr(args, 'checkpoint', None):
        candidate = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(args.model_dir, args.checkpoint)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Specified checkpoint not found: {candidate}")
        # Normalize to absolute path to avoid double-joining downstream
        args.checkpoint = os.path.abspath(candidate)
    else:
        model_files = ['best_model.pth', 'final_model.pth', 'model_checkpoint.pth']
        model_exists = any(os.path.exists(os.path.join(args.model_dir, f)) for f in model_files)
        if not model_exists:
            raise FileNotFoundError(f"No model checkpoint found in {args.model_dir}")
    
    # Resolve preprocessor path if explicitly provided; otherwise leave None
    # (the inference pipeline will build a CheckpointPreprocessor from checkpoint data)
    if getattr(args, 'preprocessor_path', None):
        candidate = args.preprocessor_path if os.path.isabs(args.preprocessor_path) else os.path.join(args.model_dir, args.preprocessor_path)
        if os.path.exists(candidate):
            args.preprocessor_path = os.path.abspath(candidate)
    
    # Check input files
    for input_file in args.input_data:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    return args


def load_model_info(model_dir):
    """Load and display model information."""
    try:
        # Try to load training config
        config_files = ['run_config.json', 'training_config.pkl']
        model_info = {}
        
        for config_file in config_files:
            config_path = os.path.join(model_dir, config_file)
            if os.path.exists(config_path):
                if config_file.endswith('.json'):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'config' in config:
                            model_info = config['config']
                        break
                # Could add pickle loading here if needed
        
        if model_info:
            print(f"Model Information:")
            print(f"  Model size: {model_info.get('model_size', 'unknown')}")
            print(f"  Hidden size: {model_info.get('hidden_size', 'unknown')}")
            print(f"  Number of genes: {model_info.get('n_top_genes', 'unknown')}")
            print(f"  Mask probability: {model_info.get('mask_probability', 'unknown')}")
        
    except Exception as e:
        print(f"Could not load model info: {e}")


def main():
    args = parse_args()
    args = validate_args(args)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("Occam Inference Pipeline")
    print("=" * 50)
    print(f"Model directory: {args.model_dir}")
    if getattr(args, 'checkpoint', None):
        print(f"Checkpoint: {args.checkpoint}")
    
    # Load and display model info
    load_model_info(args.model_dir)
    
    print(f"Gene column: {args.gene_col}")
    print(f"Input files: {len(args.input_data)} file(s)")
    for i, file in enumerate(args.input_data, 1):
        print(f"  {i}. {file}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)
    
    try:
        if len(args.input_data) == 1 and args.output:
            # Single file mode with specific output path
            input_file = args.input_data[0]
            print(f"\nProcessing single file: {input_file}")
            
            gene_arg = None if (args.gene_col is None or (isinstance(args.gene_col, str) and args.gene_col.lower() == 'auto')) else args.gene_col

            result = infer_from_h5ad(
                adata_path=input_file,
                model_dir=args.model_dir,
                output_path=args.output,
                gene_name_col=gene_arg,
                batch_size=args.batch_size,
                device=args.device,
                checkpoint_path=args.checkpoint,
                preprocessor_path=args.preprocessor_path,
                eval_after_inference=args.eval_after_inference,
                eval_frequency=args.eval_frequency,
                eval_seed=args.eval_seed
            )
            
            print(f"\nInference completed!")
            # Handle both chunked mode (returns tuple) and standard mode (returns array)
            if isinstance(result, tuple):
                npy_path, h5_path = result
                # Load shape from the saved file
                embeddings = np.load(npy_path)
                print(f"Embeddings shape: {embeddings.shape}")
                print(f"Results saved to:")
                print(f"  NPY: {npy_path}")
                print(f"  HDF5: {h5_path}")
                del embeddings  # Free memory
            else:
                embeddings = result
                print(f"Embeddings shape: {embeddings.shape}")
                print(f"Results saved with prefix: {args.output}")
            
        else:
            # Multiple files mode or single file with directory output
            output_dir = args.output_dir if args.output_dir else args.output
            
            print(f"\nProcessing {len(args.input_data)} file(s)")
            print(f"Output directory: {output_dir}")
            
            gene_arg = None if (args.gene_col is None or (isinstance(args.gene_col, str) and args.gene_col.lower() == 'auto')) else args.gene_col

            results = batch_inference(
                adata_paths=args.input_data,
                model_dir=args.model_dir,
                output_dir=output_dir,
                gene_name_col=gene_arg,
                batch_size=args.batch_size,
                device=args.device,
                checkpoint_path=args.checkpoint,
                preprocessor_path=args.preprocessor_path,
                eval_after_inference=args.eval_after_inference,
                eval_frequency=args.eval_frequency,
                eval_seed=args.eval_seed
            )
            
            print(f"\nBatch inference completed!")
            print(f"Processed {len(results)} files")
            
            # Create summary
            summary_data = []
            for filename, embeddings in results.items():
                summary_data.append({
                    'filename': filename,
                    'n_cells': embeddings.shape[0],
                    'embedding_dim': embeddings.shape[1],
                    'embedding_mean': np.mean(embeddings),
                    'embedding_std': np.std(embeddings)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, 'inference_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            
            print(f"Summary saved to: {summary_path}")
            print(f"Results saved to: {output_dir}")
        
        print("\nInference pipeline completed successfully!")
        print("\nOutput files for each processed dataset:")
        print("  Format: <original_filename>_inference_YYYY-MM-DD_HH-MM-SS_<suffix>")
        print("  *_embeddings.npy     - Cell embeddings")
        print("  *_metadata.csv       - Cell metadata")
        print("  *_summary.txt        - Inference report")
        print("  *_inference_report.pkl - Detailed analysis")
        
        if args.save_processed:
            print("  *_processed_data.npy - Processed expression data")
        
    except Exception as e:
        print(f"\nError during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 