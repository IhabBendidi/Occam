#!/usr/bin/env python3
"""
Main reconstruction script for Occam with h5ad data support

Reconstructs raw gene expression counts from input h5ad files using trained Occam models.

Usage:
    python occam_reconstruct.py --model_dir ./trained_model --input_data data.h5ad --output results.h5ad

Example with specific checkpoint:
    python occam_reconstruct.py \
        --model_dir ./trained_model/checkpoints \
        --input_data input_raw_counts.h5ad \
        --output reconstructed_counts.h5ad \
        --checkpoint checkpoint_epoch_1120.pth \
        --preprocessor_path ./trained_model/preprocessor.pkl \
        --gene_col auto \
        --batch_size 1024

Example with external decoder (Occam encoder + separate decoder for ~17k genes):
    python occam_reconstruct.py \
        --model_dir ./trained_model/checkpoints \
        --input_data input_raw_counts.h5ad \
        --output reconstructed_17k_genes.h5ad \
        --checkpoint checkpoint_epoch_1120.pth \
        --preprocessor_path ./trained_model/preprocessor.pkl \
        --external_decoder_path ./decoder_checkpoint.pt \
        --batch_size 1024
"""

import argparse
import os
import sys
import torch

from occam.inference.reconstruct import reconstruct_from_h5ad


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Reconstruct gene expression counts with trained Occam model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model and preprocessor')
    parser.add_argument('--checkpoint', type=str,
                        help='Optional checkpoint path or filename (absolute or relative to model_dir). If omitted, defaults to best_model.pth, then final_model.pth, then model_checkpoint.pth')
    parser.add_argument('--preprocessor_path', type=str,
                        help='Optional preprocessor path or filename (absolute or relative to model_dir). If omitted, loader will search next to checkpoint, parent, then cache, then debug fallback')
    parser.add_argument('--external_decoder_path', type=str,
                        help='Optional path to external SEResNetDecoder checkpoint (.pt file). If provided, uses Occam encoder combined with the external decoder to output ~17k genes')
    parser.add_argument('--decoder_config_path', type=str,
                        help='Optional separate checkpoint for decoder config/gene metadata. Use when weights checkpoint lacks model_config')
    
    # Data arguments
    parser.add_argument('--input_data', type=str, required=True,
                        help='Path to input h5ad file (must contain raw counts)')
    parser.add_argument('--gene_col', type=str, default='auto',
                        help='Gene name column in adata.var; use "auto" to pick best-overlap column')
    
    # Output arguments
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for reconstructed h5ad file')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for reconstruction')
    
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
    # (the reconstruction pipeline will build a CheckpointPreprocessor from checkpoint data)
    if getattr(args, 'preprocessor_path', None):
        candidate = args.preprocessor_path if os.path.isabs(args.preprocessor_path) else os.path.join(args.model_dir, args.preprocessor_path)
        if os.path.exists(candidate):
            args.preprocessor_path = os.path.abspath(candidate)
    
    # Check external decoder path if provided
    if getattr(args, 'external_decoder_path', None):
        if not os.path.exists(args.external_decoder_path):
            raise FileNotFoundError(f"External decoder checkpoint not found: {args.external_decoder_path}")
        args.external_decoder_path = os.path.abspath(args.external_decoder_path)
    
    # Check decoder config path if provided
    if getattr(args, 'decoder_config_path', None):
        if not os.path.exists(args.decoder_config_path):
            raise FileNotFoundError(f"Decoder config checkpoint not found: {args.decoder_config_path}")
        args.decoder_config_path = os.path.abspath(args.decoder_config_path)
    
    # Check input file
    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f"Input file not found: {args.input_data}")
    
    # Check output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"📁 Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    return args


def main():
    args = parse_args()
    args = validate_args(args)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 70)
    print("Occam Reconstruction Pipeline")
    print("=" * 70)
    print(f"Model directory: {args.model_dir}")
    if getattr(args, 'checkpoint', None):
        print(f"Checkpoint: {args.checkpoint}")
    if getattr(args, 'preprocessor_path', None):
        print(f"Preprocessor: {args.preprocessor_path}")
    if getattr(args, 'external_decoder_path', None):
        print(f"External decoder: {args.external_decoder_path}")
        if getattr(args, 'decoder_config_path', None):
            print(f"Decoder config: {args.decoder_config_path}")
        print(f"  (Using Occam encoder + external decoder for ~17k gene output)")
    print(f"Input file: {args.input_data}")
    print(f"Output file: {args.output}")
    print(f"Gene column: {args.gene_col}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    print()
    
    try:
        # Handle gene_col 'auto' conversion
        gene_arg = None if (args.gene_col is None or (isinstance(args.gene_col, str) and args.gene_col.lower() == 'auto')) else args.gene_col
        
        # Run reconstruction
        adata_reconstructed = reconstruct_from_h5ad(
            adata_path=args.input_data,
            model_dir=args.model_dir,
            output_path=args.output,
            gene_name_col=gene_arg,
            batch_size=args.batch_size,
            device=args.device,
            checkpoint_path=args.checkpoint,
            preprocessor_path=args.preprocessor_path,
            external_decoder_path=getattr(args, 'external_decoder_path', None),
            decoder_config_path=getattr(args, 'decoder_config_path', None)
        )
        
        print()
        print("=" * 70)
        print("✅ Reconstruction completed successfully!")
        print("=" * 70)
        print(f"Output saved to: {args.output}")
        print(f"Reconstructed data shape: {adata_reconstructed.shape}")
        print(f"  Cells: {adata_reconstructed.n_obs:,}")
        print(f"  Genes: {adata_reconstructed.n_vars:,}")
        print()
        print("Output AnnData contains:")
        print("  - X: Reconstructed raw counts")
        print("  - obs: Cell metadata (copied from input)")
        if getattr(args, 'external_decoder_path', None):
            print("  - var: Gene names from external decoder (~17k ENSG IDs)")
        else:
            print("  - var: Gene names/IDs from training")
        print("  - uns['reconstruction_info']: Reconstruction metadata")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Error during reconstruction: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


