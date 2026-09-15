"""
Reconstruction pipeline for Occam with h5ad data support

Provides reconstruction capabilities using both encoder and decoder to predict
raw gene expression counts from input data with adaptive inverse preprocessing.

Supports external decoder mode where Occam encoder is combined with a separate
SEResNetDecoder for decoding to ~17k genes.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Dict, Optional, Tuple, Any, Union, List
import pickle
from pathlib import Path

# Chunked processing constants
CHUNK_THRESHOLD = 300_000  # Activate chunked processing when n_obs > this
CHUNK_SIZE = 150_000  # Number of samples per chunk

from occam.model import AutoEncoder
from occam.mae.config import PretrainConfig
from occam.inference.embed import load_trained_model, OccamEmbedder, CheckpointPreprocessor


class HybridEncoderDecoder(nn.Module):
    """
    Hybrid model combining Occam encoder with an external SEResNetDecoder.

    This allows using the pretrained Occam encoder to generate embeddings,
    then decode with a separately trained decoder that outputs ~17k genes.
    """

    def __init__(self, occam_model: AutoEncoder, external_decoder: nn.Module):
        """
        Initialize hybrid model.

        Args:
            occam_model: Pretrained Occam model (encoder will be extracted)
            external_decoder: External SEResNetDecoder for decoding
        """
        super().__init__()
        # Extract encoder from Occam model
        self.encoder = occam_model.encoder
        self.external_decoder = external_decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode with Occam encoder, decode with external decoder.

        Args:
            x: Input tensor (preprocessed gene expression)

        Returns:
            Decoded output in log1p space (~17k genes)
        """
        # Get embeddings from Occam encoder
        latent = self.encoder(x)
        # Decode with external decoder (outputs in log1p space)
        decoded = self.external_decoder(latent)
        return decoded
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from encoder only."""
        return self.encoder(x)


class OccamReconstructor:
    """Reconstruction pipeline for trained Occam models."""
    
    def __init__(self, 
                 model_path: str,
                 preprocessor_path: Optional[str] = None,
                 device: str = 'cuda',
                 external_decoder_path: Optional[str] = None,
                 decoder_config_path: Optional[str] = None):
        """
        Initialize the reconstruction pipeline.
        
        Args:
            model_path: Path to saved model checkpoint
            preprocessor_path: Optional path to saved preprocessor pickle.
                If None, a CheckpointPreprocessor is built from the checkpoint.
            device: Device to use for reconstruction
            external_decoder_path: Optional path to external SEResNetDecoder checkpoint.
                                   If provided, uses Occam encoder + external decoder.
            decoder_config_path: Optional separate checkpoint for decoder config/gene metadata.
                                If provided, loads config from this file but weights from external_decoder_path.
        """
        self.device = device
        self.external_decoder_gene_names = None
        self.external_decoder_gene_to_index = None

        # Load model checkpoint first to get the correct configuration
        checkpoint = torch.load(model_path, map_location=device)

        # Detect unified encoder+decoder checkpoint
        self._is_unified_checkpoint = (
            isinstance(checkpoint, dict) and checkpoint.get('is_unified_checkpoint', False)
        )
        self.use_external_decoder = (
            external_decoder_path is not None or self._is_unified_checkpoint
        )
        
        # Load or build preprocessor
        if preprocessor_path is not None:
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            if not hasattr(self.preprocessor, 'gene_names'):
                raise ValueError(f"Loaded object is not a valid preprocessor: {type(self.preprocessor)}")
            print(f"Loaded preprocessor from {preprocessor_path} ({len(self.preprocessor.gene_names)} genes)")
        else:
            self.preprocessor = CheckpointPreprocessor.from_checkpoint(checkpoint)
            print(f"Built preprocessor from checkpoint ({len(self.preprocessor.gene_names)} genes)")
        
        # Store preprocessing flags for inverse operations
        self.normalize_per_cell = getattr(self.preprocessor, 'normalize_per_cell', True)
        self.log1p_transform = getattr(self.preprocessor, 'log1p_transform', True)
        self.scale = getattr(self.preprocessor, 'scale', True)
        self.scaler_params = getattr(self.preprocessor, 'scaler_params', None)
        self.target_sum = getattr(self.preprocessor, 'target_sum', 1e4)
        self.computed_median = getattr(self.preprocessor, 'computed_median', None)
        
        # If target_sum was None during training, use the computed median
        if self.target_sum is None and self.computed_median is not None:
            self.effective_target_sum = self.computed_median
            print(f"Using computed median from training: {self.computed_median:.2f}")
        else:
            self.effective_target_sum = self.target_sum
        
        print(f"Preprocessing flags:")
        print(f"  normalize_per_cell: {self.normalize_per_cell}")
        print(f"  log1p_transform: {self.log1p_transform}")
        print(f"  scale: {self.scale}")
        print(f"  scaler_params available: {self.scaler_params is not None}")
        print(f"  target_sum: {self.target_sum}")
        print(f"  computed_median: {self.computed_median}")
        print(f"  effective_target_sum (for encoding): {self.effective_target_sum}")
        
        # Handle checkpoint formats: unified, standard, or legacy
        if self._is_unified_checkpoint:
            # Unified encoder+decoder checkpoint
            enc_config = checkpoint['encoder_config']
            num_genes = enc_config.get('num_genes', len(self.preprocessor.gene_names))
            model_size = enc_config.get('model_size', 'small')
            hidden_size = enc_config.get('hidden_size', None)

            self.config = PretrainConfig()
            self.config.model_size = model_size
            self.config.hidden_size = hidden_size
            self.config.num_genes = num_genes

            self.model = AutoEncoder(
                num_genes=num_genes,
                model_size=model_size,
                hidden_size=hidden_size,
            ).to(device)

        elif 'config' in checkpoint:
            # New format with training config
            self.config = PretrainConfig.from_dict(checkpoint['config'])

            # Infer actual num_genes from checkpoint state_dict (most reliable)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
            num_genes_from_checkpoint = None

            # Try to infer from encoder.1.weight which is (hidden_size, num_genes)
            for key in ['encoder.1.weight', 'module.encoder.1.weight', '_orig_mod.encoder.1.weight']:
                if key in state_dict:
                    num_genes_from_checkpoint = state_dict[key].shape[1]
                    break

            # If not found, try decoder.weight which is (num_genes, hidden_size_last)
            if num_genes_from_checkpoint is None:
                for key in ['decoder.weight', 'module.decoder.weight']:
                    if key in state_dict:
                        num_genes_from_checkpoint = state_dict[key].shape[0]
                        break

            num_genes_from_config = self.config.num_genes
            num_genes_from_preprocessor = len(self.preprocessor.gene_names)

            # Use checkpoint value as ground truth
            if num_genes_from_checkpoint is not None:
                num_genes = num_genes_from_checkpoint
                if num_genes != num_genes_from_config or num_genes != num_genes_from_preprocessor:
                    print(f"Gene count mismatch detected:")
                    print(f"   - Config num_genes: {num_genes_from_config}")
                    print(f"   - Preprocessor genes: {num_genes_from_preprocessor}")
                    print(f"   - Checkpoint actual genes: {num_genes_from_checkpoint}")
                    print(f"   - Using checkpoint value: {num_genes_from_checkpoint}")
            else:
                # Fallback to preprocessor (most conservative)
                num_genes = num_genes_from_preprocessor
                print(f"Could not infer num_genes from checkpoint")
                print(f"   - Using preprocessor value: {num_genes}")

            # Create model using actual checkpoint num_genes
            self.model = AutoEncoder(
                num_genes=num_genes,
                model_size=self.config.model_size,
                hidden_size=self.config.hidden_size,
                dropout=self.config.dropout,
                masked_data_weight=self.config.masked_data_weight,
                mask_loss_weight=self.config.mask_loss_weight,
                reconstruction_loss=getattr(self.config, 'reconstruction_loss', 'mse')
            ).to(device)
        else:
            # Old format compatibility
            self.config = None
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            else:
                # Fallback to default config using preprocessor
                model_config = {
                    'num_genes': len(self.preprocessor.gene_names),
                    'hidden_size': 128,
                    'dropout': 0.0,
                    'masked_data_weight': 0.75,
                    'mask_loss_weight': 0.7,
                    'reconstruction_loss': 'mse'
                }
            
            self.model = AutoEncoder(**model_config).to(device)
        
        # Load model weights and handle DDP/torch.compile prefixes gracefully
        if self._is_unified_checkpoint:
            # Unified checkpoint: encoder_state_dict has only encoder keys
            state_dict = checkpoint['encoder_state_dict']
            state_dict = {k.replace('module.', '', 1).replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint.get('model', None)
            if state_dict is None:
                raise KeyError("Checkpoint does not contain 'model_state_dict' or 'model'")
            state_dict = {k.replace('module.', '', 1).replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        self.model.eval()

        # Check if model was trained with presence-as-input
        _cfg = checkpoint.get('config', {})
        self._gene_presence_input = _cfg.get('gene_presence_mask', False) if isinstance(_cfg, dict) else False
        self._presence_shift = (_cfg.get('presence_shift', 1.0) if isinstance(_cfg, dict) else 1.0) if self._gene_presence_input else 1.0

        print(f"Loaded model from {model_path}")
        print(f"Model loaded with {len(self.preprocessor.gene_names)} genes")

        # Load decoder (from unified checkpoint or separate file)
        if self._is_unified_checkpoint:
            print(f"\nLoading Occam Decoder from unified checkpoint")
            from occam.decoder.external import HybridNormGlobalSkipDecoder as _HybridDecoder

            dec_config = checkpoint['decoder_config']
            ext_decoder = _HybridDecoder(
                input_dim=dec_config['input_dim'],
                output_dim=dec_config['output_dim'],
                hidden_dim=dec_config['hidden_dim'],
                num_blocks=dec_config['num_blocks'],
                dropout_rate=dec_config.get('dropout_rate', 0.3),
                se_reduction=dec_config.get('se_reduction', 16),
            ).to(device)
            ext_decoder.load_state_dict(checkpoint['decoder_state_dict'])
            ext_decoder.eval()

            gene_names = checkpoint['gene_names']
            gene_to_index = checkpoint.get('gene_to_index', {g: i for i, g in enumerate(gene_names)})
            self.external_decoder_gene_names = gene_names
            self.external_decoder_gene_to_index = gene_to_index

            print(f"  Decoder config: input={dec_config['input_dim']}, output={dec_config['output_dim']}, "
                  f"hidden={dec_config['hidden_dim']}, blocks={dec_config['num_blocks']}")

            self.hybrid_model = HybridEncoderDecoder(self.model, ext_decoder)
            self.hybrid_model.eval()
            print(f"  Created hybrid encoder-decoder model ({len(gene_names)} genes)")

        elif external_decoder_path is not None:
            print(f"\nLoading Occam Decoder from: {external_decoder_path}")
            if decoder_config_path:
                print(f"   Config/gene metadata from: {decoder_config_path}")
            from occam.decoder.external import load_decoder, SEResNetDecoder

            ext_decoder, gene_names, gene_to_index, decoder_config = load_decoder(
                external_decoder_path, device=device, config_checkpoint_path=decoder_config_path
            )

            self.external_decoder_gene_names = gene_names
            self.external_decoder_gene_to_index = gene_to_index

            print(f"  Decoder config:")
            print(f"    Input dim: {decoder_config['input_dim']}")
            print(f"    Output dim: {decoder_config['output_dim']} genes")
            print(f"    Hidden dim: {decoder_config['hidden_dim']}")
            print(f"    Num blocks: {decoder_config['num_blocks']}")

            # Verify input dimension matches Occam encoder output
            occam_hidden_size = self.model.hidden_size
            if decoder_config['input_dim'] != occam_hidden_size:
                raise ValueError(
                    f"Decoder input_dim ({decoder_config['input_dim']}) does not match "
                    f"Occam encoder hidden_size ({occam_hidden_size})"
                )

            # Create hybrid model
            self.hybrid_model = HybridEncoderDecoder(self.model, ext_decoder)
            self.hybrid_model.eval()
            print(f"  Created hybrid encoder-decoder model ({len(gene_names)} genes)")
        
    def _infer_library_size_from_raw(self, adata_raw: sc.AnnData) -> float:
        """
        Infer library size from raw count data.
        
        Uses median of total counts per cell as a robust estimate.
        
        Args:
            adata_raw: AnnData object with raw counts
            
        Returns:
            Median library size
        """
        # Compute total counts per cell
        if hasattr(adata_raw.X, 'toarray'):
            # Sparse matrix
            counts_per_cell = np.array(adata_raw.X.sum(axis=1)).flatten()
        else:
            # Dense matrix
            counts_per_cell = adata_raw.X.sum(axis=1)
        
        median_lib_size = np.median(counts_per_cell)
        print(f"Inferred library size from input data: {median_lib_size:.2f} (median)")
        print(f"  Min: {counts_per_cell.min():.2f}, Max: {counts_per_cell.max():.2f}")
        
        return median_lib_size
    
    def _check_for_issues(self, data: np.ndarray, step_name: str) -> None:
        """
        Check for NaN, Inf, and other problematic values and log diagnostics.
        
        Args:
            data: Array to check
            step_name: Name of the step for logging
        """
        has_nan = np.any(np.isnan(data))
        has_inf = np.any(np.isinf(data))
        has_negative = np.any(data < 0)
        
        if has_nan or has_inf or has_negative:
            print(f"\n{'='*70}")
            print(f"⚠️  ISSUE DETECTED AT: {step_name}")
            print(f"{'='*70}")
            
            if has_nan:
                nan_count = np.sum(np.isnan(data))
                nan_percent = (nan_count / data.size) * 100
                print(f"❌ NaN values found: {nan_count:,} ({nan_percent:.2f}% of total)")
                # Find which cells/genes have NaN
                nan_cells = np.any(np.isnan(data), axis=1)
                nan_genes = np.any(np.isnan(data), axis=0)
                print(f"   Cells with NaN: {np.sum(nan_cells):,} / {data.shape[0]:,}")
                print(f"   Genes with NaN: {np.sum(nan_genes):,} / {data.shape[1]:,}")
            
            if has_inf:
                inf_count = np.sum(np.isinf(data))
                inf_percent = (inf_count / data.size) * 100
                print(f"❌ Inf values found: {inf_count:,} ({inf_percent:.2f}% of total)")
                pos_inf = np.sum(data == np.inf)
                neg_inf = np.sum(data == -np.inf)
                print(f"   Positive Inf: {pos_inf:,}, Negative Inf: {neg_inf:,}")
            
            if has_negative:
                neg_count = np.sum(data < 0)
                neg_percent = (neg_count / data.size) * 100
                print(f"⚠️  Negative values found: {neg_count:,} ({neg_percent:.2f}% of total)")
                print(f"   Min value: {np.min(data):.6f}")
            
            print(f"{'='*70}\n")
        else:
            # All good - log success
            print(f"  ✅ {step_name}: No issues detected")
            print(f"     Stats: min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}")
    
    def _inverse_preprocess(self, 
                           reconstructed: np.ndarray, 
                           library_size: float,
                           external_decoder_mode: bool = False) -> np.ndarray:
        """
        Apply inverse preprocessing to convert reconstructed values back to raw counts.
        
        Adaptively applies inverse transformations based on what the preprocessor did:
        1. Inverse scaling (if scale=True) - SKIPPED for external decoder mode
        2. Inverse log1p (if log1p_transform=True)
        3. Inverse library normalization (if normalize_per_cell=True)
        
        Args:
            reconstructed: Reconstructed values from decoder (cells × genes)
            library_size: Library size to use for denormalization
            external_decoder_mode: If True, skip inverse scaling (decoder outputs log1p directly)
            
        Returns:
            Reconstructed raw counts
        """
        result = reconstructed.copy()
        
        print("\n" + "="*70)
        print("INVERSE PREPROCESSING DIAGNOSTICS")
        if external_decoder_mode:
            print("(External Decoder Mode - decoder outputs in log1p space)")
        print("="*70)
        
        # Initial check
        print("\n[STEP 0] Initial reconstructed output from model:")
        self._check_for_issues(result, "Initial model output")
        
        # Step 1: Inverse scaling (if applied) - SKIP for external decoder mode
        if external_decoder_mode:
            print("\n[STEP 1] Inverse scaling SKIPPED (external decoder outputs log1p directly)")
        elif self.scale and self.scaler_params is not None:
            mean = self.scaler_params.get('mean', None)
            std = self.scaler_params.get('std', None)
            
            if mean is not None and std is not None:
                print("\n[STEP 1] Applying inverse scaling...")
                print(f"  Scaler mean shape: {mean.shape}, std shape: {std.shape}")
                print(f"  Scaler mean range: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
                print(f"  Scaler std range: [{np.min(std):.4f}, {np.max(std):.4f}]")
                
                # Check for zero or very small std
                small_std = np.sum(std < 1e-8)
                if small_std > 0:
                    print(f"  ⚠️  Warning: {small_std} genes have very small std (<1e-8)")
                
                result = result * std + mean
                self._check_for_issues(result, "After inverse scaling")
            else:
                print("\n[STEP 1] ⚠️ Scaling was applied but scaler params not available - SKIPPED")
        else:
            print("\n[STEP 1] Inverse scaling not needed (scale=False)")
        
        # Step 2: Inverse log1p (if applied)
        if self.log1p_transform:
            print("\n[STEP 2] Applying inverse log1p (expm1)...")
            print(f"  Input to expm1 - min: {np.min(result):.4f}, max: {np.max(result):.4f}")
            
            # Check for very large values that might overflow
            very_large = np.sum(result > 700)  # exp(700) is near float max
            if very_large > 0:
                print(f"  ⚠️  Warning: {very_large} values > 700 (may overflow in expm1)")
            
            # Check for negative values before expm1
            neg_values = np.sum(result < 0)
            if neg_values > 0:
                print(f"  ⚠️  Warning: {neg_values} negative values before expm1")
            
            result = np.expm1(result)
            self._check_for_issues(result, "After expm1")
            
            # Clip negative values that might arise from numerical issues
            neg_after = np.sum(result < 0)
            if neg_after > 0:
                print(f"  Clipping {neg_after} negative values to 0...")
                result = np.maximum(result, 0)
                self._check_for_issues(result, "After clipping negatives")
        else:
            print("\n[STEP 2] Inverse log1p not needed (log1p_transform=False)")
        
        # Step 3: Inverse library normalization (if applied)
        if self.normalize_per_cell:
            print(f"\n[STEP 3] Applying inverse normalization (target library size: {library_size:.2f})...")
            
            # Compute per-cell library sizes BEFORE normalization
            prediction_library_size = result.sum(axis=1, keepdims=True)
            print(f"  Current per-cell totals before renormalization:")
            print(f"    Min: {np.min(prediction_library_size):.4f}")
            print(f"    Max: {np.max(prediction_library_size):.4f}")
            print(f"    Mean: {np.mean(prediction_library_size):.4f}")
            print(f"    Median: {np.median(prediction_library_size):.4f}")
            
            # Check for zero totals (will cause division by zero!)
            zero_totals = np.sum(prediction_library_size == 0)
            if zero_totals > 0:
                print(f"\n{'='*70}")
                print(f"❌ CRITICAL: {zero_totals} cells have ZERO total counts!")
                print(f"   This will cause division by zero and result in NaN!")
                print(f"   Replacing zeros with 1.0 to prevent NaN...")
                print(f"{'='*70}\n")
                prediction_library_size = np.where(prediction_library_size == 0, 1.0, prediction_library_size)
            
            # Very small totals (near zero) can also cause issues
            very_small = np.sum((prediction_library_size > 0) & (prediction_library_size < 1e-6))
            if very_small > 0:
                print(f"  ⚠️  Warning: {very_small} cells have very small totals (<1e-6)")
            
            # Perform the normalization
            print(f"  Normalizing: (counts / current_total) * {library_size:.2f}")
            result = (result / prediction_library_size) * library_size
            
            self._check_for_issues(result, "After library size normalization")
            
            # Verify final library sizes
            final_library_sizes = result.sum(axis=1)
            print(f"\n  Final per-cell library sizes:")
            print(f"    Min: {np.min(final_library_sizes):.4f}")
            print(f"    Max: {np.max(final_library_sizes):.4f}")
            print(f"    Mean: {np.mean(final_library_sizes):.4f}")
            print(f"    Median: {np.median(final_library_sizes):.4f}")
            print(f"    Target was: {library_size:.2f}")
        else:
            print("\n[STEP 3] Inverse library normalization not needed (normalize_per_cell=False)")
        
        print("\n" + "="*70)
        print("INVERSE PREPROCESSING COMPLETE")
        print("="*70 + "\n")
        
        return result
    
    def reconstruct_counts(self, 
                          adata: sc.AnnData,
                          gene_name_col: Optional[str] = None,
                          batch_size: int = 512) -> Tuple[np.ndarray, float]:
        """
        Reconstruct counts for input data with proper gene alignment.
        
        Args:
            adata: Input AnnData object (should contain raw counts)
            gene_name_col: Gene name column (auto-detect if None)
            batch_size: Batch size for reconstruction
            
        Returns:
            Tuple of (reconstructed_raw_counts, library_size_used)
        """
        print(f"🧬 Reconstructing counts for {adata.shape[0]:,} cells...")
        
        # Store original raw counts for library size inference
        adata_raw_for_libsize = adata.copy()
        
        # Decide gene column: auto-select best-overlap if not provided or if 'auto'
        used_gene_col = gene_name_col
        try:
            training_genes = set(getattr(self.preprocessor, 'gene_names', []) or [])
            var_cols = list(adata.var.columns)
            candidate_cols = []
            train_gene_col = getattr(self.preprocessor, 'train_gene_col', None)
            if train_gene_col:
                candidate_cols.append(train_gene_col)
            candidate_cols.extend(['gene_name', 'gene_symbol', 'gene_symbols', 'symbol', 'feature_name', 'feature_names', 'Gene', 'ensembl_gene_id', 'gene_id'])
            seen = set()
            candidate_cols = [c for c in candidate_cols if (c not in seen and not (seen.add(c)))]
            def overlap_for(col: Optional[str]) -> int:
                if col is None or col not in var_cols:
                    return -1
                try:
                    names = {str(x).strip().upper() for x in adata.var[col].tolist()}
                    return len({g.strip().upper() for g in training_genes} & names)
                except Exception:
                    return -1
            force_auto = (gene_name_col is None) or (isinstance(gene_name_col, str) and gene_name_col.lower() == 'auto')
            current_overlap = overlap_for(gene_name_col) if (gene_name_col and not force_auto) else -1
            best_col = None
            best_overlap = -1
            for col in candidate_cols:
                ov = overlap_for(col)
                if ov > best_overlap:
                    best_overlap = ov
                    best_col = col
            if force_auto or best_overlap > current_overlap:
                used_gene_col = best_col
                if used_gene_col:
                    print(f"Auto-selected gene column: {used_gene_col} (overlap={best_overlap}/{len(training_genes)})")
                else:
                    print("No positive-overlap gene column found; proceeding with default behavior")
        except Exception as e:
            print(f"Gene column auto-selection failed: {e}; proceeding with provided setting")

        # Disable cell filtering during reconstruction to preserve all cells
        original_filter_setting = getattr(self.preprocessor, 'filter_min_counts', False)
        self.preprocessor.filter_min_counts = False
        
        try:
            # Handle different preprocessor types
            if hasattr(self.preprocessor, 'chunk_size'):  # MemoryEfficientOccamDataPreprocessor (legacy: MemoryEfficientOccamDataPreprocessor)
                print("Using MemoryEfficientOccamDataPreprocessor (legacy: MemoryEfficientOccamDataPreprocessor) - creating temporary file...")

                # Create temporary file for MemoryEfficientOccamDataPreprocessor (legacy: MemoryEfficientOccamDataPreprocessor)
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                    
                try:
                    # Save AnnData to temporary file
                    adata.write_h5ad(temp_path)
                    
                    # Use the memory-efficient preprocessor with file path
                    X_processed = self.preprocessor.transform(temp_path, used_gene_col)
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            else:  # Regular OccamDataPreprocessor (legacy: OccamDataPreprocessor)
                print("Using regular OccamDataPreprocessor (legacy: OccamDataPreprocessor)...")
                
                # Use regular preprocessor with AnnData object
                X_processed = self.preprocessor.transform(adata, used_gene_col)
        finally:
            # Restore original filter setting
            self.preprocessor.filter_min_counts = original_filter_setting
        
        print(f"Preprocessed data shape: {X_processed.shape}")
        print(f"Using {len(self.preprocessor.gene_names)} genes (intersection with training data)")
        
        # Infer library size from raw input data
        library_size = self._infer_library_size_from_raw(adata_raw_for_libsize)
        
        # Create dataset and dataloader
        dataset = TensorDataset(torch.tensor(X_processed, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Running reconstruction with batch size {batch_size}...")
        if self.use_external_decoder:
            print(f"  Using external decoder (output: {len(self.external_decoder_gene_names)} genes)")
        print("="*70)
        
        # Run reconstruction
        reconstructions_list = []
        
        # Select model to use
        if self.use_external_decoder:
            model_to_use = self.hybrid_model
        else:
            model_to_use = self.model
        model_to_use.eval()
        
        # Track issues during forward pass
        nan_batches = []
        inf_batches = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                    
                batch_x = batch_x.to(self.device)
                if self._gene_presence_input:
                    batch_x = batch_x + self._presence_shift  # All genes present in single-file reconstruction

                # Check input batch for issues
                if batch_idx == 0:
                    print(f"\n[FORWARD PASS] First batch diagnostics:")
                    print(f"  Input shape: {batch_x.shape}")
                    print(f"  Input range: [{batch_x.min().item():.4f}, {batch_x.max().item():.4f}]")
                    print(f"  Input mean: {batch_x.mean().item():.4f}, std: {batch_x.std().item():.4f}")
                    if torch.isnan(batch_x).any():
                        print(f"  ❌ WARNING: Input batch contains NaN!")
                    if torch.isinf(batch_x).any():
                        print(f"  ❌ WARNING: Input batch contains Inf!")

                if self.use_external_decoder:
                    # Hybrid model: encoder -> external decoder (outputs log1p directly)
                    reconstruction = model_to_use(batch_x)
                    latent = None
                    predicted_mask = None
                else:
                    # Original Occam: encoder -> mask predictor -> decoder
                    latent, predicted_mask, reconstruction = self.model.forward_mask(batch_x)
                    # Apply Softplus activation to decoder output
                    reconstruction = torch.nn.functional.softplus(reconstruction)
                
                # Check reconstruction for issues
                recon_cpu = reconstruction.cpu().numpy()
                
                if np.isnan(recon_cpu).any():
                    nan_batches.append(batch_idx)
                    if len(nan_batches) == 1:  # First NaN batch
                        print(f"\n{'='*70}")
                        print(f"❌ NaN DETECTED in batch {batch_idx}")
                        print(f"{'='*70}")
                        print(f"  Reconstruction output range: [{recon_cpu.min():.4f}, {recon_cpu.max():.4f}]")
                        print(f"  NaN count: {np.sum(np.isnan(recon_cpu)):,}")
                        if latent is not None:
                            print(f"  Latent stats: min={latent.min().item():.4f}, max={latent.max().item():.4f}")
                        if predicted_mask is not None:
                            print(f"  Predicted mask stats: min={predicted_mask.min().item():.4f}, max={predicted_mask.max().item():.4f}")
                        print(f"{'='*70}\n")
                
                if np.isinf(recon_cpu).any():
                    inf_batches.append(batch_idx)
                    if len(inf_batches) == 1:  # First Inf batch
                        print(f"\n{'='*70}")
                        print(f"❌ Inf DETECTED in batch {batch_idx}")
                        print(f"{'='*70}")
                        print(f"  Inf count: {np.sum(np.isinf(recon_cpu)):,}")
                        print(f"{'='*70}\n")
                
                reconstructions_list.append(recon_cpu)
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches", end="")
                    if nan_batches:
                        print(f" [⚠️  {len(nan_batches)} batches with NaN so far]")
                    else:
                        print()
        
        # Summary of forward pass issues
        if nan_batches or inf_batches:
            print(f"\n{'='*70}")
            print(f"⚠️  FORWARD PASS ISSUES SUMMARY")
            print(f"{'='*70}")
            if nan_batches:
                print(f"❌ Batches with NaN: {len(nan_batches)} / {len(dataloader)}")
                print(f"   First 5 affected batches: {nan_batches[:5]}")
            if inf_batches:
                print(f"❌ Batches with Inf: {len(inf_batches)} / {len(dataloader)}")
                print(f"   First 5 affected batches: {inf_batches[:5]}")
            print(f"{'='*70}\n")
        
        # Concatenate all reconstructions
        reconstructed = np.concatenate(reconstructions_list, axis=0)
        
        print(f"\n✅ Generated reconstructions with shape: {reconstructed.shape}")
        print(f"[FORWARD PASS COMPLETE] Checking concatenated output:")
        self._check_for_issues(reconstructed, "Concatenated model output")
        
        # Apply inverse preprocessing
        print("🔄 Applying inverse preprocessing to recover raw counts...")
        reconstructed_raw = self._inverse_preprocess(
            reconstructed, library_size, external_decoder_mode=self.use_external_decoder
        )

        
        # Final quality check
        print(f"\n{'='*70}")
        print(f"FINAL OUTPUT QUALITY CHECK")
        print(f"{'='*70}")
        
        has_nan_final = np.any(np.isnan(reconstructed_raw))
        has_inf_final = np.any(np.isinf(reconstructed_raw))
        has_negative_final = np.any(reconstructed_raw < 0)
        
        if has_nan_final or has_inf_final:
            print(f"❌ CRITICAL ISSUES IN FINAL OUTPUT:")
            if has_nan_final:
                nan_count = np.sum(np.isnan(reconstructed_raw))
                nan_pct = (nan_count / reconstructed_raw.size) * 100
                print(f"   ❌ NaN values: {nan_count:,} ({nan_pct:.2f}%)")
            if has_inf_final:
                inf_count = np.sum(np.isinf(reconstructed_raw))
                inf_pct = (inf_count / reconstructed_raw.size) * 100
                print(f"   ❌ Inf values: {inf_count:,} ({inf_pct:.2f}%)")
            if has_negative_final:
                neg_count = np.sum(reconstructed_raw < 0)
                print(f"   ⚠️  Negative values: {neg_count:,}")
            print(f"\n⚠️  OUTPUT MAY NOT BE USABLE - REVIEW DIAGNOSTICS ABOVE")
        else:
            print(f"✅ OUTPUT QUALITY: GOOD (no NaN or Inf detected)")
        
        print(f"\n✅ Final reconstructed raw counts shape: {reconstructed_raw.shape}")
        print(f"   - Input cells: {adata.shape[0]:,}")
        print(f"   - Output cells: {reconstructed_raw.shape[0]:,}")
        
        if not (has_nan_final or has_inf_final):
            print(f"   - Reconstructed count statistics:")
            print(f"     Min: {reconstructed_raw.min():.2f}, Max: {reconstructed_raw.max():.2f}")
            print(f"     Mean: {reconstructed_raw.mean():.2f}, Median: {np.median(reconstructed_raw):.2f}")
            
            # Check distribution sanity
            zeros = np.sum(reconstructed_raw == 0)
            zero_pct = (zeros / reconstructed_raw.size) * 100
            print(f"     Zero values: {zeros:,} ({zero_pct:.2f}%)")
        else:
            print(f"   - Reconstructed count statistics:")
            print(f"     Min: nan, Max: nan")
            print(f"     Mean: nan, Median: nan")
            print(f"     ⚠️  Cannot compute statistics due to NaN/Inf values")
        
        print(f"{'='*70}\n")
        
        return reconstructed_raw, library_size
    
    def reconstruct_and_save(self, 
                            adata: sc.AnnData, 
                            output_path: str,
                            gene_name_col: Optional[str] = None,
                            batch_size: int = 512) -> sc.AnnData:
        """
        Reconstruct counts and save as AnnData h5ad file.
        
        Args:
            adata: Input AnnData object (raw counts)
            output_path: Output path for h5ad file
            gene_name_col: Gene name column (auto-detect if None)
            batch_size: Batch size for reconstruction
            
        Returns:
            AnnData with reconstructed counts
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"📁 Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate reconstructed counts
        reconstructed_raw, library_size = self.reconstruct_counts(
            adata, gene_name_col, batch_size
        )
        
        # Create output AnnData
        print(f"📦 Creating output AnnData...")
        
        # Determine gene names to use
        if self.use_external_decoder:
            output_gene_names = self.external_decoder_gene_names
            var_col_name = 'ensembl_gene_id'  # External decoder uses ENSG IDs
        else:
            output_gene_names = self.preprocessor.gene_names
            # Determine the column name to use for gene identifiers
            if gene_name_col and gene_name_col != 'auto':
                var_col_name = gene_name_col
            else:
                # Check if preprocessor has a stored train_gene_col
                var_col_name = getattr(self.preprocessor, 'train_gene_col', None)
                if not var_col_name:
                    # Default fallback
                    var_col_name = 'ensembl_gene_id'
        
        # Create new AnnData with reconstructed counts
        adata_reconstructed = sc.AnnData(
            X=reconstructed_raw,
            obs=adata.obs.copy(),  # Preserve cell metadata
            var=pd.DataFrame(index=[str(g) for g in output_gene_names])
        )
        
        adata_reconstructed.var[var_col_name] = adata_reconstructed.var_names
        
        # Add metadata
        reconstruction_info = {
            'library_size_used': library_size,
            'gene_column_name': var_col_name,
            'preprocessing': {
                'normalize_per_cell': self.normalize_per_cell,
                'log1p_transform': self.log1p_transform,
                'scale': self.scale,
                'target_sum': self.target_sum,
                'computed_median': self.computed_median,
                'effective_target_sum': self.effective_target_sum
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if self.use_external_decoder:
            reconstruction_info['external_decoder'] = True
            reconstruction_info['encoder_genes'] = len(self.preprocessor.gene_names)
            reconstruction_info['decoder_output_genes'] = len(self.external_decoder_gene_names)
        else:
            reconstruction_info['external_decoder'] = False
            reconstruction_info['model_genes'] = len(self.preprocessor.gene_names)
        
        adata_reconstructed.uns['reconstruction_info'] = reconstruction_info

        # Save to h5ad
        output_path_abs = os.path.abspath(output_path)
        print(f"💾 Saving reconstructed AnnData to: {output_path_abs}")
        adata_reconstructed.write_h5ad(output_path_abs)
        
        print(f"✅ Reconstruction complete!")
        print(f"   Output saved to: {output_path_abs}")
        print(f"   Output shape: {adata_reconstructed.shape}")
        print(f"   Genes: {adata_reconstructed.n_vars}")
        print(f"   Cells: {adata_reconstructed.n_obs}")
        
        return adata_reconstructed

    def _infer_library_size_from_backed(self, adata_backed: sc.AnnData) -> float:
        """Compute library size from backed AnnData by iterating through chunks."""
        n_obs = adata_backed.n_obs
        all_sums = []
        for start in range(0, n_obs, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_obs)
            chunk = adata_backed[start:end].to_memory()
            if hasattr(chunk.X, 'toarray'):
                sums = np.array(chunk.X.sum(axis=1)).flatten()
            else:
                sums = chunk.X.sum(axis=1)
            all_sums.append(sums)
        counts_per_cell = np.concatenate(all_sums)
        median_lib_size = np.median(counts_per_cell)
        print(f"Inferred library size from backed data: {median_lib_size:.2f} (median)")
        return median_lib_size

    def _auto_select_gene_column(self, adata: sc.AnnData, gene_name_col: Optional[str]) -> Optional[str]:
        """Auto-select the best gene column based on overlap with training genes."""
        used_gene_col = gene_name_col
        try:
            training_genes = set(getattr(self.preprocessor, 'gene_names', []) or [])
            var_cols = list(adata.var.columns)
            candidate_cols = []
            train_gene_col = getattr(self.preprocessor, 'train_gene_col', None)
            if train_gene_col:
                candidate_cols.append(train_gene_col)
            candidate_cols.extend(['gene_name', 'gene_symbol', 'gene_symbols', 'symbol', 
                                   'feature_name', 'feature_names', 'Gene', 'ensembl_gene_id', 'gene_id'])
            seen = set()
            candidate_cols = [c for c in candidate_cols if (c not in seen and not (seen.add(c)))]
            
            def overlap_for(col: Optional[str]) -> int:
                if col is None or col not in var_cols:
                    return -1
                try:
                    names = {str(x).strip().upper() for x in adata.var[col].tolist()}
                    return len({g.strip().upper() for g in training_genes} & names)
                except Exception:
                    return -1
            
            force_auto = (gene_name_col is None) or (isinstance(gene_name_col, str) and gene_name_col.lower() == 'auto')
            current_overlap = overlap_for(gene_name_col) if (gene_name_col and not force_auto) else -1
            best_col, best_overlap = None, -1
            for col in candidate_cols:
                ov = overlap_for(col)
                if ov > best_overlap:
                    best_overlap, best_col = ov, col
            if force_auto or best_overlap > current_overlap:
                used_gene_col = best_col
                if used_gene_col:
                    print(f"Auto-selected gene column: {used_gene_col} (overlap={best_overlap}/{len(training_genes)})")
        except Exception as e:
            print(f"Gene column auto-selection failed: {e}")
        return used_gene_col

    def _process_chunk_reconstruction(self, 
                                       chunk_adata: sc.AnnData,
                                       used_gene_col: Optional[str],
                                       batch_size: int,
                                       library_size: float) -> np.ndarray:
        """Process a single chunk: preprocess, run model, inverse preprocess.
        
        Cell filtering is disabled during chunked processing to preserve cell count.
        """
        # Temporarily disable cell filtering to preserve all cells
        original_filter_setting = getattr(self.preprocessor, 'filter_min_counts', False)
        self.preprocessor.filter_min_counts = False
        
        try:
            # Preprocess chunk
            if hasattr(self.preprocessor, 'chunk_size'):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                try:
                    chunk_adata.write_h5ad(temp_path)
                    X_processed = self.preprocessor.transform(temp_path, used_gene_col)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                X_processed = self.preprocessor.transform(chunk_adata, used_gene_col)
        finally:
            # Restore original filter setting
            self.preprocessor.filter_min_counts = original_filter_setting
        
        # Run model inference
        dataset = TensorDataset(torch.tensor(X_processed, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        model_to_use = self.hybrid_model if self.use_external_decoder else self.model
        model_to_use.eval()
        
        reconstructions_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_x = batch_x.to(self.device)
                if self._gene_presence_input:
                    batch_x = batch_x + self._presence_shift

                if self.use_external_decoder:
                    reconstruction = model_to_use(batch_x)
                else:
                    _, _, reconstruction = self.model.forward_mask(batch_x)
                    reconstruction = torch.nn.functional.softplus(reconstruction)
                reconstructions_list.append(reconstruction.cpu().numpy())
        
        reconstructed = np.concatenate(reconstructions_list, axis=0)
        
        # Apply inverse preprocessing (minimal logging for chunks)
        result = reconstructed.copy()
        
        # Inverse scaling
        if not self.use_external_decoder and self.scale and self.scaler_params is not None:
            mean = self.scaler_params.get('mean')
            std = self.scaler_params.get('std')
            if mean is not None and std is not None:
                result = result * std + mean
        
        # Inverse log1p
        if self.log1p_transform:
            result = np.expm1(result)
            result = np.maximum(result, 0)
        
        # Inverse library normalization
        if self.normalize_per_cell:
            prediction_library_size = result.sum(axis=1, keepdims=True)
            prediction_library_size = np.where(prediction_library_size == 0, 1.0, prediction_library_size)
            result = (result / prediction_library_size) * library_size
        
        return result.astype(np.float32)

    def reconstruct_and_save_chunked(self,
                                      adata_backed: sc.AnnData,
                                      output_path: str,
                                      gene_name_col: Optional[str] = None,
                                      batch_size: int = 512) -> str:
        """
        Reconstruct counts in chunks and save incrementally to h5ad file.
        
        Args:
            adata_backed: Backed AnnData object (opened with backed='r')
            output_path: Output path for reconstructed h5ad file
            gene_name_col: Gene name column (auto-detect if None)
            batch_size: Batch size for model inference
            
        Returns:
            Path to the output file
        """
        n_obs = adata_backed.n_obs
        print(f"🧬 Chunked reconstruction for {n_obs:,} cells (threshold: {CHUNK_THRESHOLD:,})")
        print(f"   Chunk size: {CHUNK_SIZE:,}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Auto-select gene column using first chunk for var info
        first_chunk = adata_backed[0:min(1000, n_obs)].to_memory()
        used_gene_col = self._auto_select_gene_column(first_chunk, gene_name_col)
        
        # Compute library size from backed data
        library_size = self._infer_library_size_from_backed(adata_backed)
        
        # Determine output genes
        if self.use_external_decoder:
            output_gene_names = self.external_decoder_gene_names
            var_col_name = 'ensembl_gene_id'
        else:
            output_gene_names = self.preprocessor.gene_names
            var_col_name = getattr(self.preprocessor, 'train_gene_col', None)
            if gene_name_col and gene_name_col != 'auto':
                var_col_name = gene_name_col
            if not var_col_name:
                var_col_name = 'ensembl_gene_id'
        
        n_vars = len(output_gene_names)
        
        # Read full obs DataFrame (metadata is typically small)
        obs_df = adata_backed.obs.copy()
        var_df = pd.DataFrame(index=[str(g) for g in output_gene_names])
        var_df[var_col_name] = var_df.index
        
        # Create output h5ad file structure
        output_path_abs = os.path.abspath(output_path)
        print(f"📁 Creating output h5ad: {output_path_abs}")
        
        # Create shell AnnData with minimal X, then replace with h5py
        shell_adata = sc.AnnData(
            X=np.zeros((1, n_vars), dtype=np.float32),  # minimal placeholder
            obs=obs_df.iloc[[0]],  # single row placeholder
            var=var_df
        )
        
        # Add reconstruction metadata
        reconstruction_info = {
            'library_size_used': library_size,
            'gene_column_name': var_col_name,
            'chunked_processing': True,
            'chunk_size': CHUNK_SIZE,
            'preprocessing': {
                'normalize_per_cell': self.normalize_per_cell,
                'log1p_transform': self.log1p_transform,
                'scale': self.scale,
                'target_sum': self.target_sum,
                'computed_median': self.computed_median,
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        if self.use_external_decoder:
            reconstruction_info['external_decoder'] = True
            reconstruction_info['encoder_genes'] = len(self.preprocessor.gene_names)
            reconstruction_info['decoder_output_genes'] = len(self.external_decoder_gene_names)
        else:
            reconstruction_info['external_decoder'] = False
            reconstruction_info['model_genes'] = len(self.preprocessor.gene_names)
        
        shell_adata.uns['reconstruction_info'] = reconstruction_info
        shell_adata.write_h5ad(output_path_abs)
        del shell_adata
        
        # Rewrite obs and X with correct dimensions using h5py
        with h5py.File(output_path_abs, 'a') as f:
            # Delete placeholder X and create correctly-sized dataset
            del f['X']
            dset = f.create_dataset('X', shape=(n_obs, n_vars), dtype='float32',
                                    chunks=(min(CHUNK_SIZE, n_obs), n_vars))
            dset.attrs['encoding-type'] = 'array'
            dset.attrs['encoding-version'] = '0.2.0'
            
            # Rewrite obs with correct number of rows
            if 'obs' in f:
                del f['obs']
            obs_group = f.create_group('obs')
            obs_group.attrs['encoding-type'] = 'dataframe'
            obs_group.attrs['encoding-version'] = '0.2.0'
            obs_group.attrs['_index'] = '_index'
            # Write index
            index_data = np.array(obs_df.index.astype(str), dtype='S')
            obs_group.create_dataset('_index', data=index_data)
            obs_group['_index'].attrs['encoding-type'] = 'string-array'
            obs_group['_index'].attrs['encoding-version'] = '0.2.0'
            # Write columns
            col_order = []
            for col in obs_df.columns:
                col_order.append(col)
                col_data = obs_df[col].values
                if pd.api.types.is_categorical_dtype(col_data):
                    # Handle categorical
                    cat_group = obs_group.create_group(col)
                    cat_group.attrs['encoding-type'] = 'categorical'
                    cat_group.attrs['encoding-version'] = '0.2.0'
                    codes = col_data.codes.astype(np.int32)
                    cat_group.create_dataset('codes', data=codes)
                    categories = np.array(col_data.categories.astype(str), dtype='S')
                    cat_group.create_dataset('categories', data=categories)
                    cat_group['categories'].attrs['encoding-type'] = 'string-array'
                    cat_group['categories'].attrs['encoding-version'] = '0.2.0'
                    cat_group.attrs['ordered'] = col_data.ordered
                elif pd.api.types.is_bool_dtype(col_data):
                    obs_group.create_dataset(col, data=col_data.astype(bool))
                elif pd.api.types.is_numeric_dtype(col_data):
                    obs_group.create_dataset(col, data=col_data)
                else:
                    # String column
                    str_data = np.array(col_data.astype(str), dtype='S')
                    obs_group.create_dataset(col, data=str_data)
                    obs_group[col].attrs['encoding-type'] = 'string-array'
                    obs_group[col].attrs['encoding-version'] = '0.2.0'
            obs_group.attrs['column-order'] = col_order
        
        # Process chunks and write to output
        n_chunks = (n_obs + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"🔄 Processing {n_chunks} chunks...")
        
        for chunk_idx, start in enumerate(range(0, n_obs, CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, n_obs)
            print(f"\n📦 Chunk {chunk_idx + 1}/{n_chunks}: cells [{start:,}, {end:,})")
            
            # Read chunk to memory
            chunk_adata = adata_backed[start:end].to_memory()
            if hasattr(chunk_adata.X, 'toarray'):
                chunk_adata.X = chunk_adata.X.toarray()
            
            # Process chunk
            reconstructed_chunk = self._process_chunk_reconstruction(
                chunk_adata, used_gene_col, batch_size, library_size
            )
            
            # Write chunk to output
            with h5py.File(output_path_abs, 'a') as f:
                f['X'][start:end, :] = reconstructed_chunk
            
            print(f"   ✅ Written {reconstructed_chunk.shape[0]:,} cells to output")
            
            # Free memory
            del chunk_adata, reconstructed_chunk
        
        print(f"\n✅ Chunked reconstruction complete!")
        print(f"   Output: {output_path_abs}")
        print(f"   Shape: ({n_obs:,}, {n_vars:,})")
        
        return output_path_abs


def load_reconstruction_model(model_dir: str, 
                              device: str = 'cuda', 
                              checkpoint_path: Optional[str] = None, 
                              preprocessor_path: Optional[str] = None,
                              external_decoder_path: Optional[str] = None,
                              decoder_config_path: Optional[str] = None) -> OccamReconstructor:
    """
    Load a trained Occam model for reconstruction.
    
    When preprocessor_path is None the preprocessor is reconstructed directly
    from checkpoint data (gene_names + dataset_info).
    
    Args:
        model_dir: Directory containing saved model checkpoint(s)
        device: Device to load model on
        checkpoint_path: Optional path or filename of checkpoint to load
        preprocessor_path: Optional path to a legacy preprocessor.pkl file.
        external_decoder_path: Optional path to external SEResNetDecoder checkpoint.
        decoder_config_path: Optional separate checkpoint for decoder config/gene metadata.
        
    Returns:
        Initialized reconstruction pipeline
    """
    # Resolve model checkpoint path
    model_path = None
    if checkpoint_path:
        candidate = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(model_dir, checkpoint_path)
        if os.path.exists(candidate):
            model_path = candidate
        else:
            raise FileNotFoundError(f"Specified checkpoint not found: {candidate}")
    else:
        model_files = ['best_model.pth', 'final_model.pth', 'model_checkpoint.pth']
        for fname in model_files:
            full_path = os.path.join(model_dir, fname)
            if os.path.exists(full_path):
                model_path = full_path
                break
        if model_path is None:
            raise FileNotFoundError(f"No model checkpoint found in {model_dir}")

    # Resolve preprocessor path only if explicitly provided
    resolved_preprocessor = None
    if preprocessor_path is not None:
        explicit = preprocessor_path if os.path.isabs(preprocessor_path) else os.path.join(model_dir, preprocessor_path)
        if not os.path.exists(explicit):
            alt = os.path.abspath(preprocessor_path)
            if os.path.exists(alt):
                explicit = alt
            else:
                raise FileNotFoundError(f"Specified preprocessor not found: {preprocessor_path}")
        resolved_preprocessor = explicit

    return OccamReconstructor(model_path, resolved_preprocessor, device, external_decoder_path, decoder_config_path)


def reconstruct_from_h5ad(
    adata_path: str,
    model_dir: str,
    output_path: str,
    gene_name_col: Optional[str] = None,
    batch_size: int = 512,
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
    external_decoder_path: Optional[str] = None,
    decoder_config_path: Optional[str] = None
) -> Union[sc.AnnData, str]:
    """
    Run reconstruction on h5ad file and save results.
    
    Automatically uses chunked processing for files with >300k samples to reduce
    memory usage.
    
    Args:
        adata_path: Path to input h5ad file (raw counts)
        model_dir: Directory containing model and preprocessor
        output_path: Output path for reconstructed h5ad file
        gene_name_col: Gene name column (auto-detect if None)
        batch_size: Batch size for reconstruction
        device: Device to use
        checkpoint_path: Optional specific checkpoint to use
        preprocessor_path: Optional specific preprocessor to use
        external_decoder_path: Optional path to external SEResNetDecoder checkpoint.
                               If provided, uses Occam encoder + external decoder.
        decoder_config_path: Optional separate checkpoint for decoder config/gene metadata.
        
    Returns:
        AnnData with reconstructed counts (standard mode) or path to output file (chunked mode)
    """
    # Load model and preprocessor
    reconstruction_pipeline = load_reconstruction_model(
        model_dir, device, checkpoint_path, preprocessor_path, external_decoder_path, decoder_config_path
    )
    
    # Check file size using backed mode first
    print(f"📂 Checking file size: {adata_path}")
    adata_backed = sc.read_h5ad(adata_path, backed='r')
    n_obs = adata_backed.n_obs
    print(f"   Total samples: {n_obs:,}")
    
    # Use chunked mode for large files
    if n_obs > CHUNK_THRESHOLD:
        print(f"🔄 Using CHUNKED processing (n_obs={n_obs:,} > threshold={CHUNK_THRESHOLD:,})")
        output_file = reconstruction_pipeline.reconstruct_and_save_chunked(
            adata_backed, output_path, gene_name_col, batch_size
        )
        adata_backed.file.close()
        return output_file
    
    # Standard mode for smaller files
    print(f"📦 Using STANDARD processing (n_obs={n_obs:,} <= threshold={CHUNK_THRESHOLD:,})")
    adata_backed.file.close()
    
    # Load full data
    adata = sc.read_h5ad(adata_path)
    print(f"Input data shape: {adata.shape}")
    
    # Run reconstruction and save
    adata_reconstructed = reconstruction_pipeline.reconstruct_and_save(
        adata, output_path, gene_name_col, batch_size
    )
    
    return adata_reconstructed


def decode_from_embeddings(
    embeddings_path: str,
    decoder_path: str,
    output_path: str,
    metadata_csv: Optional[str] = None,
    library_size: Optional[float] = None,
    batch_size: int = 1024,
    device: str = 'cuda',
    decoder_config_path: Optional[str] = None
) -> sc.AnnData:
    """
    Decode pre-computed embeddings using an external decoder.
    
    Takes embeddings (.npy or .h5 with 'embeddings' key) and decodes them to raw counts
    using an external decoder checkpoint. Applies postprocessing (expm1 + library 
    renormalization) and saves as h5ad with gene names.
    
    Args:
        embeddings_path: Path to .npy or .h5 embeddings file
        decoder_path: Path to external decoder .pt checkpoint
        output_path: Output path for decoded h5ad file
        metadata_csv: Optional path to CSV with cell-level obs metadata
        library_size: Library size for renormalization (default: from decoder checkpoint or 10000)
        batch_size: Batch size for decoder inference
        device: Device to use (cuda or cpu)
        decoder_config_path: Optional separate checkpoint for decoder config/gene metadata
        
    Returns:
        AnnData object with decoded raw counts
    """
    print("=" * 70)
    print("Decoding Embeddings with External Decoder")
    print("=" * 70)
    
    # Load embeddings
    print(f"\n📂 Loading embeddings from: {embeddings_path}")
    if embeddings_path.endswith('.npy'):
        embeddings = np.load(embeddings_path)
        print(f"   Loaded .npy: shape={embeddings.shape}")
    elif embeddings_path.endswith('.h5'):
        with h5py.File(embeddings_path, 'r') as f:
            if 'embeddings' in f:
                embeddings = f['embeddings'][:]
            else:
                raise KeyError(f"No 'embeddings' key in HDF5 file. Available keys: {list(f.keys())}")
        print(f"   Loaded .h5: shape={embeddings.shape}")
    else:
        raise ValueError(f"Unsupported embeddings format: {embeddings_path} (expected .npy or .h5)")
    
    n_cells, n_dims = embeddings.shape
    print(f"   Cells: {n_cells:,}, Dimensions: {n_dims}")
    
    # Load decoder
    print(f"\n🔄 Loading external decoder from: {decoder_path}")
    from occam.decoder.external import load_decoder
    
    decoder_model, gene_names, gene_to_index, decoder_config = load_decoder(
        decoder_path, device=device, config_checkpoint_path=decoder_config_path
    )
    
    print(f"   Decoder input dim: {decoder_config['input_dim']}")
    print(f"   Decoder output dim: {decoder_config['output_dim']} genes")
    
    # Verify embedding dimension matches decoder
    if embeddings.shape[1] != decoder_config['input_dim']:
        raise ValueError(
            f"Embedding dimension ({embeddings.shape[1]}) does not match "
            f"decoder input dimension ({decoder_config['input_dim']})"
        )
    
    # Determine library size
    if library_size is None:
        library_size = decoder_config.get('target_sum')
        if library_size is None:
            library_size = 1e4
            print(f"\n📊 No library size found in decoder checkpoint, using default: {library_size:.0f}")
        else:
            print(f"\n📊 Using library size from decoder checkpoint: {library_size:.0f}")
    else:
        print(f"\n📊 Using provided library size: {library_size:.0f}")
    
    # Run decoder inference
    print(f"\n🧬 Running decoder inference on {n_cells:,} cells...")
    from occam.decoder.external import run_inference
    predictions = run_inference(decoder_model, embeddings, device=device, batch_size=batch_size)
    
    print(f"   Decoder output shape: {predictions.shape}")
    print(f"   Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Postprocessing: inverse log1p
    print(f"\n🔄 Applying inverse transformations...")
    print(f"   [STEP 1] Inverse log1p (expm1)...")
    reconstructed_raw = np.expm1(predictions)
    reconstructed_raw = np.maximum(reconstructed_raw, 0)  # Clip negatives
    print(f"   [STEP 2] Inverse library normalization (target: {library_size:.2f})...")
    
    # Library renormalization
    prediction_library_size = reconstructed_raw.sum(axis=1, keepdims=True)
    prediction_library_size = np.where(prediction_library_size == 0, 1.0, prediction_library_size)
    reconstructed_raw = (reconstructed_raw / prediction_library_size) * library_size
    
    print(f"   ✅ Postprocessing complete")
    print(f"   Output shape: {reconstructed_raw.shape}")
    print(f"   Output range: [{reconstructed_raw.min():.4f}, {reconstructed_raw.max():.4f}]")
    
    # Load metadata if provided
    obs_df = None
    if metadata_csv:
        print(f"\n📋 Loading cell metadata from: {metadata_csv}")
        obs_df = pd.read_csv(metadata_csv, index_col=0)
        print(f"   Loaded {len(obs_df)} cells with {len(obs_df.columns)} columns")
        
        if len(obs_df) != n_cells:
            print(f"   ⚠️  Warning: metadata has {len(obs_df)} cells, but embeddings have {n_cells} cells")
            if len(obs_df) > n_cells:
                obs_df = obs_df.iloc[:n_cells]
                print(f"   Trimmed metadata to {n_cells} cells")
    
    # Build output AnnData
    print(f"\n📦 Creating output AnnData...")
    var_df = pd.DataFrame(index=[str(g) for g in gene_names])
    var_df['ensembl_gene_id'] = var_df.index
    
    if obs_df is not None:
        adata_decoded = sc.AnnData(
            X=reconstructed_raw,
            obs=obs_df,
            var=var_df
        )
    else:
        adata_decoded = sc.AnnData(
            X=reconstructed_raw,
            var=var_df
        )
    
    # Add decode metadata
    decode_info = {
        'library_size_used': float(library_size),
        'gene_column_name': 'ensembl_gene_id',
        'decoder_genes': len(gene_names),
        'decoder_checkpoint': os.path.basename(decoder_path),
        'embeddings_path': embeddings_path,
        'postprocessing': {
            'inverse_log1p': True,
            'library_normalization': True
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    if decoder_config_path:
        decode_info['decoder_config_path'] = os.path.basename(decoder_config_path)
    
    adata_decoded.uns['decode_info'] = decode_info
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to h5ad
    output_path_abs = os.path.abspath(output_path)
    print(f"\n💾 Saving decoded AnnData to: {output_path_abs}")
    adata_decoded.write_h5ad(output_path_abs)
    
    print(f"\n✅ Decoding complete!")
    print(f"   Output saved to: {output_path_abs}")
    print(f"   Shape: {adata_decoded.shape}")
    print(f"   Cells: {adata_decoded.n_obs:,}")
    print(f"   Genes: {adata_decoded.n_vars:,}")
    print("=" * 70)
    
    return adata_decoded


# Backward compatibility aliases
OccamReconstruction = OccamReconstructor
