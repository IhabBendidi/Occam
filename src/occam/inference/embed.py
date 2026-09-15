"""
Inference pipeline for Occam with h5ad data support

Provides inference capabilities with gene alignment for trained Occam models.
Handles loading trained models and preprocessing new data to match training format.
"""

import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Dict, Optional, Tuple, Any, Union
import pickle
from pathlib import Path
from datetime import datetime

# Chunked processing constants
CHUNK_THRESHOLD = 300_000  # Activate chunked processing when n_obs > this
CHUNK_SIZE = 150_000  # Number of samples per chunk

from occam.model import AutoEncoder
from occam.mae.config import PretrainConfig
from occam.run_dir import get_output_manager, ExperimentManager
from occam.evaluation.perturbation import tvn_on_controls, aggregate, known_relationship_benchmark, DEFAULT_RECALL_THR_PAIRS


def _make_output_prefix(input_path: str, model_dir: str, checkpoint_path: Optional[str] = None) -> str:
    """Build a descriptive output prefix: {data}__{experiment}__{checkpoint}__{timestamp}."""
    data_stem = os.path.splitext(os.path.basename(input_path))[0]

    # Extract experiment name from model_dir (strip trailing /checkpoints if present)
    exp_dir = model_dir.rstrip("/")
    if os.path.basename(exp_dir) == "checkpoints":
        exp_dir = os.path.dirname(exp_dir)
    experiment = os.path.basename(exp_dir)

    ckpt_stem = ""
    if checkpoint_path:
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parts = [data_stem, experiment]
    if ckpt_stem:
        parts.append(ckpt_stem)
    parts.append(timestamp)
    return "__".join(parts)


class CheckpointPreprocessor:
    """Lightweight preprocessor reconstructed from checkpoint data.

    Provides the same interface that inference/reconstruction pipelines expect
    (gene_names, transform, is_fitted, etc.) without requiring an external
    preprocessor.pkl file.  Applies NormalizePerCell + Log1p only (no scaling).
    """

    def __init__(self, gene_names, target_sum=1e4, train_gene_col=None):
        self.gene_names = [str(g).strip() for g in gene_names]
        self.target_sum = target_sum
        self.train_gene_col = train_gene_col
        self.is_fitted = True
        self.filter_min_counts = False
        self.normalize_per_cell = True
        self.log1p_transform = True
        self.scale = False
        self.scaler_params = None
        self.computed_median = None

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        """Build from a loaded checkpoint dict."""
        gene_names = checkpoint.get('gene_names')
        if gene_names is None:
            ds_info = checkpoint.get('dataset_info', {})
            gene_names = ds_info.get('gene_names')
        if gene_names is None:
            raise ValueError(
                "Checkpoint contains no 'gene_names' or 'dataset_info.gene_names'; "
                "cannot build preprocessor from checkpoint."
            )
        ds_info = checkpoint.get('dataset_info', {})
        target_sum = ds_info.get('target_sum')
        if target_sum is None:
            target_sum = checkpoint.get('target_sum', 1e4)
        train_gene_col = ds_info.get('gene_name_col')
        return cls(gene_names, target_sum=target_sum, train_gene_col=train_gene_col)

    _GENE_COL_CANDIDATES = [
        "gene_name", "gene_symbol", "gene_symbols", "symbol", "Gene",
        "feature_name", "feature_names", "ensembl_gene_id", "gene_id",
    ]

    def _resolve_gene_col(self, adata, gene_name_col=None):
        """Determine the best gene-name column for *adata*.

        Priority: explicit arg > train_gene_col > candidate overlap > index.
        """
        if gene_name_col and gene_name_col in adata.var.columns:
            return gene_name_col

        training_set = {str(g).strip().upper() for g in self.gene_names}

        def _overlap(col):
            names = {str(x).strip().upper() for x in adata.var[col].tolist()}
            return len(training_set & names)

        candidates = []
        if self.train_gene_col and self.train_gene_col != "index" and self.train_gene_col in adata.var.columns:
            candidates.append(self.train_gene_col)
        candidates.extend(c for c in self._GENE_COL_CANDIDATES if c in adata.var.columns and c not in candidates)

        best_col, best_ov = None, 0
        for col in candidates:
            ov = _overlap(col)
            if ov > best_ov:
                best_col, best_ov = col, ov

        # Also check var_names (index)
        idx_ov = len(training_set & {str(g).strip().upper() for g in adata.var_names.tolist()})
        if idx_ov > best_ov:
            return None  # signals: use var_names

        if best_col is not None:
            print(f"Auto-detected gene column: {best_col} (overlap={best_ov}/{len(self.gene_names)})")
            return best_col

        return None  # fallback to var_names

    def _get_gene_names(self, adata, gene_name_col=None):
        """Extract gene names from AnnData."""
        if gene_name_col and gene_name_col in adata.var.columns:
            return [str(g).strip() for g in adata.var[gene_name_col].tolist()]
        return [str(g).strip() for g in adata.var_names.tolist()]

    def transform(self, adata, gene_name_col=None):
        """Align genes to training vocabulary, apply NormalizePerCell + Log1p."""
        resolved_col = self._resolve_gene_col(adata, gene_name_col)
        input_genes = self._get_gene_names(adata, resolved_col)
        input_gene_to_idx = {g: i for i, g in enumerate(input_genes)}

        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        n_cells = X.shape[0]
        n_genes = len(self.gene_names)
        X_aligned = np.zeros((n_cells, n_genes), dtype=np.float32)

        matched = 0
        for j, gene in enumerate(self.gene_names):
            if gene in input_gene_to_idx:
                X_aligned[:, j] = X[:, input_gene_to_idx[gene]]
                matched += 1
        print(f"Gene alignment: {matched}/{n_genes} training genes found in input")

        # NormalizePerCell
        row_sums = X_aligned.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        X_aligned = X_aligned / row_sums * (self.target_sum or 1e4)

        # Log1p
        X_aligned = np.log1p(X_aligned)
        return X_aligned


class OccamEmbedder:
    """Inference pipeline for trained Occam models."""
    
    def __init__(self, 
                 model_path: str,
                 preprocessor_path: Optional[str] = None,
                 device: str = 'cuda'):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to saved model checkpoint
            preprocessor_path: Optional path to saved preprocessor pickle.
                If None, a CheckpointPreprocessor is built from the checkpoint.
            device: Device to use for inference
        """
        self.device = device
        
        # Load model checkpoint first to get the correct configuration
        checkpoint = torch.load(model_path, map_location=device)
        
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
        
        # Handle both old and new checkpoint formats
        if 'config' in checkpoint:
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
                    print(f"⚠️  Gene count mismatch detected:")
                    print(f"   - Config num_genes: {num_genes_from_config}")
                    print(f"   - Preprocessor genes: {num_genes_from_preprocessor}")
                    print(f"   - Checkpoint actual genes: {num_genes_from_checkpoint}")
                    print(f"   - Using checkpoint value: {num_genes_from_checkpoint} ✅")
            else:
                # Fallback to preprocessor (most conservative)
                num_genes = num_genes_from_preprocessor
                print(f"⚠️  Could not infer num_genes from checkpoint")
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
        # Get actual model size from loaded weights
        actual_num_genes = None
        try:
            # Check encoder.1.weight for actual size
            for name, param in self.model.named_parameters():
                if name.endswith('encoder.1.weight'):
                    actual_num_genes = param.shape[1]
                    break
            if actual_num_genes is None:
                for name, param in self.model.named_parameters():
                    if name.endswith('decoder.weight'):
                        actual_num_genes = param.shape[0]
                        break
        except Exception:
            pass
        
        if actual_num_genes is not None:
            print(f"Model loaded with {actual_num_genes} genes (actual)")
        elif hasattr(self, 'config') and self.config:
            print(f"Model expects {self.config.num_genes} genes (from config)")
        else:
            print(f"Model expects {len(self.preprocessor.gene_names)} genes")
        
    def predict_embeddings(self, 
                          adata: sc.AnnData,
                          gene_name_col: Optional[str] = None,
                          batch_size: int = 512) -> np.ndarray:
        """
        Predict embeddings for input data with proper gene alignment.
        
        Args:
            adata: Input AnnData object
            gene_name_col: Gene name column (auto-detect if None)
            batch_size: Batch size for inference
            
        Returns:
            Numpy array of embeddings with same cell order as input
        """
        print(f"Processing data with shape {adata.shape}")
        
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

        # Disable cell filtering during inference to preserve all cells
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
        
        # Create dataset and dataloader
        # No labels needed for inference (unsupervised)
        dataset = TensorDataset(torch.tensor(X_processed, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Running inference with batch size {batch_size}...")
        
        # Run inference
        embeddings_list = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    batch_x = batch[0]
                else:
                    batch_x = batch
                    
                batch_x = batch_x.to(self.device)
                if self._gene_presence_input:
                    batch_x = batch_x + self._presence_shift  # All genes present in single-file inference

                # Get embeddings from encoder
                embeddings = self.model.feature(batch_x)
                embeddings_list.append(embeddings.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings_list, axis=0)
        
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        print(f"   - Input cells: {adata.shape[0]}")
        print(f"   - Output embeddings: {embeddings.shape[0]}")
        print(f"   - Cell order preserved: {adata.shape[0] == embeddings.shape[0]}")
        
        return embeddings
    
    def predict_and_save(self, 
                        adata: sc.AnnData, 
                        output_path: str,
                        gene_name_col: Optional[str] = None,
                        batch_size: int = 512,
                        save_processed: bool = False) -> np.ndarray:
        """
        Predict embeddings and save results with proper interface handling.
        
        Args:
            adata: Input AnnData object
            output_path: Base path for saving results
            gene_name_col: Gene name column (auto-detect if None)
            batch_size: Batch size for inference
            save_processed: Whether to save processed data
            
        Returns:
            Embeddings array
        """
        print(f"🧬 Generating embeddings for {adata.shape[0]:,} cells...")
        
        # Use predict_embeddings which handles the preprocessor interface correctly
        embeddings = self.predict_embeddings(adata, gene_name_col, batch_size)
        
        # Save embeddings
        embeddings_path = f"{output_path}_embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"💾 Embeddings saved to: {embeddings_path}")
        
        # Save processed data if requested
        if save_processed:
            # For memory-efficient preprocessor, we need to get the processed data properly
            # The embeddings generation already did the preprocessing, so we can note this
            print(f"ℹ️  Processed data was used for embedding generation (not separately saved)")
        
        # Save metadata
        metadata = {
            'input_shape': adata.shape,
            'output_shape': embeddings.shape,
            'gene_name_col': gene_name_col,
            'model_genes': len(self.preprocessor.gene_names),
            'batch_size': batch_size,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = f"{output_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        print(f"📋 Metadata saved to: {metadata_path}")
        
        return embeddings
    
    def _save_inference_report(self, 
                              adata: sc.AnnData, 
                              embeddings: np.ndarray,
                              save_path: str,
                              gene_name_col: Optional[str]):
        """Save a summary report of the inference."""
        
        # Get gene names for comparison
        try:
            input_gene_names = self.preprocessor._get_gene_names(adata, gene_name_col)
        except:
            input_gene_names = []
        
        # Gene alignment statistics
        input_genes = set(input_gene_names)
        training_genes = set(self.preprocessor.gene_names)
        common_genes = input_genes & training_genes
        
        report = {
            'inference_summary': {
                'input_cells': adata.n_obs,
                'input_genes': adata.n_vars,
                'training_genes': len(training_genes),
                'common_genes': len(common_genes),
                'gene_overlap_percentage': len(common_genes) / len(training_genes) * 100 if len(training_genes) > 0 else 0,
                'embedding_dimension': embeddings.shape[1]
            },
            'gene_alignment': {
                'genes_only_in_input': list(input_genes - training_genes)[:50],  # Limit for readability
                'genes_only_in_training': list(training_genes - input_genes)[:50],
                'common_genes_count': len(common_genes)
            },
            'data_statistics': {
                'embedding_mean': float(np.mean(embeddings)),
                'embedding_std': float(np.std(embeddings)),
                'embedding_min': float(np.min(embeddings)),
                'embedding_max': float(np.max(embeddings))
            }
        }
        
        # Add model info if available
        if self.config is not None:
            report['model_info'] = {
                'model_size': self.config.model_size,
                'hidden_size': getattr(self.config, 'hidden_size', None),
                'mask_probability': getattr(self.config, 'mask_probability', None)
            }
        
        # Save report
        with open(f"{save_path}_inference_report.pkl", 'wb') as f:
            pickle.dump(report, f)
        
        # Save human-readable summary
        with open(f"{save_path}_summary.txt", 'w') as f:
            f.write("Occam Inference Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input data: {adata.n_obs} cells × {adata.n_vars} genes\n")
            f.write(f"Training genes: {len(training_genes)}\n")
            f.write(f"Common genes: {len(common_genes)} ({len(common_genes)/len(training_genes)*100:.1f}%)\n")
            f.write(f"Generated embeddings: {embeddings.shape[0]} cells × {embeddings.shape[1]} dimensions\n\n")
            
            if len(common_genes) < len(training_genes) * 0.5:
                f.write("⚠️  WARNING: Less than 50% gene overlap with training data!\n")
                f.write("This may affect embedding quality.\n\n")
            
            f.write(f"Embedding statistics:\n")
            f.write(f"  Mean: {np.mean(embeddings):.4f}\n")
            f.write(f"  Std:  {np.std(embeddings):.4f}\n")
            f.write(f"  Min:  {np.min(embeddings):.4f}\n")
            f.write(f"  Max:  {np.max(embeddings):.4f}\n")
            
            # Add model info if available
            if self.config is not None:
                f.write(f"\nModel configuration:\n")
                f.write(f"  Model size: {self.config.model_size}\n")
                if hasattr(self.config, 'hidden_size') and self.config.hidden_size:
                    f.write(f"  Hidden size: {self.config.hidden_size}\n")
                if hasattr(self.config, 'mask_probability'):
                    f.write(f"  Mask probability: {self.config.mask_probability}\n")
        
        print(f"Inference report saved to {save_path}_summary.txt")
    
    def _save_inference_report_to_experiment(self, 
                                          adata: sc.AnnData, 
                                          embeddings: np.ndarray,
                                          base_name: str,
                                          gene_name_col: Optional[str],
                                          experiment: ExperimentManager):
        """Save inference report to experiment."""
        
        # Get gene overlap stats
        gene_overlap_stats = self._get_gene_overlap_stats(adata, gene_name_col)
        
        # Create comprehensive report
        report = {
            'inference_summary': {
                'input_cells': adata.n_obs,
                'input_genes': adata.n_vars,
                'training_genes': len(gene_overlap_stats['training_genes']),
                'common_genes': len(gene_overlap_stats['common_genes']),
                'gene_overlap_percentage': gene_overlap_stats['overlap_percentage'],
                'embedding_dimension': embeddings.shape[1]
            },
            'gene_alignment': gene_overlap_stats,
            'data_statistics': {
                'embedding_mean': float(np.mean(embeddings)),
                'embedding_std': float(np.std(embeddings)),
                'embedding_min': float(np.min(embeddings)),
                'embedding_max': float(np.max(embeddings))
            }
        }
        
        # Add model info if available
        if self.config is not None:
            report['model_info'] = {
                'model_size': self.config.model_size,
                'hidden_size': getattr(self.config, 'hidden_size', None),
                'mask_probability': getattr(self.config, 'mask_probability', None)
            }
        
        # Save report to experiment
        report_path = experiment.get_path('reports', f'{base_name}_inference_report.pkl')
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        # Save human-readable summary
        summary_path = experiment.get_path('reports', f'{base_name}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Occam Inference Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input data: {adata.n_obs} cells × {adata.n_vars} genes\n")
            f.write(f"Training genes: {len(gene_overlap_stats['training_genes'])}\n")
            f.write(f"Common genes: {len(gene_overlap_stats['common_genes'])} ({gene_overlap_stats['overlap_percentage']:.1f}%)\n")
            f.write(f"Generated embeddings: {embeddings.shape[0]} cells × {embeddings.shape[1]} dimensions\n\n")
            
            if gene_overlap_stats['overlap_percentage'] < 50:
                f.write("⚠️  WARNING: Less than 50% gene overlap with training data!\n")
                f.write("This may affect embedding quality.\n\n")
            
            f.write(f"Embedding statistics:\n")
            f.write(f"  Mean: {np.mean(embeddings):.4f}\n")
            f.write(f"  Std:  {np.std(embeddings):.4f}\n")
            f.write(f"  Min:  {np.min(embeddings):.4f}\n")
            f.write(f"  Max:  {np.max(embeddings):.4f}\n")
            
            # Add model info if available
            if self.config is not None:
                f.write(f"\nModel configuration:\n")
                f.write(f"  Model size: {self.config.model_size}\n")
                if hasattr(self.config, 'hidden_size') and self.config.hidden_size:
                    f.write(f"  Hidden size: {self.config.hidden_size}\n")
                if hasattr(self.config, 'mask_probability'):
                    f.write(f"  Mask probability: {self.config.mask_probability}\n")
        
        print(f"Inference report saved to experiment: {summary_path}")
        experiment.log_progress("Inference report generated and saved")
    
    def _get_gene_overlap_stats(self, adata: sc.AnnData, gene_name_col: Optional[str]) -> Dict[str, Any]:
        """Get gene overlap statistics between input data and training data."""
        
        # Get gene names for comparison
        try:
            input_gene_names = self.preprocessor._get_gene_names(adata, gene_name_col)
        except:
            input_gene_names = []
        
        # Gene alignment statistics
        input_genes = set(input_gene_names)
        training_genes = set(self.preprocessor.gene_names)
        common_genes = input_genes & training_genes
        
        return {
            'input_genes': list(input_genes),
            'training_genes': list(training_genes),
            'common_genes': list(common_genes),
            'genes_only_in_input': list(input_genes - training_genes)[:50],  # Limit for readability
            'genes_only_in_training': list(training_genes - input_genes)[:50],
            'overlap_percentage': len(common_genes) / len(training_genes) * 100 if len(training_genes) > 0 else 0,
            'common_genes_count': len(common_genes)
        }

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

    def _process_chunk_embeddings(self, 
                                   chunk_adata: sc.AnnData,
                                   used_gene_col: Optional[str],
                                   batch_size: int) -> np.ndarray:
        """Process a single chunk: preprocess and get embeddings.
        
        Cell filtering is disabled during chunked processing to preserve cell count.
        """
        original_n_cells = chunk_adata.n_obs
        
        # Temporarily disable cell filtering to preserve all cells
        original_filter_setting = getattr(self.preprocessor, 'filter_min_counts', True)
        self.preprocessor.filter_min_counts = False
        print(f"   [DEBUG] Preprocessor type: {type(self.preprocessor).__name__}")
        print(f"   [DEBUG] Has chunk_size attr: {hasattr(self.preprocessor, 'chunk_size')}")
        print(f"   [DEBUG] Cell filtering disabled: filter_min_counts={self.preprocessor.filter_min_counts} (was {original_filter_setting})")
        
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
        
        # Verify cell count preserved
        if X_processed.shape[0] != original_n_cells:
            print(f"   ⚠️  [DEBUG] Cell count mismatch! Input: {original_n_cells}, Output: {X_processed.shape[0]}")
            print(f"   ⚠️  [DEBUG] Preprocessor type: {type(self.preprocessor).__name__}")
            print(f"   ⚠️  [DEBUG] filter_min_counts after transform: {self.preprocessor.filter_min_counts}")
        
        # Run model inference
        dataset = TensorDataset(torch.tensor(X_processed, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        embeddings_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_x = batch_x.to(self.device)
                if self._gene_presence_input:
                    batch_x = batch_x + self._presence_shift
                embeddings = self.model.feature(batch_x)
                embeddings_list.append(embeddings.cpu().numpy())
        
        return np.concatenate(embeddings_list, axis=0).astype(np.float32)

    def predict_embeddings_chunked(self,
                                    adata_backed: sc.AnnData,
                                    output_path: str,
                                    gene_name_col: Optional[str] = None,
                                    batch_size: int = 512) -> Tuple[str, str]:
        """
        Predict embeddings in chunks and save incrementally.
        
        Args:
            adata_backed: Backed AnnData object (opened with backed='r')
            output_path: Base output path (without extension)
            gene_name_col: Gene name column (auto-detect if None)
            batch_size: Batch size for model inference
            
        Returns:
            Tuple of (npy_path, h5_path) for the saved embeddings
        """
        n_obs = adata_backed.n_obs
        print(f"🧬 Chunked embedding inference for {n_obs:,} cells (threshold: {CHUNK_THRESHOLD:,})")
        print(f"   Chunk size: {CHUNK_SIZE:,}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Auto-select gene column using first chunk for var info
        first_chunk = adata_backed[0:min(1000, n_obs)].to_memory()
        used_gene_col = self._auto_select_gene_column(first_chunk, gene_name_col)
        
        # Determine embedding dimension by running a small batch
        test_chunk = adata_backed[0:min(100, n_obs)].to_memory()
        if hasattr(test_chunk.X, 'toarray'):
            test_chunk.X = test_chunk.X.toarray()
        test_embeddings = self._process_chunk_embeddings(test_chunk, used_gene_col, batch_size)
        embed_dim = test_embeddings.shape[1]
        del test_chunk, test_embeddings
        
        print(f"   Embedding dimension: {embed_dim}")
        
        # Get cell identifiers
        obs_names = list(adata_backed.obs_names)
        
        # Create output files
        npy_path = f"{output_path}_embeddings.npy"
        h5_path = f"{output_path}_embeddings.h5"
        
        # Create HDF5 file with chunked dataset
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('embeddings', shape=(n_obs, embed_dim), dtype='float32',
                             chunks=(min(CHUNK_SIZE, n_obs), embed_dim))
            # Store obs_names as string array
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('obs_names', data=np.array(obs_names, dtype=object), dtype=dt)
        
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
            chunk_embeddings = self._process_chunk_embeddings(
                chunk_adata, used_gene_col, batch_size
            )
            
            # Write chunk to HDF5
            with h5py.File(h5_path, 'a') as f:
                f['embeddings'][start:end, :] = chunk_embeddings
            
            print(f"   ✅ Written {chunk_embeddings.shape[0]:,} embeddings")
            
            # Free memory
            del chunk_adata, chunk_embeddings
        
        # Also save as single .npy file by reading back from h5
        print(f"\n💾 Creating consolidated .npy file...")
        with h5py.File(h5_path, 'r') as f:
            all_embeddings = f['embeddings'][:]
        np.save(npy_path, all_embeddings)
        
        print(f"\n✅ Chunked embedding inference complete!")
        print(f"   HDF5 file: {h5_path}")
        print(f"   NPY file: {npy_path}")
        print(f"   Shape: ({n_obs:,}, {embed_dim})")
        
        return npy_path, h5_path


def load_trained_model(model_dir: str, device: str = 'cuda', checkpoint_path: Optional[str] = None, preprocessor_path: Optional[str] = None) -> OccamEmbedder:
    """
    Load a trained Occam model from directory.
    
    When preprocessor_path is None the preprocessor is reconstructed directly
    from checkpoint data (gene_names + dataset_info), so no external
    preprocessor.pkl file is required.
    
    Args:
        model_dir: Directory containing saved model checkpoint(s)
        device: Device to load model on
        checkpoint_path: Optional path or filename of checkpoint to load. If a relative
            filename is provided, it will be resolved relative to model_dir.
        preprocessor_path: Optional path to a legacy preprocessor.pkl file.
        
    Returns:
        Initialized inference pipeline
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

    return OccamEmbedder(model_path, resolved_preprocessor, device)


def infer_from_h5ad(
    adata_path: str,
    model_dir: str,
    output_path: str,
    gene_name_col: Optional[str] = None,
    batch_size: int = 512,
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
    eval_after_inference: bool = False,
    eval_frequency: int = 0,
    eval_seed: int = 42
) -> Union[np.ndarray, Tuple[str, str]]:
    """
    Run inference on h5ad file and save results.
    
    Automatically uses chunked processing for files with >300k samples to reduce
    memory usage.
    
    Args:
        adata_path: Path to h5ad file
        model_dir: Directory containing model and preprocessor
        output_path: Base path for saving results (without extension)
        gene_name_col: Gene name column (auto-detect if None)
        batch_size: Batch size for inference
        device: Device to use
        checkpoint_path: Optional specific checkpoint to use
        preprocessor_path: Optional specific preprocessor to use
        eval_after_inference: Whether to run evaluation after inference
        eval_frequency: Evaluation frequency for batch processing
        
    Returns:
        Embeddings array (standard mode) or tuple of (npy_path, h5_path) (chunked mode)
    """
    # Load model and preprocessor
    inference_pipeline = load_trained_model(model_dir, device, checkpoint_path, preprocessor_path)
    
    # Check file size using backed mode first
    print(f"📂 Checking file size: {adata_path}")
    adata_backed = sc.read_h5ad(adata_path, backed='r')
    n_obs = adata_backed.n_obs
    print(f"   Total samples: {n_obs:,}")
    
    # Build descriptive output prefix once
    prefix = _make_output_prefix(adata_path, model_dir, checkpoint_path)

    # Use chunked mode for large files
    if n_obs > CHUNK_THRESHOLD:
        print(f"🔄 Using CHUNKED processing (n_obs={n_obs:,} > threshold={CHUNK_THRESHOLD:,})")
        
        # Ensure output directory exists
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        output_base = os.path.join(output_path, prefix)
        
        npy_path, h5_path = inference_pipeline.predict_embeddings_chunked(
            adata_backed, output_base, gene_name_col, batch_size
        )
        
        # Save cell metadata if available
        obs_df = adata_backed.obs
        if obs_df is not None and len(obs_df.columns) > 0:
            metadata_path = f"{output_base}_metadata.csv"
            obs_df.to_csv(metadata_path)
            print(f"Cell metadata saved to {metadata_path}")
        
        adata_backed.file.close()
        return npy_path, h5_path
    
    # Standard mode for smaller files
    print(f"📦 Using STANDARD processing (n_obs={n_obs:,} <= threshold={CHUNK_THRESHOLD:,})")
    adata_backed.file.close()
    
    # Load full data
    print(f"Loading data from {adata_path}")
    adata = sc.read_h5ad(adata_path)
    
    # Ensure dense matrix for inference if sparse
    if hasattr(adata.X, 'toarray'):
        print("Converting sparse matrix to dense for inference...")
        adata.X = adata.X.toarray()
    
    # Generate embeddings using corrected interface
    embeddings = inference_pipeline.predict_embeddings(adata, gene_name_col, batch_size)

    # Optional evaluation on the provided input data
    if eval_after_inference:
        try:
            print(f"\n========== EVALUATION (inference, seed={eval_seed}) ==========")
            # Use same columns as training evaluation
            metadata = adata.obs.copy() if adata.obs is not None else pd.DataFrame(index=np.arange(embeddings.shape[0]))
            pert_colname = "gene_name"
            control_col = "is_control"
            batch_col = "dataset_batch_num"
            control_key = True
            # TVN alignment
            tvn_embeddings = tvn_on_controls(
                embeddings=embeddings,
                metadata=metadata,
                pert_col=control_col,
                control_key=control_key,
                batch_col=batch_col,
            )
            # Metric 1: Perturbation signal consistency
            from occam.evaluation.perturbation import pert_signal_consistency_benchmark
            vc = metadata.get(pert_colname, pd.Series(index=metadata.index)).value_counts() if pert_colname in metadata.columns else pd.Series()
            keys_to_drop = vc[vc == 1].index.tolist() if len(vc) > 0 else []
            neg_ctrl_perts = (
                metadata.loc[metadata[control_col] == control_key, pert_colname].unique().tolist()
                if (pert_colname in metadata.columns and control_col in metadata.columns)
                else []
            )
            np.random.seed(eval_seed)
            cons_res = pert_signal_consistency_benchmark(
                tvn_embeddings,
                metadata,
                pert_col=pert_colname if pert_colname in metadata.columns else (metadata.columns[0] if len(metadata.columns) > 0 else 'pert'),
                neg_ctrl_perts=neg_ctrl_perts,
                keys_to_drop=keys_to_drop,
                stat="cossim",
                num_null_cells=min(50000, tvn_embeddings.shape[0])
            )
            denom = np.sum(~pd.isna(cons_res["pval"])) if "pval" in cons_res.columns else 0
            metric1 = float(np.sum(cons_res.pval <= 0.01)) / float(denom if denom > 0 else 1) * 100.0 if denom > 0 else float('nan')
            print(f"[Eval] Metric1 (percent significant p<=0.01): {metric1:.3f}")
            # Metric 2: Known relationships benchmark
            try:
                all_controls = []
                pert_key = pert_colname if pert_colname in metadata.columns else (metadata.columns[0] if len(metadata.columns) > 0 else 'pert')
                map_data = aggregate(tvn_embeddings, metadata, pert_col=pert_key, keys_to_remove=all_controls)
                env_bmdb = os.environ.get('OCCAM_BMDB_DIR') or os.environ.get('OCCAM_BMDB_DIR')
                candidates = [env_bmdb, os.path.join(os.getcwd(), 'bmdb_database'), os.path.join(os.path.dirname(__file__), 'bmdb_database')]
                benchmark_dir = None
                for c in candidates:
                    if c and os.path.isdir(c):
                        benchmark_dir = c
                        break
                if benchmark_dir is None:
                    raise FileNotFoundError("bmdb_database directory not found. Set OCCAM_BMDB_DIR (or OCCAM_BMDB_DIR) or ensure ./bmdb_database exists.")
                sources = ["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"]
                metrics_df = known_relationship_benchmark(
                    map_data,
                    pert_col=pert_key,
                    benchmark_sources=sources,
                    recall_thr_pairs=DEFAULT_RECALL_THR_PAIRS,
                    benchmark_data_dir=benchmark_dir,
                    min_req_entity_cnt=20,
                    log_stats=True,
                )
                if metrics_df is not None and not metrics_df.empty and 'recall_0.05_0.95' in metrics_df.columns:
                    metric2 = float(metrics_df['recall_0.05_0.95'].sum()) / 5.0
                else:
                    metric2 = float('nan')
                print(f"[Eval] Metric2 (mean recall@5-95 across sources): {metric2:.5f}")
            except Exception as e:
                print(f"[Eval] Relationship benchmark failed: {e}")
            print("========== EVALUATION END (inference) =========\n")
        except Exception as e:
            print(f"[Eval] Inference-time evaluation failed: {e}")
    
    # Save results to specified output path
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    embeddings_path = os.path.join(output_path, f"{prefix}_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")
    
    # Save cell metadata if available
    if adata.obs is not None and len(adata.obs.columns) > 0:
        metadata_path = os.path.join(output_path, f"{prefix}_metadata.csv")
        adata.obs.to_csv(metadata_path)
        print(f"Cell metadata saved to {metadata_path}")
    
    return embeddings


def batch_inference(
    adata_paths: list,
    model_dir: str,
    output_dir: str,
    gene_name_col: Optional[str] = None,
    batch_size: int = 512,
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
    eval_after_inference: bool = False,
    eval_frequency: int = 0,
    eval_seed: int = 42
) -> Dict[str, Union[np.ndarray, str]]:
    """
    Run inference on multiple h5ad files.
    
    Automatically uses chunked processing for files with >300k samples.
    
    Args:
        adata_paths: List of paths to h5ad files
        model_dir: Directory containing trained model
        output_dir: Directory to save outputs
        gene_name_col: Column name for gene names (None for auto-detection)
        batch_size: Batch size for inference
        device: Device to use
        checkpoint_path: Optional specific checkpoint to use
        preprocessor_path: Optional specific preprocessor to use
        eval_after_inference: Whether to run evaluation
        eval_frequency: Evaluation frequency
        eval_seed: Random seed for perturbation consistency null sampling
        
    Returns:
        Dictionary mapping file names to embeddings (array) or npy_path (chunked mode)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model once
    inference_pipeline = load_trained_model(model_dir, device, checkpoint_path, preprocessor_path)
    
    results = {}
    
    for adata_path in adata_paths:
        print(f"\nProcessing {adata_path}")
        
        # Get descriptive output prefix
        base_name = _make_output_prefix(adata_path, model_dir, checkpoint_path)
        output_path = os.path.join(output_dir, base_name)
        
        # Check file size using backed mode
        adata_backed = sc.read_h5ad(adata_path, backed='r')
        n_obs = adata_backed.n_obs
        
        # Use chunked mode for large files
        if n_obs > CHUNK_THRESHOLD:
            print(f"🔄 Using CHUNKED processing (n_obs={n_obs:,} > threshold={CHUNK_THRESHOLD:,})")
            npy_path, h5_path = inference_pipeline.predict_embeddings_chunked(
                adata_backed, output_path, gene_name_col, batch_size
            )
            # Save metadata
            obs_df = adata_backed.obs
            if obs_df is not None and len(obs_df.columns) > 0:
                metadata_path = f"{output_path}_metadata.csv"
                obs_df.to_csv(metadata_path)
                print(f"Cell metadata saved to {metadata_path}")
            adata_backed.file.close()
            results[base_name] = npy_path
            # Load embeddings for evaluation if needed
            if eval_after_inference:
                embeddings = np.load(npy_path)
                adata = sc.read_h5ad(adata_path)  # Load for metadata only
        else:
            print(f"📦 Using STANDARD processing (n_obs={n_obs:,} <= threshold={CHUNK_THRESHOLD:,})")
            adata_backed.file.close()
            
            # Load and process data
            adata = sc.read_h5ad(adata_path)
            
            # Ensure dense matrix for inference if sparse
            if hasattr(adata.X, 'toarray'):
                print("Converting sparse matrix to dense for inference...")
                adata.X = adata.X.toarray()
                
            embeddings = inference_pipeline.predict_and_save(
                adata, 
                output_path, 
                gene_name_col, 
                batch_size
            )
            
            results[base_name] = embeddings
    
        # Optional periodic evaluation on each file, based on frequency
        if eval_after_inference:
            run_eval = False
            if eval_frequency and eval_frequency > 0:
                if ((len(results)) % eval_frequency) == 0:
                    run_eval = True
            else:
                # if eval_frequency == 0, only evaluate after all files
                run_eval = False
            if run_eval:
                try:
                    print(f"\n========== EVALUATION (batch, periodic, seed={eval_seed}) ==========")
                    metadata = adata.obs.copy() if adata.obs is not None else pd.DataFrame(index=np.arange(embeddings.shape[0]))
                    pert_colname = "gene_name"
                    control_col = "is_control"
                    batch_col = "dataset_batch_num"
                    control_key = True
                    tvn_embeddings = tvn_on_controls(
                        embeddings=embeddings,
                        metadata=metadata,
                        pert_col=control_col,
                        control_key=control_key,
                        batch_col=batch_col,
                    )
                    from occam.evaluation.perturbation import pert_signal_consistency_benchmark
                    vc = metadata.get(pert_colname, pd.Series(index=metadata.index)).value_counts() if pert_colname in metadata.columns else pd.Series()
                    keys_to_drop = vc[vc == 1].index.tolist() if len(vc) > 0 else []
                    neg_ctrl_perts = (
                        metadata.loc[metadata[control_col] == control_key, pert_colname].unique().tolist()
                        if (pert_colname in metadata.columns and control_col in metadata.columns)
                        else []
                    )
                    np.random.seed(eval_seed)
                    cons_res = pert_signal_consistency_benchmark(
                        tvn_embeddings,
                        metadata,
                        pert_col=pert_colname if pert_colname in metadata.columns else (metadata.columns[0] if len(metadata.columns) > 0 else 'pert'),
                        neg_ctrl_perts=neg_ctrl_perts,
                        keys_to_drop=keys_to_drop,
                        stat="cossim",
                        num_null_cells=min(50000, tvn_embeddings.shape[0])
                    )
                    denom = np.sum(~pd.isna(cons_res["pval"])) if "pval" in cons_res.columns else 0
                    metric1 = float(np.sum(cons_res.pval <= 0.01)) / float(denom if denom > 0 else 1) * 100.0 if denom > 0 else float('nan')
                    print(f"[Eval] Metric1 (percent significant p<=0.01): {metric1:.3f}")
                    try:
                        all_controls = []
                        pert_key = pert_colname if pert_colname in metadata.columns else (metadata.columns[0] if len(metadata.columns) > 0 else 'pert')
                        map_data = aggregate(tvn_embeddings, metadata, pert_col=pert_key, keys_to_remove=all_controls)
                        env_bmdb = os.environ.get('OCCAM_BMDB_DIR') or os.environ.get('OCCAM_BMDB_DIR')
                        candidates = [env_bmdb, os.path.join(os.getcwd(), 'bmdb_database'), os.path.join(os.path.dirname(__file__), 'bmdb_database')]
                        benchmark_dir = None
                        for c in candidates:
                            if c and os.path.isdir(c):
                                benchmark_dir = c
                                break
                        if benchmark_dir is None:
                            raise FileNotFoundError("bmdb_database directory not found. Set OCCAM_BMDB_DIR (or OCCAM_BMDB_DIR) or ensure ./bmdb_database exists.")
                        sources = ["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"]
                        metrics_df = known_relationship_benchmark(
                            map_data,
                            pert_col=pert_key,
                            benchmark_sources=sources,
                            recall_thr_pairs=DEFAULT_RECALL_THR_PAIRS,
                            benchmark_data_dir=benchmark_dir,
                            min_req_entity_cnt=20,
                            log_stats=True,
                        )
                        if metrics_df is not None and not metrics_df.empty and 'recall_0.05_0.95' in metrics_df.columns:
                            metric2 = float(metrics_df['recall_0.05_0.95'].sum()) / 5.0
                        else:
                            metric2 = float('nan')
                        print(f"[Eval] Metric2 (mean recall@5-95 across sources): {metric2:.5f}")
                    except Exception as e:
                        print(f"[Eval] Relationship benchmark failed: {e}")
                    print("========== EVALUATION END (batch, periodic) =========\n")
                except Exception as e:
                    print(f"[Eval] Batch periodic evaluation failed: {e}")

    # Final evaluation after all files if requested and not already covered by frequency
    if eval_after_inference and (not eval_frequency or eval_frequency <= 0):
        try:
            print("\n========== EVALUATION (batch, final) ==========")
            # Concatenate embeddings and metadata across files for a global evaluation
            # Note: To keep memory reasonable, we evaluate per-file above; here we just notify completion
            print("[Eval] Final evaluation: metrics were computed per-file or skipped due to frequency=0.")
            print("========== EVALUATION END (batch, final) =========\n")
        except Exception as e:
            print(f"[Eval] Final batch evaluation failed: {e}")

    print(f"\nBatch inference completed! Results saved to {output_dir}")
    return results


def compare_embeddings(embeddings1: np.ndarray, 
                      embeddings2: np.ndarray,
                      labels1: Optional[np.ndarray] = None,
                      labels2: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compare two sets of embeddings using various metrics.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        labels1: Optional labels for first set
        labels2: Optional labels for second set
        
    Returns:
        Dictionary of comparison metrics
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr
    
    metrics = {}
    
    # Cosine similarity between embedding sets
    if embeddings1.shape == embeddings2.shape:
        cosine_sims = np.diag(cosine_similarity(embeddings1, embeddings2))
        metrics['mean_cosine_similarity'] = np.mean(cosine_sims)
        metrics['std_cosine_similarity'] = np.std(cosine_sims)
    
    # Compare embedding distributions
    flat1 = embeddings1.flatten()
    flat2 = embeddings2.flatten()
    
    if len(flat1) == len(flat2):
        correlation, p_value = pearsonr(flat1, flat2)
        metrics['embedding_correlation'] = correlation
        metrics['correlation_p_value'] = p_value
    
    # Compare pairwise distances
    dist1 = pdist(embeddings1[:min(1000, len(embeddings1))])  # Sample for efficiency
    dist2 = pdist(embeddings2[:min(1000, len(embeddings2))])
    
    if len(dist1) == len(dist2):
        dist_corr, _ = pearsonr(dist1, dist2)
        metrics['distance_structure_correlation'] = dist_corr
    
    return metrics


# Backward compatibility alias
OccamInference = OccamEmbedder