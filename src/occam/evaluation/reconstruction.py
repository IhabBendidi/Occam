"""Pure PyTorch reconstruction (decoder) evaluation (no pytorch-lightning).

Rewritten from biomodal_codebase/biomodalities/eval/reconstruction.py.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .lr_scheduler import LinearWarmupCosineAnnealingLR
from .metrics import evaluate_reconstruction_metrics, StructuralTranscriptomeDistance


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_decoder(embedding_dim: int, output_dim: int, model_depth: int = 3):
    """Build MLP decoder with linearly interpolated hidden dims."""
    hidden_dims = [
        int(embedding_dim + (output_dim - embedding_dim) * i / (model_depth + 1))
        for i in range(1, model_depth + 1)
    ]
    layers = []
    input_dim = embedding_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def run_reconstruction(
    train_embeddings: np.ndarray,
    train_expressions: np.ndarray,
    test_embeddings: np.ndarray,
    test_expressions: np.ndarray,
    batch_ids_test: np.ndarray,
    control_labels_test: np.ndarray,
    control_label=True,
    num_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 0.0001,
    weight_decay: float = 1e-6,
    model_depth: int = 3,
    warmup_epochs: int = 10,
    seed: int = 42,
    device: torch.device = None,
) -> dict:
    """Train a decoder from embeddings to gene expression and evaluate.

    Args:
        train_embeddings: (n_train, emb_dim) training embeddings
        train_expressions: (n_train, n_genes) training gene expression targets
        test_embeddings: (n_test, emb_dim) test embeddings
        test_expressions: (n_test, n_genes) test gene expression targets
        batch_ids_test: (n_test,) batch IDs for test samples
        control_labels_test: (n_test,) control labels for test samples
        control_label: value indicating control samples
        num_epochs: training epochs
        batch_size: training batch size
        lr: learning rate
        weight_decay: weight decay
        model_depth: number of hidden layers
        warmup_epochs: warmup epochs for scheduler
        seed: random seed
        device: torch device

    Returns:
        dict with 'Average Spearman Correlation' and 'structural_distance'
    """
    _set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dim = train_embeddings.shape[1]
    output_dim = train_expressions.shape[1]

    def _to_float_tensor(arr):
        if isinstance(arr, torch.Tensor):
            return arr.clone().detach().float()
        return torch.tensor(arr, dtype=torch.float32)

    train_x = _to_float_tensor(train_embeddings)
    train_y = _to_float_tensor(train_expressions)
    test_x = _to_float_tensor(test_embeddings)
    test_y = _to_float_tensor(test_expressions)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    decoder = _build_decoder(embedding_dim, output_dim, model_depth).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler steps per batch (scheduler_interval='step'), matching biomodal
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=warmup_steps, max_epochs=total_steps,
        warmup_start_lr=3e-5, eta_min=0.0,
    )

    # Training loop (scheduler stepped per batch, matching biomodal)
    for epoch in range(num_epochs):
        decoder.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = decoder(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Evaluation on test set
    decoder.eval()
    all_preds = []
    test_loader = DataLoader(TensorDataset(test_x), batch_size=batch_size * 2, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            pred = decoder(batch_x)
            all_preds.append(pred.cpu())

    all_preds = torch.cat(all_preds, dim=0)

    # Subsample to match biomodal's batch_idx%100 sampling for metrics
    batch_size_eval = batch_size * 2  # test loader uses batch_size*2
    sampled_indices = []
    total_samples = len(all_preds)
    num_batches = (total_samples + batch_size_eval - 1) // batch_size_eval
    for bidx in range(num_batches):
        if bidx % 100 == 0:
            start = bidx * batch_size_eval
            end = min(start + batch_size_eval, total_samples)
            sampled_indices.extend(range(start, end))
    sampled_preds = all_preds[sampled_indices]
    sampled_targets = test_y[sampled_indices]
    sampled_batch_ids = batch_ids_test[sampled_indices]
    sampled_control_labels = control_labels_test[sampled_indices]

    # Compute reconstruction metrics on sampled data
    recon_metrics = evaluate_reconstruction_metrics(sampled_preds, sampled_targets)

    # Compute structural transcriptome distance on sampled data
    structural_distance_metric = StructuralTranscriptomeDistance()
    structural_distance_metric.update(
        sampled_preds, sampled_targets, sampled_batch_ids, sampled_control_labels, control_label=control_label
    )
    structural_dist = structural_distance_metric.compute()
    structural_distance_metric.reset()

    return {
        "Average Spearman Correlation": recon_metrics.get("Average Spearman Correlation", float('nan')),
        "structural_distance": structural_dist,
    }
