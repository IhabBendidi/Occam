"""Pure PyTorch linear probing evaluation (no pytorch-lightning).

Rewritten from biomodal_codebase/biomodalities/eval/linear_probing.py.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .lr_scheduler import LinearWarmupCosineAnnealingLR
from .metrics import accuracy_at_k


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_linear_probing(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    unique_labels: list,
    num_epochs: int = 250,
    lr: float = 0.001,
    weight_decay: float = 1e-6,
    batch_size: int = 2048,
    warmup_epochs: int = 10,
    seed: int = 42,
    device: torch.device = None,
) -> dict:
    """Train a linear classifier on embeddings and evaluate on test set.

    Args:
        train_embeddings: (n_train, dim) array of training embeddings
        train_labels: (n_train,) array of string labels
        test_embeddings: (n_test, dim) array of test embeddings
        test_labels: (n_test,) array of string labels
        unique_labels: list of all unique label strings
        num_epochs: number of training epochs
        lr: learning rate
        weight_decay: weight decay
        batch_size: training batch size
        warmup_epochs: warmup epochs for cosine scheduler
        seed: random seed
        device: torch device

    Returns:
        dict with 'test_acc1' and 'test_acc5' (percentages)
    """
    _set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    input_dim = train_embeddings.shape[1]

    if isinstance(train_embeddings, torch.Tensor):
        train_x = train_embeddings.clone().detach().float()
    else:
        train_x = torch.tensor(train_embeddings, dtype=torch.float32)
    train_y = torch.tensor([label_to_idx[l] for l in train_labels], dtype=torch.long)
    if isinstance(test_embeddings, torch.Tensor):
        test_x = test_embeddings.clone().detach().float()
    else:
        test_x = torch.tensor(test_embeddings, dtype=torch.float32)
    test_y = torch.tensor([label_to_idx[l] for l in test_labels], dtype=torch.long)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    classifier = nn.Linear(input_dim, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)

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
        classifier.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = classifier(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Evaluation
    classifier.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = classifier(batch_x)
            all_logits.append(logits.cpu())
            all_targets.append(batch_y)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    acc1, acc5 = accuracy_at_k(all_logits, all_targets, top_k=(1, 5))

    return {
        "test_acc1": acc1.item(),
        "test_acc5": acc5.item(),
    }
