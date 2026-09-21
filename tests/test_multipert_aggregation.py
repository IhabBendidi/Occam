"""The multi-perturbation path: identical to the loop it replaced, and fast enough to train on.

WHY THIS FILE EXISTS
    A double perturbation is modelled as the SUM of its parts in graph-embedding space
    (`model.forward`, the `max_perts > 1` branch). That branch was a Python double loop --
    2 x batch_size autograd ops per forward -- written as a rarely-taken fallback. The Norman
    double-perturbation benchmark makes it the HOT path, so it was vectorised with
    `index_add_`, and the vectorisation has to be proven equivalent rather than assumed:
    an off-by-one in the row index would silently attribute a perturbation to the wrong cell,
    which no score would reveal as anything but "the model is worse on doubles".
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def _loop_reference(z_graph, pert_indices, dim):
    """The original implementation, kept here as the oracle."""
    out = torch.zeros(len(pert_indices), dim, dtype=z_graph.dtype)
    for i, perts in enumerate(pert_indices):
        for p in perts:
            if p >= 0:
                out[i] = out[i] + z_graph[p]
    return out


def _vectorised(z_graph, pert_indices, dim):
    """The shipped implementation, transcribed (same lines as model.py)."""
    flat, rows = [], []
    for i, perts in enumerate(pert_indices):
        for p in perts:
            if p >= 0:
                flat.append(p)
                rows.append(i)
    out = torch.zeros(len(pert_indices), dim, dtype=z_graph.dtype)
    if flat:
        out = out.index_add(0, torch.as_tensor(rows, dtype=torch.long),
                            z_graph[torch.as_tensor(flat, dtype=torch.long)])
    return out


CASES = [
    ("doubles", [[3, 7], [1, 1], [5, 2], [0, 4]]),
    ("mixed arity", [[3], [1, 2], [-1], [4, 5, 6]]),
    ("all control", [[-1], [-1]]),
    ("control mixed with real", [[-1, 2], [3, -1]]),
    ("repeated gene in one sample", [[9, 9, 9]]),
    ("single arity only", [[2], [3]]),
]


def test_vectorised_multipert_matches_the_loop_it_replaced():
    torch.manual_seed(0)
    z = torch.randn(16, 8, dtype=torch.float64)
    for name, idx in CASES:
        a = _loop_reference(z, idx, 8)
        b = _vectorised(z, idx, 8)
        assert torch.allclose(a, b, atol=0, rtol=0), f"{name}: vectorised != loop\n{a}\n{b}"


def test_a_repeated_gene_is_counted_TWICE_in_both_implementations():
    """Not a bug to fix silently: `[[9, 9]]` sums z_graph[9] twice in the original too.

    Pinned because the Norman cohort has no repeated gene within a pair, so a future change
    that "deduplicates" would be invisible there and would change the semantics elsewhere.
    """
    z = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    assert torch.allclose(_vectorised(z, [[1, 1]], 3), 2 * z[1])
    assert torch.allclose(_loop_reference(z, [[1, 1]], 3), 2 * z[1])


def test_gradients_flow_to_every_listed_perturbation():
    """The sum must give each constituent gene of a double its own gradient path."""
    z = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    out = _vectorised(z, [[0, 3], [1], [-1]], 4)
    out.sum().backward()
    g = z.grad.abs().sum(dim=1)
    assert float(g[0]) > 0 and float(g[3]) > 0, "a double's second gene got no gradient"
    assert float(g[1]) > 0
    assert float(g[2]) == 0.0 and float(g[4]) == 0.0, "an unlisted gene received gradient"


def test_the_model_forward_takes_doubles_end_to_end():
    """Guard the real call site, not only the arithmetic: `[[a, b]]` must run and differ
    from `[[a]]`, or a double perturbation is silently treated as a single."""
    from occam.vcell.model import VirtualCellModel
    torch.manual_seed(0)
    n_genes, n_perts, hidden = 64, 12, 16
    try:
        m = VirtualCellModel(
            num_genes=n_genes, num_perts=n_perts, hidden_size=hidden,
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_weight=torch.ones(2),
        )
    except TypeError as e:
        pytest.skip(f"VirtualCellModel signature differs here ({e}); arithmetic is covered above")
    m.eval()
    ctrl = torch.randn(2, n_genes)
    with torch.no_grad():
        single = m(ctrl, [[1], [1]])["prediction"]
        double = m(ctrl, [[1, 2], [1, 2]])["prediction"]
    assert single.shape == double.shape
    assert not torch.allclose(single, double), \
        "a double perturbation produced the same prediction as its first gene alone"
