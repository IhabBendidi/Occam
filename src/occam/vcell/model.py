"""Virtual Cell Model — combines Occam encoder, perturbation GNN, and decoder.

Architecture (v1 — additive):
    control_expr → Occam Encoder → z_ctrl (hidden_size)
    pert_gene    → GNN(graph)      → e_pert (pert_latent_dim)
    delta_z      = Linear(e_pert)
    z_hat        = z_ctrl + delta_z
    prediction   = Decoder(z_hat)

Architecture (v2 — interaction functions):
    delta_z      = f(z_ctrl, e_pert)   # FiLM, Gated MLP, or Cell-Aware FiLM
    z_hat        = z_ctrl + delta_z
    prediction   = Decoder(z_hat)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from occam.vcell.gnn import PerturbationGNN


# ---------------------------------------------------------------------------
# Activation factory (shared with adjacency_map.py)
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "gelu": nn.GELU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU, "mish": nn.Mish, "softplus": nn.Softplus,
    "tanh": nn.Tanh, "elu": nn.ELU, "selu": nn.SELU, "none": nn.Identity,
}


def _make_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name!r}. Expected one of: {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# Interaction modules: f(z_ctrl, e_pert) -> delta_z
# ---------------------------------------------------------------------------


class AdditiveInteraction(nn.Module):
    """v1-compatible additive interaction: delta_z = Linear(e_pert).

    z_ctrl is ignored — the perturbation effect is a fixed vector
    independent of cell state. Equivalent to the original pert_projection.
    """

    def __init__(self, pert_latent_dim: int, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(pert_latent_dim, hidden_size)

    def forward(self, z_ctrl: torch.Tensor, e_pert: torch.Tensor) -> torch.Tensor:
        return self.projection(e_pert)


class FiLMInteraction(nn.Module):
    """Feature-wise Linear Modulation: gamma/beta from e_pert only.

    gamma = 1 + scale * (sigmoid(gamma_net(e_pert)) - 0.5)
    beta  = beta_net(e_pert)
    delta_z = (gamma - 1) * z_ctrl + beta

    When ``film_hidden_dim > 0``, gamma_net and beta_net are 2-layer MLPs
    (Linear → activation → Linear) instead of single linear projections.
    ``film_hidden_dim = 0`` (default) reproduces the original linear FiLM.
    """

    def __init__(
        self,
        pert_latent_dim: int,
        hidden_size: int,
        scale: float = 2.0,
        film_hidden_dim: int = 0,
        film_activation: str = "none",
        film_mode: str = "full",
    ):
        super().__init__()
        self.scale = scale
        self.film_mode = film_mode

        if film_hidden_dim > 0:
            act = _make_activation(film_activation)
            self.gamma_net = nn.Sequential(
                nn.Linear(pert_latent_dim, film_hidden_dim),
                act,
                nn.Linear(film_hidden_dim, hidden_size),
            )
            self.beta_net = nn.Sequential(
                nn.Linear(pert_latent_dim, film_hidden_dim),
                _make_activation(film_activation),
                nn.Linear(film_hidden_dim, hidden_size),
            )
        else:
            self.gamma_net = nn.Linear(pert_latent_dim, hidden_size)
            self.beta_net = nn.Linear(pert_latent_dim, hidden_size)

    def forward(self, z_ctrl: torch.Tensor, e_pert: torch.Tensor) -> torch.Tensor:
        if self.film_mode == "additive_only":
            return self.beta_net(e_pert)
        elif self.film_mode == "multiplicative_only":
            gamma = 1.0 + self.scale * (torch.sigmoid(self.gamma_net(e_pert)) - 0.5)
            return (gamma - 1.0) * z_ctrl
        else:
            gamma = 1.0 + self.scale * (torch.sigmoid(self.gamma_net(e_pert)) - 0.5)
            beta = self.beta_net(e_pert)
            return (gamma - 1.0) * z_ctrl + beta


class GatedMLPInteraction(nn.Module):
    """Gated MLP: both gate and shift depend on z_ctrl and e_pert.

    h = MLP([z_ctrl; e_pert])
    gate = sigmoid(W_g @ h)      # which dims to modify [0, 1]
    shift = W_s @ h              # proposed modification
    delta_z = gate * shift

    The gate is interpretable: near-1 = dimension being modified,
    near-0 = dimension preserved. More parameters than FiLM.

    Stores ``_last_gate`` (detached) for diagnostic logging.
    """

    def __init__(
        self,
        pert_latent_dim: int,
        hidden_size: int,
        mlp_hidden_dim: Optional[int] = None,
        num_mlp_layers: int = 2,
    ):
        super().__init__()
        mlp_hidden_dim = mlp_hidden_dim or hidden_size
        input_dim = hidden_size + pert_latent_dim

        # Build MLP: input_dim -> mlp_hidden_dim (x num_mlp_layers) with GELU
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_mlp_layers):
            layers.extend([nn.Linear(in_dim, mlp_hidden_dim), nn.GELU()])
            in_dim = mlp_hidden_dim
        self.mlp = nn.Sequential(*layers)

        self.W_g = nn.Linear(mlp_hidden_dim, hidden_size)
        self.W_s = nn.Linear(mlp_hidden_dim, hidden_size)

        # Diagnostic: last gate activation (detached, not part of computation graph)
        self._last_gate: Optional[torch.Tensor] = None

    def forward(self, z_ctrl: torch.Tensor, e_pert: torch.Tensor) -> torch.Tensor:
        h = self.mlp(torch.cat([z_ctrl, e_pert], dim=-1))
        gate = torch.sigmoid(self.W_g(h))
        shift = self.W_s(h)
        self._last_gate = gate.detach()
        return gate * shift


class CellAwareFiLMInteraction(nn.Module):
    """Cell-state-aware FiLM: gamma/beta see both z_ctrl and e_pert.

    gamma = 1 + scale * (sigmoid(W_gamma @ [z_ctrl; e_pert]) - 0.5)
    beta  = W_beta @ [z_ctrl; e_pert]
    delta_z = (gamma - 1) * z_ctrl + beta

    Strictly more expressive than FiLM, comparable to Gated MLP
    but retains explicit multiplicative structure.
    """

    def __init__(self, pert_latent_dim: int, hidden_size: int, scale: float = 2.0):
        super().__init__()
        input_dim = hidden_size + pert_latent_dim
        self.W_gamma = nn.Linear(input_dim, hidden_size)
        self.W_beta = nn.Linear(input_dim, hidden_size)
        self.scale = scale

    def forward(self, z_ctrl: torch.Tensor, e_pert: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z_ctrl, e_pert], dim=-1)
        gamma = 1.0 + self.scale * (torch.sigmoid(self.W_gamma(combined)) - 0.5)
        beta = self.W_beta(combined)
        return (gamma - 1.0) * z_ctrl + beta


def build_interaction(
    interaction_type: str,
    pert_latent_dim: int,
    hidden_size: int,
    **kwargs,
) -> nn.Module:
    """Factory: build an interaction module by name.

    Args:
        interaction_type: "additive", "film", "gated_mlp", or "cell_aware_film".
        pert_latent_dim: GNN output dimension per gene.
        hidden_size: Encoder output / decoder input dimension.
        **kwargs: Passed to the interaction constructor (e.g. scale, mlp_hidden_dim).

    Returns:
        nn.Module with forward(z_ctrl, e_pert) -> delta_z.
    """
    if interaction_type == "additive":
        return AdditiveInteraction(pert_latent_dim, hidden_size)
    elif interaction_type == "film":
        return FiLMInteraction(
            pert_latent_dim, hidden_size,
            scale=kwargs.get("film_scale", 2.0),
            film_hidden_dim=kwargs.get("film_hidden_dim", 0),
            film_activation=kwargs.get("film_activation", "none"),
            film_mode=kwargs.get("film_mode", "full"),
        )
    elif interaction_type == "gated_mlp":
        return GatedMLPInteraction(
            pert_latent_dim,
            hidden_size,
            mlp_hidden_dim=kwargs.get("gated_mlp_hidden_dim"),
            num_mlp_layers=kwargs.get("gated_mlp_num_layers", 2),
        )
    elif interaction_type == "cell_aware_film":
        return CellAwareFiLMInteraction(
            pert_latent_dim, hidden_size, scale=kwargs.get("film_scale", 2.0)
        )
    else:
        raise ValueError(
            f"Unknown interaction_type: {interaction_type!r}. "
            f"Expected one of: additive, film, gated_mlp, cell_aware_film"
        )


# ---------------------------------------------------------------------------
# VirtualCellModel
# ---------------------------------------------------------------------------


class VirtualCellModel(nn.Module):
    """Virtual cell perturbation effect prediction model.

    Forward pass returns a dict with keys:
        - "prediction": (B, num_genes) predicted perturbed expression
        - "delta_z": (B, hidden_size) predicted latent perturbation effect
        - "z_hat": (B, hidden_size) perturbed latent = z_ctrl + delta_z
        - "z_ctrl": (B, hidden_size) control latent embedding

    Args:
        encoder: Occam encoder (nn.Sequential).
        gnn: PerturbationGNN for gene embeddings.
        decoder: HybridNormGlobalSkipDecoder instance.
        edge_index: (2, E) graph connectivity tensor.
        edge_weight: (E,) edge weight tensor.
        encoder_hidden_size: Encoder output dimension.
        pert_latent_dim: GNN output dimension per gene.
        freeze_encoder: Whether to freeze encoder parameters.
        freeze_decoder: Whether to freeze decoder parameters.
        interaction_type: Interaction function type (default "additive" for v1 compat).
        interaction_kwargs: Extra kwargs for the interaction constructor.
        film_mean_free: Subtract the interaction evaluated at the MEAN graph embedding, so a
            component shared across perturbations is unrepresentable. Perturbed rows only.
    """

    def __init__(
        self,
        encoder: nn.Module,
        gnn: nn.Module,
        decoder: nn.Module,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        encoder_hidden_size: int,
        pert_latent_dim: int,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        interaction_type: str = "additive",
        interaction_kwargs: Optional[Dict] = None,
        source_cond: str = "none",
        n_sources: int = 0,
        source_dropout: float = 0.25,
        film_mean_free: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.gnn = gnn
        self.decoder = decoder
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.encoder_hidden_size = encoder_hidden_size
        self.pert_latent_dim = pert_latent_dim
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.interaction_type = interaction_type

        # ⭐ Mean-free interaction (default off). Emit
        #       interaction(z, e_p) - interaction(z, ebar)
        # so a component SHARED across perturbations is unrepresentable by construction rather
        # than penalised after the fact. In the research tree the shared component was measured
        # at cos(mean delta_z, FiLM(ebar)) = 1.0000/0.9998, i.e. this subtracts almost exactly
        # the object that made up ~99.8% of every predicted delta.
        #
        # `ebar` is the mean over the WHOLE graph node table, never a batch statistic, so
        # inference does not depend on batch composition.
        #
        # ⚠️ Applied only to rows that carry a perturbation, so control rows stay bit-identical
        #    to the base model and the control-identity loss is unaffected.
        #
        # Note it also removes the interaction's own bias contribution (`beta_net(0)`), which is
        # why it is not equivalent to centering the embedding table.
        self.film_mean_free = bool(film_mean_free)

        # Interaction module (replaces the old pert_projection Linear)
        self.interaction = build_interaction(
            interaction_type,
            pert_latent_dim,
            encoder_hidden_size,
            **(interaction_kwargs or {}),
        )

        # ---- F181/E3: source + modality conditioning of the perturbation code -------------
        # all9 mixes incompatible ESTIMANDS, not just distributions. `joung_2023` is a TF
        # OVEREXPRESSION atlas (median relative self-delta +0.85) inside an otherwise
        # all-knockdown corpus (-0.39..-0.90), and 99.1% of its perturbations also carry a
        # knockdown label elsewhere -- so 62 of the 255 scored perturbations get two opposite
        # direction labels through this one shared encoder. Sources also span 5.9x in effect
        # norm (0.99 HEK293T .. 5.84 joung) because knockdown penetrance differs.
        #
        # With no modality input the Bayes-optimal unconditioned prediction is the mean of Δ
        # over sources, which is shrunk and direction-scrambled -> pds ~ 0 (F179).
        # Conditioning turns "incompatible estimands" into "one estimand + nuisance covariate".
        #
        # Injected ADDITIVELY on the perturbation latent, so the interaction sees a
        # modality-aware perturbation code and can learn a per-source offset/scale.
        # Index `n_sources` is a NEUTRAL slot used at inference (H1 has no training source);
        # `source_dropout` routes a fraction of training samples through it so the neutral
        # code is a trained marginal rather than an out-of-distribution vector.
        # Modality: 0 = knockdown, 1 = activation, 2 = neutral. H1 inference uses 0.
        self.source_cond = str(source_cond or "none")
        self.n_sources = int(n_sources)
        self.source_dropout = float(source_dropout)
        if self.source_cond != "none" and self.n_sources > 0:
            self.source_emb = nn.Embedding(self.n_sources + 1, pert_latent_dim)
            self.modality_emb = nn.Embedding(3, pert_latent_dim)
            nn.init.zeros_(self.source_emb.weight)
            nn.init.zeros_(self.modality_emb.weight)
        else:
            self.source_emb = None
            self.modality_emb = None

        # GNN output cache (invalidated after optimizer step during training)
        self._gnn_cache: Optional[torch.Tensor] = None

        # Pre-computed control encoder embeddings (set via precompute_control_embeddings)
        self._control_embeddings: Optional[Dict[str, torch.Tensor]] = None

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False
            self.decoder.eval()

    def invalidate_gnn_cache(self):
        """Invalidate GNN cache — call after optimizer step during training."""
        self._gnn_cache = None

    def precompute_control_embeddings(
        self, control_pool: Dict[str, torch.Tensor], device: str = "cuda",
        batch_size: int = 2048,
    ):
        """Pre-compute encoder embeddings for all controls.

        After calling this, forward() will skip the encoder and look up
        pre-computed embeddings from control_indices in the batch.

        Args:
            control_pool: {cell_type: (N_ct, num_genes)} raw expression tensors.
            device: Device to run encoder on.
            batch_size: Batch size for encoder forward passes.
        """
        self._control_embeddings = {}
        self.encoder.eval()
        with torch.no_grad():
            for ct, pool in control_pool.items():
                embeddings = []
                for i in range(0, pool.shape[0], batch_size):
                    batch = pool[i : i + batch_size].to(device)
                    embeddings.append(self.encoder(batch).cpu())
                self._control_embeddings[ct] = torch.cat(embeddings)
                print(
                    f"  Pre-computed {self._control_embeddings[ct].shape[0]} "
                    f"encoder embeddings for {ct}"
                )

    def forward(
        self,
        control_expression: torch.Tensor,
        pert_indices: List[List[int]],
        source_ids: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict perturbed gene expression.

        Args:
            control_expression: (B, num_encoder_genes) in log1p space,
                or (B, hidden_size) if pre-computed embeddings.
            pert_indices: Per-sample list of pert2id gene indices.
                          -1 = control token (no perturbation).

        Returns:
            Dict with keys: "prediction", "delta_z", "z_hat", "z_ctrl".
        """
        batch_size = control_expression.shape[0]
        device = control_expression.device

        # 1. Encode control state (skip if pre-computed embeddings)
        if control_expression.shape[1] == self.encoder_hidden_size:
            # Already encoded (pre-computed embeddings from control pool)
            z_intrinsic = control_expression
        elif self.freeze_encoder:
            with torch.no_grad():
                z_intrinsic = self.encoder(control_expression)
        else:
            z_intrinsic = self.encoder(control_expression)

        # 2. GNN: compute embeddings (cached when possible)
        if self._gnn_cache is not None:
            z_graph = self._gnn_cache
        else:
            z_graph = self.gnn(self.edge_index, self.edge_weight)
            if not self.training:
                self._gnn_cache = z_graph

        # 3. Vectorized perturbation aggregation
        max_perts = max(len(p) for p in pert_indices)
        if max_perts == 1:
            # Single-pert fast path — simple batched gather
            pert_flat = torch.tensor(
                [p[0] if p[0] >= 0 else 0 for p in pert_indices],
                device=device,
                dtype=torch.long,
            )
            pert_mask = torch.tensor(
                [float(p[0] >= 0) for p in pert_indices],
                device=device,
            ).unsqueeze(-1)
            pert_z = z_graph[pert_flat] * pert_mask  # (B, pert_latent_dim)
        else:
            # ─── MULTI-PERT: sum the graph embeddings of a sample's perturbations ─────────
            # This is the COMPOSITION OPERATOR for combinatorial perturbations (e.g. Norman's
            # 131 double-perturbation pairs): a double is modelled as the sum of its parts in
            # graph-embedding space. It is the architectural assumption such a benchmark
            # exists to test, so it is worth naming rather than leaving as a "fallback".
            #
            # ⚠️ VECTORISED. The original implementation was a Python double loop building
            # `pert_z[i] = pert_z[i] + z_graph[p]` one autograd node at a time -- 2 x
            # batch_size graph ops per forward at batch 2048, written as a rarely-taken
            # fallback and never for a training run. `index_add_` does it in one kernel and is
            # mathematically identical (a sum is a sum, and index_add_ is deterministic for
            # a fixed index order on a single device).
            flat, rows = [], []
            for i, perts in enumerate(pert_indices):
                for p in perts:
                    if p >= 0:
                        flat.append(p); rows.append(i)
            pert_mask = torch.tensor(
                [float(any(p >= 0 for p in perts)) for perts in pert_indices],
                device=device,
            ).unsqueeze(-1)
            pert_z = torch.zeros(batch_size, self.pert_latent_dim, device=device,
                                 dtype=z_graph.dtype)
            if flat:
                idx_row = torch.as_tensor(rows, device=device, dtype=torch.long)
                idx_pert = torch.as_tensor(flat, device=device, dtype=torch.long)
                pert_z = pert_z.index_add(0, idx_row, z_graph[idx_pert])

        # 3b. F181/E3 source + modality conditioning (no-op unless --source_cond is set)
        pert_z = self._condition_pert(pert_z, pert_mask, source_ids, modality_ids, device)

        # 4. Interaction function: f(z_ctrl, e_pert) -> delta_z
        delta_z = self._interact(z_intrinsic, pert_z, z_graph, pert_mask)
        z_hat = z_intrinsic + delta_z
        decoder_out = self.decoder(z_hat)

        if isinstance(decoder_out, dict):
            prediction = decoder_out["prediction"]
            result = {"prediction": prediction, "delta_z": delta_z, "z_hat": z_hat, "z_ctrl": z_intrinsic}
            result.update({k: v for k, v in decoder_out.items() if k != "prediction"})
            return result

        return {
            "prediction": decoder_out,
            "delta_z": delta_z,
            "z_hat": z_hat,
            "z_ctrl": z_intrinsic,
        }

    def _condition_pert(self, pert_z, pert_mask, source_ids, modality_ids, device):
        """Add source/modality offsets to the perturbation code (F181/E3).

        `pert_mask` is (B,1) with 0 for control rows; the conditioning is masked by it so a
        control/identity row keeps an exactly-zero perturbation code and stays "unperturbed".
        `source_ids=None` selects the NEUTRAL slot, which is what H1 inference uses.
        """
        if self.source_emb is None:
            return pert_z
        B = pert_z.shape[0]
        if source_ids is None:
            sid = torch.full((B,), self.n_sources, device=device, dtype=torch.long)
        else:
            sid = source_ids.to(device=device, dtype=torch.long).clamp(0, self.n_sources)
            if self.training and self.source_dropout > 0:
                drop = torch.rand(B, device=device) < self.source_dropout
                sid = torch.where(drop, torch.full_like(sid, self.n_sources), sid)
        mid = (torch.zeros(B, device=device, dtype=torch.long) if modality_ids is None
               else modality_ids.to(device=device, dtype=torch.long).clamp(0, 2))
        cond = self.source_emb(sid) + self.modality_emb(mid)
        if pert_mask is not None:
            cond = cond * pert_mask
        return pert_z + cond

    def _encode_ctrl(self, control_expression: torch.Tensor) -> torch.Tensor:
        if control_expression.shape[1] == self.encoder_hidden_size:
            return control_expression
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(control_expression)
        return self.encoder(control_expression)

    def _pert_embeddings(self) -> torch.Tensor:
        if self._gnn_cache is not None:
            return self._gnn_cache
        z_graph = self.gnn(self.edge_index, self.edge_weight)
        if not self.training:
            self._gnn_cache = z_graph
        return z_graph

    def _slot_embed(self, z_graph, slot, device):
        """(embedding, mask) for one perturbation slot; index <0 -> zero vector, mask 0.

        The mask is returned as well as applied because `_interact` needs it to leave
        control rows untouched by the mean-free correction.
        """
        ids = torch.tensor([p if p >= 0 else 0 for p in slot], device=device, dtype=torch.long)
        msk = torch.tensor([float(p >= 0) for p in slot], device=device).unsqueeze(-1)
        return z_graph[ids] * msk, msk

    def _gather_pert(self, z_graph, slot, device):
        """Embedding for one perturbation slot; index <0 -> zero vector."""
        return self._slot_embed(z_graph, slot, device)[0]

    def _interact(self, z, e, z_graph, mask):
        """delta_z for one interaction, with the mean-free correction when enabled.

        THE ONLY place `self.interaction` is called for a perturbation effect. Both the
        batched `forward` and the sequential multi-perturbation path route through here, so
        enabling `film_mean_free` cannot apply to one and silently skip the other.

        `mask` zeroes the correction on rows with no perturbation, keeping control rows
        bit-identical to the base model.
        """
        d = self.interaction(z, e)
        if not self.film_mean_free:
            return d
        ebar = z_graph.mean(0, keepdim=True)
        # ⚠️ DTYPE, and it is not cosmetic. `_gnn_cache` can be populated under bf16 autocast
        #    and then reused OUTSIDE it (the eval path), so `ebar` may be bf16 while the
        #    interaction's weights are fp32. `e` escapes this only incidentally — the fp32 mask
        #    multiply in `_slot_embed` upcasts it — so match `e` explicitly rather than rely on
        #    that accident. Two research-tree jobs died here AFTER a full epoch plus eval.
        ref = self.interaction(z, ebar.expand(z.shape[0], -1).to(e.dtype))
        return d - ref * mask

    def forward_sequential(
        self,
        control_expression: torch.Tensor,
        pert_indices: List[List[int]],
        average_orders: bool = True,
        order: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply a sample's perturbations one at a time, chained in LATENT space.

            control + pert_A -> cell_A ;  cell_A + pert_B -> cell_AB

        The whole chain runs through the interaction module on latents; the decoder
        is invoked EXACTLY ONCE, on the final state. Calling ``forward()`` twice
        would decode the intermediate too, which nothing consumes -- the composition
        never needs cell_A in gene space.

        ``order`` selects which chain to return and SUPERSEDES ``average_orders``:

            "ab"   the order as given in ``pert_indices``
            "ba"   the reversed order
            "mean" the average of the two latent states

        ⚠️ THE TWO ORDERS ARE NOT THE SAME PREDICTION, but the biology has no order: a
        double perturbation is a set. "mean" is therefore the defensible default, and the
        AB-vs-BA disagreement is worth reporting in its own right -- it measures how much
        of a sequential composition's answer is an artifact of the order it was given.
        ``average_orders`` is kept for backward compatibility and maps to "mean"/"ab".

        Samples with 0 or 1 perturbations take the single-step path, so a mixed
        batch of controls / singles / doubles is handled in one call. **For a single
        perturbation every order and every composition mode is algebraically identical to
        `forward()`**, which is what `tests/test_composition_modes.py` pins.
        """
        if order is None:
            order = "mean" if average_orders else "ab"
        if order not in ("ab", "ba", "mean"):
            raise ValueError(f"order must be ab|ba|mean, got {order!r}")
        device = control_expression.device
        z_ctrl = self._encode_ctrl(control_expression)
        z_graph = self._pert_embeddings()

        n_steps = torch.tensor([sum(1 for x in p if x >= 0) for p in pert_indices],
                               device=device, dtype=torch.long)
        multi = (n_steps > 1)

        def chain(order_idx):
            """order_idx: 0 = as given, 1 = reversed. Returns final z_hat."""
            slot0 = [(list(reversed(p)) if order_idx else list(p))[0] if p else -1
                     for p in pert_indices]
            e0, m0 = self._slot_embed(z_graph, slot0, device)
            z1 = z_ctrl + self._interact(z_ctrl, e0, z_graph, m0)
            if not bool(multi.any()):
                return z1
            slot1 = [(list(reversed(p)) if order_idx else list(p))[1] if len(p) > 1 else -1
                     for p in pert_indices]
            e1, m1 = self._slot_embed(z_graph, slot1, device)
            z2 = z1 + self._interact(z1, e1, z_graph, m1)
            return torch.where(multi.unsqueeze(-1), z2, z1)

        if order == "ba":
            z_hat = chain(1)
        else:
            z_hat = chain(0)
            if order == "mean" and bool(multi.any()):
                # only the multi-perturbation rows differ between orders; single-pert rows
                # are left bit-identical rather than averaged with themselves
                z_hat = torch.where(multi.unsqueeze(-1), 0.5 * (z_hat + chain(1)), z_hat)

        decoder_out = self.decoder(z_hat)          # <-- the only decoder pass
        delta_z = z_hat - z_ctrl                   # cumulative effect vs original control

        if isinstance(decoder_out, dict):
            result = {"prediction": decoder_out["prediction"], "delta_z": delta_z,
                      "z_hat": z_hat, "z_ctrl": z_ctrl, "n_steps": n_steps}
            result.update({k: v for k, v in decoder_out.items() if k != "prediction"})
            return result
        return {"prediction": decoder_out, "delta_z": delta_z, "z_hat": z_hat,
                "z_ctrl": z_ctrl, "n_steps": n_steps}

    def forward_decode_sequential(
        self,
        control_expression: torch.Tensor,
        pert_indices: List[List[int]],
        order: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Chain the perturbations through the FULL pipeline, re-encoding in between.

            control -> [encode, interact A, decode] -> profile_A
            profile_A -> [encode, interact B, decode] -> profile_AB

        i.e. the first perturbation's PREDICTED EXPRESSION becomes the control input of the
        second round. Unlike `forward_sequential`, which chains in latent space and decodes
        once, this passes through gene space at every step, so each intermediate is a real
        prediction rather than a latent that nothing ever decodes.

        ⚠️ MEASURED HEADWIND, stated before the result: F-recon1 measured the decoder's own
        control reconstruction error at ~5x the effect size on rhj_fewshot
        (||D(E(c)) - c|| = 15.34 against a true effect of 2.88). This mode feeds a decoded
        profile back into the encoder, so step 2's input carries that error -- several times
        the signal being composed. It is run because it is cheap and the prediction is
        falsifiable, not because it is expected to win.

        ⚠️ Costs 2 encoder + 2 decoder passes (against 1 + 1 for `forward_sequential`).
        Rows with fewer than 2 perturbations keep their FIRST-round prediction; the second
        round is computed for the whole batch and then selected away, which is simpler than
        a ragged gather and exactly equivalent.
        """
        if order not in ("ab", "ba", "mean"):
            raise ValueError(f"order must be ab|ba|mean, got {order!r}")
        if order == "mean":
            a = self.forward_decode_sequential(control_expression, pert_indices, "ab")
            b = self.forward_decode_sequential(control_expression, pert_indices, "ba")
            n_steps = a["n_steps"]
            multi = (n_steps > 1).unsqueeze(-1)
            out = dict(a)
            for k in ("prediction", "z_hat", "delta_z"):
                # average only the rows the order can change; singles stay bit-identical
                out[k] = torch.where(multi, 0.5 * (a[k] + b[k]), a[k])
            return out

        device = control_expression.device
        z_graph = self._pert_embeddings()
        n_steps = torch.tensor([sum(1 for x in p if x >= 0) for p in pert_indices],
                               device=device, dtype=torch.long)
        multi = (n_steps > 1)
        seq = [(list(reversed(p)) if order == "ba" else list(p)) for p in pert_indices]

        def round_once(x, slot):
            z = self._encode_ctrl(x)
            e, m = self._slot_embed(z_graph, slot, device)
            z_hat = z + self._interact(z, e, z_graph, m)
            dec = self.decoder(z_hat)
            pred = dec["prediction"] if isinstance(dec, dict) else dec
            return pred, z, z_hat

        slot0 = [p[0] if p else -1 for p in seq]
        pred1, z_ctrl, z_hat1 = round_once(control_expression, slot0)
        if not bool(multi.any()):
            return {"prediction": pred1, "delta_z": z_hat1 - z_ctrl, "z_hat": z_hat1,
                    "z_ctrl": z_ctrl, "n_steps": n_steps}

        slot1 = [p[1] if len(p) > 1 else -1 for p in seq]
        pred2, z_mid, z_hat2 = round_once(pred1, slot1)
        pick = multi.unsqueeze(-1)
        prediction = torch.where(pick, pred2, pred1)
        z_hat = torch.where(pick, z_hat2, z_hat1)
        # ⚠️ `delta_z` is measured against the ORIGINAL control's latent, not against the
        # re-encoded intermediate: the diagnostics that read it (delta_z_ratio,
        # offmanifold_ratio) are defined relative to the cell the prediction started from.
        return {"prediction": prediction, "delta_z": z_hat - z_ctrl, "z_hat": z_hat,
                "z_ctrl": z_ctrl, "n_steps": n_steps}

    def forward_delta_sum(
        self,
        control_expression: torch.Tensor,
        pert_indices: List[List[int]],
        base: str = "recon",
    ) -> Dict[str, torch.Tensor]:
        """Sum the perturbations' DECODED deltas, each taken against the same reconstruction.

            recon   = D(E(c) + f(E(c), 0))               <- the zero-perturbation forward
            delta_p = D(E(c) + f(E(c), p)) - recon        for each perturbation p
            prediction = recon + sum_p delta_p

        The gene-space counterpart of the default `forward()`, which sums in the graph
        embedding instead. Two properties make it the most defensible of the sequential
        alternatives:

          * **SYMMETRIC BY CONSTRUCTION.** Summation commutes, so there is no AB-vs-BA
            question at all -- the object it builds has the same symmetry the biology has.
          * **IT CANCELS THE DECODER'S RECONSTRUCTION BIAS** instead of compounding it.
            Every delta is a difference of two decoder outputs from the same control, so the
            shared term D(E(c)) -- which F-recon1 measured at ~5x the effect size -- drops
            out. `forward_decode_sequential` feeds that term into its own second round; this
            subtracts it. Same logic as the decoder eval's `injctrl` convention.

        ⚠️ Costs 1 + n_perts decoder passes (3 for a double) and ONE encoder pass, reused.
        ⚠️ For a single perturbation this reduces EXACTLY to `forward()`:
           recon + (pred_A - recon) = pred_A. Pinned by the tests.
        """
        device = control_expression.device
        z_graph = self._pert_embeddings()
        z_ctrl = self._encode_ctrl(control_expression)

        def decode(z):
            dec = self.decoder(z)
            return dec["prediction"] if isinstance(dec, dict) else dec

        # ⚠️ THE REFERENCE IS THE ZERO-PERTURBATION FORWARD, not `decode(z_ctrl)`.
        # `forward()` gives a control row `D(z + f(z, 0))`, and `f(z, 0)` is NOT zero for
        # every interaction type (it is for `additive`, it is not for `film`). Using
        # `D(z_ctrl)` here made a control row disagree with `forward()` and would have put a
        # constant offset into every delta. Caught by
        # tests/test_composition_modes.py::test_control_rows_are_untouched_by_every_mode.
        e0, m0 = self._slot_embed(z_graph, [-1] * z_ctrl.shape[0], device)
        recon = decode(z_ctrl + self._interact(z_ctrl, e0, z_graph, m0))
        # `base` selects what the summed deltas are added BACK to. "recon" keeps the mode
        # comparable to `forward()` on a single perturbation; "real" is the `injctrl`
        # convention and is the symmetric control for `forward_residual_sequential`.
        if base not in ("real", "recon"):
            raise ValueError(f"base must be real|recon, got {base!r}")
        start = control_expression if base == "real" else recon
        n_steps = torch.tensor([sum(1 for x in p if x >= 0) for p in pert_indices],
                               device=device, dtype=torch.long)
        max_p = max((len(p) for p in pert_indices), default=1)
        total = torch.zeros_like(recon)
        delta_z_total = torch.zeros_like(z_ctrl)
        for k in range(max_p):
            slot = [p[k] if len(p) > k else -1 for p in pert_indices]
            if all(x < 0 for x in slot):
                continue
            e, m = self._slot_embed(z_graph, slot, device)
            d_z = self._interact(z_ctrl, e, z_graph, m)
            # a row with no k-th perturbation contributes exactly zero, not decoder noise
            pred_k = decode(z_ctrl + d_z)
            total = total + (pred_k - recon) * m
            delta_z_total = delta_z_total + d_z * m
        return {"prediction": start + total, "delta_z": delta_z_total,
                "z_hat": z_ctrl + delta_z_total, "z_ctrl": z_ctrl, "n_steps": n_steps}

    def _zero_pert_recon(self, z, z_graph, device):
        """``D(z + f(z, 0))`` -- what `forward()` emits for an UNPERTURBED row.

        The reference every decoded delta is taken against. `f(z, 0)` is not zero for every
        interaction (it is for `additive`, it is not for `film`), so `D(z)` is the wrong
        reference and using it puts a constant offset into every delta.
        """
        e0, m0 = self._slot_embed(z_graph, [-1] * z.shape[0], device)
        dec = self.decoder(z + self._interact(z, e0, z_graph, m0))
        return dec["prediction"] if isinstance(dec, dict) else dec

    def forward_residual_sequential(
        self,
        control_expression: torch.Tensor,
        pert_indices: List[List[int]],
        order: str = "mean",
        base: str = "real",
    ) -> Dict[str, torch.Tensor]:
        """Sequential composition in gene space on DECODED DELTAS added to the real control.

            recon_0 = D(E(c) + f(E(c), 0))
            delta_A = D(E(c) + f(E(c), e_A)) - recon_0
            x_1     = c + delta_A                      <- the REAL control plus the delta
            recon_1 = D(E(x_1) + f(E(x_1), 0))
            delta_B = D(E(x_1) + f(E(x_1), e_B)) - recon_1
            prediction = x_1 + delta_B  =  c + delta_A + delta_B

        This is `forward_decode_sequential` with the decoder's reconstruction bias REMOVED at
        every step: each round subtracts the reconstruction of its own input rather than
        passing a raw decoded profile forward. F-recon1 measured that bias at ~5x the effect
        size (||D(E(c)) - c|| = 15.34 vs an effect of 2.88), so this is the better-motivated
        of the two gene-space chains -- and unlike `forward_delta_sum`, the second delta IS
        conditioned on the first, because it is computed from E(c + delta_A).

        ⚠️ **THIS MODE CHANGES THE EMISSION CONVENTION, NOT ONLY THE COMPOSITION.** On a
        SINGLE perturbation it returns `c + delta_A`, not `pred_A`; the two differ by
        `(c - recon_0)`, i.e. the decoder's reconstruction error, which is LARGER than the
        effect being predicted. So its score against `sum` mixes two changes, and the
        comparison needs `delta_sum` with `base="real"` as the matching symmetric control:

            base="recon"  prediction = recon_0 + sum(delta)      comparable to `sum`
            base="real"   prediction = c       + sum(delta)      this mode's convention

        That 2x2 -- {recon, real} x {summed, sequential} -- separates the emission change
        from the sequencing change. Reporting only this mode would confound them.
        (The `base="real"` convention is the decoder eval's `injctrl`: score the perturbation
        delta against a real control rather than a reconstructed one.)

        ⚠️ Costs 2 encoder + 4 decoder passes per order.
        """
        if order not in ("ab", "ba", "mean"):
            raise ValueError(f"order must be ab|ba|mean, got {order!r}")
        if base not in ("real", "recon"):
            raise ValueError(f"base must be real|recon, got {base!r}")
        if order == "mean":
            a = self.forward_residual_sequential(control_expression, pert_indices, "ab", base)
            b = self.forward_residual_sequential(control_expression, pert_indices, "ba", base)
            multi = (a["n_steps"] > 1).unsqueeze(-1)
            out = dict(a)
            for k in ("prediction", "z_hat", "delta_z"):
                out[k] = torch.where(multi, 0.5 * (a[k] + b[k]), a[k])
            return out

        device = control_expression.device
        z_graph = self._pert_embeddings()
        n_steps = torch.tensor([sum(1 for x in p if x >= 0) for p in pert_indices],
                               device=device, dtype=torch.long)
        multi = (n_steps > 1)
        seq = [(list(reversed(p)) if order == "ba" else list(p)) for p in pert_indices]

        def step(x, slot):
            """(input + decoded delta, the step's latent, the step's delta_z)."""
            z = self._encode_ctrl(x)
            recon = self._zero_pert_recon(z, z_graph, device)
            e, m = self._slot_embed(z_graph, slot, device)
            d_z = self._interact(z, e, z_graph, m)
            dec = self.decoder(z + d_z)
            pred = dec["prediction"] if isinstance(dec, dict) else dec
            return x + (pred - recon) * m, z, d_z

        z0 = self._encode_ctrl(control_expression)
        start = control_expression if base == "real" else self._zero_pert_recon(
            z0, z_graph, device)
        x1, _z0, dz0 = step(start, [p[0] if p else -1 for p in seq])
        if not bool(multi.any()):
            return {"prediction": x1, "delta_z": dz0, "z_hat": z0 + dz0,
                    "z_ctrl": z0, "n_steps": n_steps}
        x2, z1, dz1 = step(x1, [p[1] if len(p) > 1 else -1 for p in seq])
        pick = multi.unsqueeze(-1)
        return {"prediction": torch.where(pick, x2, x1),
                "delta_z": dz0 + torch.where(pick, dz1, torch.zeros_like(dz1)),
                "z_hat": z0 + dz0 + torch.where(pick, dz1, torch.zeros_like(dz1)),
                "z_ctrl": z0, "n_steps": n_steps}

    # ⚠️ TWO EMISSION FAMILIES, and a number from one is not directly comparable to a
    # number from the other. Modes whose name ends in `_realctrl`/`resid_seq_*` add their
    # deltas to the REAL control cell (the `injctrl` convention); the rest add them to the
    # model's own reconstruction. On a single perturbation the first family returns
    # `c + delta` and the second returns `pred`, and they differ by the decoder's
    # reconstruction error -- which F-recon1 measured as LARGER than the effect itself.
    COMPOSE_MODES = ("sum", "latent_seq_ab", "latent_seq_ba", "latent_seq_mean",
                     "decode_seq_ab", "decode_seq_ba", "decode_seq_mean", "delta_sum",
                     "delta_sum_realctrl",
                     "resid_seq_ab", "resid_seq_ba", "resid_seq_mean")
    RECON_BASE_MODES = ("sum", "latent_seq_ab", "latent_seq_ba", "latent_seq_mean",
                        "decode_seq_ab", "decode_seq_ba", "decode_seq_mean", "delta_sum")
    REAL_BASE_MODES = ("delta_sum_realctrl", "resid_seq_ab", "resid_seq_ba",
                       "resid_seq_mean")

    def forward_composed(
        self,
        control_expression: torch.Tensor,
        pert_indices: List[List[int]],
        mode: str = "decode_seq_mean",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """One entry point for every multi-perturbation composition strategy.

        The modes differ ONLY for samples with 2+ perturbations; on a single perturbation
        they are algebraically identical, so a model trained on singles can be evaluated
        under all of them without retraining. That is what makes the comparison controlled:
        one set of weights, answered several ways.

        | mode              | where it composes            | symmetric in (A,B)? |
        |-------------------|------------------------------|---------------------|
        | `sum`             | graph embedding, before f    | yes                 |
        | `latent_seq_*`    | latent, chained through f     | only `_mean`        |
        | `decode_seq_*`    | gene space, re-encoded        | only `_mean`        |
        | `delta_sum`       | gene space, deltas summed     | yes, by construction|
        | `delta_sum_realctrl` | same, added to the REAL cell | yes, by construction|
        | `resid_seq_*`     | gene space, deltas CHAINED    | only `_mean`        |

        ⚠️ The last two are the `injctrl` emission family -- see `COMPOSE_MODES`. They do
        NOT reduce to `forward()` on a single perturbation, so read them against each other
        and against `delta_sum_realctrl`, never only against `sum`.

        ⛔ DEFAULT `decode_seq_mean`, adopted 2026-09-22 (user decision). Measured over the
        72-cell norman_combo matrix (6 arms x 12 modes x 2 panels, docs/norman_results.md):
        best mean rank of the 8 distinct modes (2.08 of 8), top-two on 5 of 6 arms -- no
        other mode is -- best non-additive `pearson_delta` marginal (0.6518), best `mae`
        marginal on both panels, and the single best cell overall (`NQ02__decode_seq_mean`,
        0.8169 additive / 0.7380 non-additive). `sum` is the only close rival (higher
        additive mean 0.7402 vs 0.7351, but it degrades FASTER into non-additivity than the
        `sum_of_singles` bar does). The default is a no-op on any single-perturbation
        corpus: single-perturbation rows keep their first-round prediction bit-identically,
        which tests/test_composition_modes.py pins.
        """
        if mode not in self.COMPOSE_MODES:
            raise ValueError(f"unknown compose mode {mode!r}; expected "
                             f"{list(self.COMPOSE_MODES)}")
        if mode == "sum":
            return self.forward(control_expression, pert_indices, **kwargs)
        if mode.startswith("latent_seq_"):
            return self.forward_sequential(control_expression, pert_indices,
                                           order=mode.rsplit("_", 1)[1])
        if mode.startswith("decode_seq_"):
            return self.forward_decode_sequential(control_expression, pert_indices,
                                                  order=mode.rsplit("_", 1)[1])
        if mode.startswith("resid_seq_"):
            return self.forward_residual_sequential(control_expression, pert_indices,
                                                    order=mode.rsplit("_", 1)[1],
                                                    base="real")
        if mode == "delta_sum_realctrl":
            return self.forward_delta_sum(control_expression, pert_indices, base="real")
        return self.forward_delta_sum(control_expression, pert_indices, base="recon")

    def train(self, mode: bool = True):
        """Override to keep encoder/decoder in eval mode when frozen."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        if self.freeze_decoder:
            self.decoder.eval()
        # Invalidate GNN cache when entering train mode
        if mode:
            self._gnn_cache = None
        return self

    def get_trainable_params(self) -> List[dict]:
        """Return param groups for optimizer.

        Returns list of dicts with 'params' and 'name' keys.
        GNN and interaction are always trainable.
        Decoder is trainable unless frozen.
        Encoder is only trainable when not frozen.
        """
        param_groups = [
            {"params": self.gnn.parameters(), "name": "gnn"},
            {"params": self.interaction.parameters(), "name": "interaction"},
        ]
        if not self.freeze_decoder:
            param_groups.append(
                {"params": self.decoder.parameters(), "name": "decoder"}
            )
        if not self.freeze_encoder:
            param_groups.append(
                {"params": self.encoder.parameters(), "name": "encoder"}
            )
        return param_groups
