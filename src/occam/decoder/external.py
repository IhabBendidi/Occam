"""
Occam Decoder module for reconstruction.

Provides decoder architectures (SEResNetDecoder, HybridNormGlobalSkipDecoder) for
decoding Occam encoder embeddings back to gene expression in log1p space.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from typing import Tuple, List, Dict, Optional


# Numpy 2.x -> 1.x compatibility shim for loading checkpoints saved with numpy 2.x
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray


# ---------------------------------------------------------------------------
# Decoder size presets
#
# These size the EXTERNAL decoder (HybridNormGlobalSkipDecoder) trained by
# train_decoder.py. They are deliberately INDEPENDENT of the encoder's
# --model_size: a large encoder can be paired with a small decoder and vice
# versa, which is exactly what a decoder-capacity ablation needs.
#
# Do not confuse with model.MODEL_CONFIGS[*]['decoder_hidden'], which sizes the
# MAE's own internal decoder during pretraining.
#
# 'base' reproduces the current train_decoder.py defaults, so --decoder_size base
# is a no-op relative to today's behaviour.
# ---------------------------------------------------------------------------
DECODER_SIZE_CONFIGS = {
    'small':  {'hidden_dim': 1024, 'num_blocks': 3,
               'description': 'Small decoder (matches the 2k-HVG Chain B decoder)'},
    'base':   {'hidden_dim': 2048, 'num_blocks': 4,
               'description': 'Base decoder (current train_decoder.py default)'},
    'large':  {'hidden_dim': 3072, 'num_blocks': 6,
               'description': 'Large decoder'},
    'xlarge': {'hidden_dim': 4096, 'num_blocks': 8,
               'description': 'XLarge decoder'},
}

# Values used before --decoder_size existed; preserved as the no-preset fallback.
LEGACY_DECODER_HIDDEN_DIM = 2048
LEGACY_DECODER_NUM_BLOCKS = 4


def resolve_decoder_size(decoder_size=None, hidden_dim=None, num_blocks=None):
    """Resolve external-decoder geometry.

    Precedence: explicit value > ``decoder_size`` preset > legacy default.
    Passing an explicit ``hidden_dim``/``num_blocks`` alongside a preset overrides
    just that field, so ``--decoder_size large --decoder_num_blocks 4`` is valid.

    Returns:
        (hidden_dim, num_blocks)
    """
    preset = {}
    if decoder_size:
        if decoder_size not in DECODER_SIZE_CONFIGS:
            raise ValueError(
                f"decoder_size must be one of {list(DECODER_SIZE_CONFIGS)}, got {decoder_size!r}"
            )
        preset = DECODER_SIZE_CONFIGS[decoder_size]

    resolved_hidden = (
        hidden_dim if hidden_dim is not None
        else preset.get('hidden_dim', LEGACY_DECODER_HIDDEN_DIM)
    )
    resolved_blocks = (
        num_blocks if num_blocks is not None
        else preset.get('num_blocks', LEGACY_DECODER_NUM_BLOCKS)
    )
    return int(resolved_hidden), int(resolved_blocks)


class RectifiedSigmoid(nn.Module):
    """This activation function enables more straightforward optimziation landscape due to its boundedness.

    It could also be worth considering this paper https://www.alphaxiv.org/abs/2412.20851
    which defines rectified sigmoid as Re-σ(x) = 1/2 [ReLU(x + 1) − ReLU(x − 1)].

    Nonetheless, we have found that the current implementation facilitates stable and effective training.
    """

    def __init__(self, upper_bound: int = 5000):
        super().__init__()
        self.upper_bound = np.log1p(upper_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upper_bound * torch.relu(2 * torch.sigmoid(x / (2 * np.e)) - 1)  # type: ignore


def _make_output_activation(name: str) -> nn.Module:
    """Final activation of the decoder output head.

    ``rectified_sigmoid`` (default, legacy) is ub * relu(2*sigmoid(x/2e) - 1), which is
    EXACTLY 0 for every pre-activation <= 0 -- and its gradient there is exactly 0 too.
    Genes whose smooth_l1-optimal constant is ~0 (on the 44k vocab most genes are
    detected in <1% of training cells) get pushed into that region and can never leave:
    a dead-unit trap. Measured on TREK, 35,305 of 44,113 genes (80%) come out constant,
    and constant-vs-varying is exactly what Wilcoxon flags -- 57% of the false DE
    signal on control cells traces to them. nadig, whose panel is pre-filtered to
    well-expressed genes, has only 2.9% dead.

    ``softplus`` is strictly positive with a everywhere-nonzero gradient, so a rarely
    detected gene settles near its small empirical mean and keeps cell-to-cell
    variation instead of collapsing to a hard constant.

    Note softplus is UNBOUNDED above, unlike rectified_sigmoid's log1p(5000) ~ 8.52 cap.
    """
    name = (name or "rectified_sigmoid").lower()
    if name == "rectified_sigmoid":
        return RectifiedSigmoid()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(f"unknown output_activation: {name}")


class MonarchLinear(nn.Module):
    """Square Monarch linear layer (Stage B3): a full-rank structured replacement for
    ``nn.Linear(N, N)``.

    A Monarch matrix is the product of two block-diagonal matrices with a
    reshape/transpose permutation between them (the core building block shared by
    Monarch, arXiv:2204.00595, and BTT, arXiv:2406.06248).  Unlike a low-rank
    bottleneck (which lowers the rank and empirically HURT — see §22 of the decoder
    results doc), a Monarch map stays **full-rank** while using sub-quadratic params:

        params = N^2 / b + N * b   (vs N^2 dense)   ->  ~N^2/b for small b,
                                                        minimized 2*N^1.5 at b=sqrt(N)

    With N = nblocks * blk (blk = N // nblocks):
        factor1 (R): nblocks blocks of (blk x blk)   -> applied to reshape (nblocks, blk)
        permute    : (nblocks, blk) -> (blk, nblocks)
        factor2 (L): blk blocks of (nblocks x nblocks)
        permute back -> flatten to N

    Requires N % nblocks == 0.  This is square (in==out), which is all the decoder's
    residual-block MLPs need.  Backward-compat is handled by the caller (nblocks<=1
    falls back to nn.Linear).
    """

    def __init__(self, features: int, nblocks: int, bias: bool = True, mup: bool = False):
        super().__init__()
        if features % nblocks != 0:
            raise ValueError(f"MonarchLinear: features ({features}) must be divisible by nblocks ({nblocks})")
        self.features = features
        self.nblocks = nblocks
        self.blk = features // nblocks
        self.mup = mup
        # factor1: (nblocks, blk, blk); factor2: (blk, nblocks, nblocks)
        self.weight1 = nn.Parameter(torch.empty(self.nblocks, self.blk, self.blk))
        self.weight2 = nn.Parameter(torch.empty(self.blk, self.nblocks, self.nblocks))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        if mup:
            # muP / spectral parameterization: store SMALL weights (std=1/fan_in) and apply a FIXED
            # forward multiplier sqrt(fan_in). At init the effective matrix has the same distribution
            # as 'default', but because the trainable weight is small + multiplier fixed, Adam updates
            # are width-correctly scaled (Tensor Programs / Compute Better Spent muP for structured layers).
            self.register_buffer("mult1", torch.tensor(self.blk ** 0.5))
            self.register_buffer("mult2", torch.tensor(self.nblocks ** 0.5))
            nn.init.normal_(self.weight1, std=1.0 / self.blk)
            nn.init.normal_(self.weight2, std=1.0 / self.nblocks)
        else:
            self.mult1 = None
            self.mult2 = None
            # Variance-preserving init: gain through factor1 = blk*Var(W1), factor2 = nblocks*Var(W2);
            # product == 1 when Var(W1)=1/blk, Var(W2)=1/nblocks.
            nn.init.normal_(self.weight1, std=(1.0 / self.blk) ** 0.5)
            nn.init.normal_(self.weight2, std=(1.0 / self.nblocks) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n0 = x.shape[0]
        x = x.reshape(n0, self.nblocks, self.blk)
        x = torch.einsum("pij,npj->npi", self.weight1, x)        # (n0, nblocks, blk)
        if self.mup:
            x = x * self.mult1
        x = x.transpose(1, 2).contiguous()                       # (n0, blk, nblocks)
        x = torch.einsum("qij,nqj->nqi", self.weight2, x)        # (n0, blk, nblocks)
        if self.mup:
            x = x * self.mult2
        x = x.transpose(1, 2).contiguous().reshape(n0, self.features)
        if self.bias is not None:
            x = x + self.bias
        return x


def _balanced_factors(n: int):
    """Most-balanced (m, p) with m*p == n and m <= p (for Kronecker A(m×m)⊗B(p×p))."""
    m = int(n ** 0.5)
    while m > 1 and n % m != 0:
        m -= 1
    return m, n // m


class KroneckerLinear(nn.Module):
    """Stage B6: square Kronecker-sum linear — a full-rank structured replacement for Linear(N,N).

    W ≈ Σ_r A_r ⊗ B_r  with A_r∈ℝ^{m×m}, B_r∈ℝ^{p×p}, m·p = N (Kronecker GPT 2110.08152; KronA 2212.10650
    — "low-rank suffers limited representation power → use Kronecker"). Params = rank·(m²+p²) ≪ N². Unlike
    low-rank (§22), a single Kronecker term is already full-rank; the rank knob adds capacity cheaply.
    Forward uses  y = Σ_r A_r X B_rᵀ  with X = x.reshape(m, p)  (the (A⊗B)x = vec(A X Bᵀ) identity).
    """

    def __init__(self, features: int, rank: int = 1, bias: bool = True, m: int = 0, p: int = 0):
        super().__init__()
        if m and p:
            assert m * p == features
        else:
            m, p = _balanced_factors(features)
        self.features, self.m, self.p, self.rank = features, m, p, rank
        self.weightA = nn.Parameter(torch.empty(rank, m, m))
        self.weightB = nn.Parameter(torch.empty(rank, p, p))
        self.bias = nn.Parameter(torch.zeros(features)) if bias else None
        # variance-preserving: one term gains m·p=N; R indep terms sum -> R·N·Var(A)Var(B)=1.
        std = (1.0 / (rank * features)) ** 0.25
        nn.init.normal_(self.weightA, std=std)
        nn.init.normal_(self.weightB, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n0 = x.shape[0]
        X = x.reshape(n0, self.m, self.p)
        tmp = torch.einsum("rik,nkp->nrip", self.weightA, X)     # A_r X         (n0,R,m,p)
        Y = torch.einsum("nrip,rjp->nij", tmp, self.weightB)     # Σ_r (A_r X) B_rᵀ (n0,m,p)
        y = Y.reshape(n0, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y


def kron_project(W: torch.Tensor, m: int, p: int, rank: int):
    """Analytic nearest rank-R Kronecker-sum projection (Pitsianis–Van Loan): rearrange W into
    R(W)[(i,j),(k,l)] = W[i*p+k, j*p+l] (shape m²×p²) = Σ_r vec(A_r)vec(B_r)ᵀ, then truncated SVD.
    Returns (weightA, weightB) shaped (rank,m,m)/(rank,p,p) matching KroneckerLinear. Optimal for
    true Kronecker-sum W; best approx otherwise (Kronecker GPT warm-start)."""
    Wf = W.detach().to(torch.float32)
    Rw = Wf.reshape(m, p, m, p).permute(0, 2, 1, 3).reshape(m * m, p * p)  # [(i,j),(k,l)]
    maxr = min(m * m, p * p)
    r = min(rank, maxr)
    # truncated SVD (top-r only) -> fast; full SVD of (m²×p²) is huge. oversample for accuracy.
    q = min(maxr, r + 8)
    U, S, V = torch.svd_lowrank(Rw, q=q)             # U:(m²,q) S:(q) V:(p²,q)
    s = S[:r].clamp_min(0).sqrt()
    A = (U[:, :r] * s).T.reshape(r, m, m)            # vec over (i,j) -> (m,m)
    B = (V[:, :r] * s).T.reshape(r, p, p)            # vec over (k,l) -> (p,p)
    if r < rank:  # pad extra terms with zeros
        A = torch.cat([A, torch.zeros(rank - r, m, m)], 0)
        B = torch.cat([B, torch.zeros(rank - r, p, p)], 0)
    return A.to(W.dtype), B.to(W.dtype)


def monarch_project(W: torch.Tensor, nblocks: int):
    """Analytical dense->Monarch projection (warm-start) matching MonarchLinear's parameterization.

    MonarchLinear's dense-equivalent map is  M[(i,q),(p,j)] = W2[q,i,p] * W1[p,q,j]  where the
    output flat index is i*blk+q and the input flat index is p*blk+j (i,p in [0,nblocks); q,j in
    [0,blk); blk = H/nblocks).  For each (q,p) the sub-block A_{q,p}[i,j] = W[i*blk+q, p*blk+j]
    must equal the rank-1 outer product u[i]*v[j] with u=W2[q,:,p], v=W1[p,q,:].  The optimal
    (Frobenius) rank-1 factor of each sub-block is its top SVD component -- giving the closed-form
    best Monarch approximation of W (Monarch, arXiv:2204.00595).  Returns (W1, W2) shaped exactly
    as MonarchLinear.weight1 (nblocks, blk, blk) / weight2 (blk, nblocks, nblocks).
    """
    H = W.shape[0]
    assert W.shape == (H, H) and H % nblocks == 0
    b = nblocks
    blk = H // b
    Wf = W.detach().to(torch.float32)
    # reshape rows (i*blk+q) -> (b, blk) = (i, q); cols (p*blk+j) -> (b, blk) = (p, j)
    W4 = Wf.reshape(b, blk, b, blk)                # [i, q, p, j]
    A = W4.permute(1, 2, 0, 3).reshape(blk * b, b, blk)   # [(q,p), i, j]
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)   # U:(QP,b,k) S:(QP,k) Vh:(QP,k,blk), k=min(b,blk)
    s0 = S[:, 0].clamp_min(0).sqrt()              # (QP,)
    u = U[:, :, 0] * s0[:, None]                  # (QP, b)   = W2[q,:,p]
    v = Vh[:, 0, :] * s0[:, None]                 # (QP, blk) = W1[p,q,:]
    u = u.reshape(blk, b, b)                       # [q, p, i]
    v = v.reshape(blk, b, blk)                     # [q, p, j]
    W2 = u.permute(0, 2, 1).contiguous()          # [q, i, p] -> weight2 (blk, b, b)
    W1 = v.permute(1, 0, 2).contiguous()          # [p, q, j] -> weight1 (b, blk, blk)
    return W1.to(W.dtype), W2.to(W.dtype)


def warmstart_monarch_from_dense(decoder: nn.Module, dense_state_dict: Dict, nblocks: int) -> int:
    """Warm-start a Monarch decoder from a trained DENSE decoder's state_dict (Stage B3 fair test).

    All non-block-MLP params (input_layer, output_head, BatchNorms, SE gate, log params) are copied
    directly (identical keys/shapes); each block's two square ``Linear(H,H)`` weights are projected
    to their best Monarch factors via ``monarch_project`` and their biases copied. Returns #linears
    projected. Assumes the decoder uses default (non-muP) MonarchLinear parameterization.
    """
    own = decoder.state_dict()
    new = {}
    projected = 0
    for k in own.keys():
        if k.endswith("weight1") or k.endswith("weight2"):
            base = k.rsplit(".", 1)[0]            # res_blocks.{i}.block.{2|6}
            Wd = dense_state_dict.get(base + ".weight")
            if Wd is None:
                new[k] = own[k]; continue
            W1, W2 = monarch_project(Wd, nblocks)
            new[base + ".weight1"] = W1
            new[base + ".weight2"] = W2
            if k.endswith("weight1"):
                projected += 1
        elif k in dense_state_dict and dense_state_dict[k].shape == own[k].shape:
            new[k] = dense_state_dict[k]
        else:
            new[k] = own[k]                        # keep init (e.g. monarch bias not in dense)
    decoder.load_state_dict(new, strict=True)
    return projected


def warmstart_kron_from_dense(decoder: nn.Module, dense_state_dict: Dict, rank: int) -> int:
    """Warm-start a Kronecker decoder from a trained DENSE decoder (Stage B6). Non-block params copied
    directly; each block's Linear(H,H) projected to its nearest rank-R Kronecker sum (`kron_project`).
    Returns #linears projected. Uses each KroneckerLinear's own (m,p)."""
    own = decoder.state_dict()
    # map weightA key -> its module (for m,p)
    mods = {n: mod for n, mod in decoder.named_modules() if isinstance(mod, KroneckerLinear)}
    new = {}
    projected = 0
    for k in own.keys():
        if k.endswith("weightA") or k.endswith("weightB"):
            base = k.rsplit(".", 1)[0]
            Wd = dense_state_dict.get(base + ".weight")
            if Wd is None:
                new[k] = own[k]; continue
            mod = mods[base]
            A, B = kron_project(Wd, mod.m, mod.p, rank)
            new[base + ".weightA"] = A
            new[base + ".weightB"] = B
            if k.endswith("weightA"):
                projected += 1
        elif k in dense_state_dict and dense_state_dict[k].shape == own[k].shape:
            new[k] = dense_state_dict[k]
        else:
            new[k] = own[k]
    decoder.load_state_dict(new, strict=True)
    return projected


def _mask_24(w: torch.Tensor) -> torch.Tensor:
    """Boolean 2:4 mask along the LAST (reduction) dim: keep top-2-of-every-4 by |w|."""
    shp = w.shape
    g = w.reshape(-1, 4)
    idx = g.abs().topk(2, dim=1, largest=True).indices
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask.reshape(shp)


class _Sparse24STE(torch.autograd.Function):
    """SR-STE (arXiv:2102.04010): forward uses the 2:4-masked weight; backward passes the
    gradient straight through (so pruned weights can re-enter the mask) PLUS a decay term
    ``decay*(1-mask)*w`` that shrinks pruned weights to stop the mask oscillating."""

    @staticmethod
    def forward(ctx, w, decay):
        mask = _mask_24(w)
        ctx.save_for_backward(w, mask)
        ctx.decay = decay
        return w * mask.to(w.dtype)

    @staticmethod
    def backward(ctx, g):
        w, mask = ctx.saved_tensors
        grad = g + ctx.decay * (~mask).to(w.dtype) * w
        return grad, None


class Sparse24Parametrization(nn.Module):
    """Weight parametrization (Stage B4): dynamically re-masks a weight to 2:4 each forward
    via SR-STE. Registered with ``torch.nn.utils.parametrize`` so the dense weight is stored
    as ``...original`` and masking is transparent. Tensors whose last dim isn't a multiple of
    4 (or < 4) are left dense (e.g. tiny SE projections if ever wrapped)."""

    def __init__(self, decay: float = 2e-4):
        super().__init__()
        self.decay = decay

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if w.shape[-1] < 4 or w.shape[-1] % 4 != 0:
            return w
        return _Sparse24STE.apply(w, self.decay)


def apply_sparsity_24(decoder: nn.Module, decay: float = 2e-4) -> int:
    """Register 2:4 SR-STE parametrization on the param-dominant linears of a decoder:
    every ``nn.Linear`` and ``MonarchLinear`` EXCEPT the small Squeeze-Excitation gate
    (names containing ``.se.``). Composes with Monarch (B3+B4). Returns #weights wrapped."""
    n = 0
    for name, m in decoder.named_modules():
        if ".se." in name or name.endswith(".se"):
            continue
        if isinstance(m, MonarchLinear):
            parametrize.register_parametrization(m, "weight1", Sparse24Parametrization(decay))
            parametrize.register_parametrization(m, "weight2", Sparse24Parametrization(decay))
            n += 2
        elif isinstance(m, nn.Linear):
            parametrize.register_parametrization(m, "weight", Sparse24Parametrization(decay))
            n += 1
    return n


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block with LayerNorm stabilization."""
    
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.attention_computer = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attention_computer(x)
        return x * y


class SEResidualBlock(nn.Module):
    """Pre-activation SE Residual Block with BatchNorm1d (Sequential variant)."""
    
    def __init__(self, channels: int, dropout_rate: float = 0.2, se_reduction: int = 16,
                 bottleneck_k: int = 1, monarch_blocks: int = 0, monarch_mup: bool = False,
                 kron_rank: int = 0):
        super().__init__()
        # Stage B3 (Monarch/BTT): if monarch_blocks > 1, the two square H×H Linears become
        # full-rank MonarchLinear maps (params ~ 2H²/b vs 2H²), keeping rank (unlike the
        # low-rank bottleneck §22). monarch_blocks <= 1 -> plain Linear (backward-compat).
        # Monarch requires square maps, so it composes only with bottleneck_k == 1.
        use_monarch = monarch_blocks is not None and monarch_blocks > 1
        use_kron = kron_rank is not None and kron_rank > 0
        if use_kron:
            # Stage B6 (Kronecker): two square Linears -> KroneckerLinear (sum of kron_rank A⊗B terms).
            self.block = nn.Sequential(
                nn.BatchNorm1d(channels), nn.GELU(),
                KroneckerLinear(channels, rank=kron_rank), nn.Dropout(dropout_rate),
                nn.BatchNorm1d(channels), nn.GELU(),
                KroneckerLinear(channels, rank=kron_rank), nn.Dropout(dropout_rate),
            )
        elif use_monarch:
            if bottleneck_k != 1:
                raise ValueError("monarch_blocks>1 requires bottleneck_k==1 (Monarch maps are square)")
            lin1 = MonarchLinear(channels, monarch_blocks, mup=monarch_mup)
            lin2 = MonarchLinear(channels, monarch_blocks, mup=monarch_mup)
            self.block = nn.Sequential(
                nn.BatchNorm1d(channels),
                nn.GELU(),
                lin1,
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                lin2,
                nn.Dropout(dropout_rate)
            )
        else:
            # Inverted bottleneck: the two H×H Linears become H→H/k→H, cutting the dominant
            # 2H² cost to 2H²/k. bottleneck_k=1 -> mid=H -> EXACTLY the original block (backward-compat).
            mid = max(1, channels // bottleneck_k)
            self.block = nn.Sequential(
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Linear(channels, mid),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(mid),
                nn.GELU(),
                nn.Linear(mid, channels),
                nn.Dropout(dropout_rate)
            )
        self.se = SqueezeExcitation(channels, reduction=se_reduction)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = self.se(residual)
        return x + residual


class GLUResidualBlock(nn.Module):
    """Stage B5: SwiGLU-gated residual block (replaces SEResidualBlock's plain Linear->act->Linear).

    A gated FFN — `down( SiLU(gate(x)) * up(x) )` — gives more quality per parameter than a plain
    2-linear MLP (GLU Variants, arXiv:2002.05202; SwiGLU is the standard FFN in LLaMA/PaLM/Mistral).
    Inner width `d = round(glu_ratio * channels)`; the three projections cost `3*channels*d`, so
    `glu_ratio = 2/3` is param-NEUTRAL vs the dense block's `2*channels^2`, and smaller ratios bank
    the savings. Keeps the pre-norm, SE gate, dropout and residual that make the decoder strong.
    """

    def __init__(self, channels: int, dropout_rate: float = 0.2, se_reduction: int = 16,
                 glu_ratio: float = 0.667):
        super().__init__()
        d = max(1, int(round(glu_ratio * channels)))
        self.norm = nn.BatchNorm1d(channels)
        self.gate_proj = nn.Linear(channels, d)
        self.up_proj = nn.Linear(channels, d)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.down_proj = nn.Linear(d, channels)
        self.se = SqueezeExcitation(channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.gate_proj(h)) * self.up_proj(h)
        h = self.dropout(self.down_proj(h))
        h = self.se(h)
        return x + h


class SEResNetDecoder(nn.Module):
    """Occam Decoder using SEResidualBlocks.

    Takes embeddings from Occam encoder and decodes to gene expression in log1p space.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 2048, 
        num_blocks: int = 3, 
        dropout_rate: float = 0.2, 
        se_reduction: int = 16
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[SEResidualBlock(hidden_dim, dropout_rate, se_reduction=se_reduction) 
              for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class HybridNormGlobalSkipDecoder(nn.Module):
    """Occam Decoder with global skip connection from input projection.

    Uses res_blocks/output_head naming convention (differs from SEResNetDecoder).

    When ``hurdle_mode=True``, the output head produces 2 channels per gene
    (mean + logit_pi) and exposes a per-gene Gamma shape parameter.  The
    forward method returns a dict instead of a tensor.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 3072,
        num_blocks: int = 4,
        dropout_rate: float = 0.3,
        se_reduction: int = 16,
        hurdle_mode: bool = False,
        pi_head_mode: bool = False,
        pi_head_arch: str = "linear",
        pi_head_bottleneck_dim: int = 256,
        pi_head_dropout: float = 0.0,
        weight_tie: bool = False,
        bottleneck_k: int = 1,
        monarch_blocks: int = 0,
        monarch_mup: bool = False,
        kron_rank: int = 0,
        glu_mode: bool = False,
        glu_ratio: float = 0.667,
        sparsity_24: bool = False,
        sparsity_24_decay: float = 2e-4,
        output_activation: str = "rectified_sigmoid",
    ):
        super().__init__()
        self.output_activation = output_activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hurdle_mode = hurdle_mode
        self.pi_head_mode = pi_head_mode
        # weight_tie: a single SEResidualBlock applied num_blocks times (recurrent) -> param
        # count is constant regardless of num_blocks (only compute scales). Backward-compat: off.
        self.weight_tie = weight_tie
        self._n_apply = num_blocks

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        n_distinct = 1 if weight_tie else num_blocks
        if glu_mode:
            # Stage B5: SwiGLU-gated blocks (mutually exclusive with monarch/bottleneck for now).
            self.res_blocks = nn.Sequential(
                *[GLUResidualBlock(hidden_dim, dropout_rate, se_reduction=se_reduction,
                                   glu_ratio=glu_ratio)
                  for _ in range(n_distinct)]
            )
        else:
            self.res_blocks = nn.Sequential(
                *[SEResidualBlock(hidden_dim, dropout_rate, se_reduction=se_reduction,
                                  bottleneck_k=bottleneck_k, monarch_blocks=monarch_blocks,
                                  monarch_mup=monarch_mup, kron_rank=kron_rank)
                  for _ in range(n_distinct)]
            )

        if hurdle_mode:
            self.output_head = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * output_dim),
            )
            self._activation = RectifiedSigmoid()
            self.log_shape_param = nn.Parameter(torch.full((output_dim,), 1.0))
        else:
            self.output_head = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                _make_output_activation(output_activation),
            )

        if pi_head_mode:
            self.pi_head_input_dropout = nn.Dropout(pi_head_dropout) if pi_head_dropout > 0 else nn.Identity()
            if pi_head_arch == "bottleneck":
                self.pi_head = nn.Sequential(
                    nn.Linear(hidden_dim, pi_head_bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(pi_head_bottleneck_dim, output_dim),
                )
            else:
                self.pi_head = nn.Linear(hidden_dim, output_dim)

        # Stage B4: 2:4 SR-STE sparsity on the param-dominant linears (composes with Monarch).
        # Off by default -> nothing registered -> bit-identical dense (backward-compat).
        self.sparsity_24 = sparsity_24
        if sparsity_24:
            apply_sparsity_24(self, decay=sparsity_24_decay)

    def forward(self, x: torch.Tensor):
        x_projected = self.input_layer(x)
        if getattr(self, "weight_tie", False):
            # recurrent: apply the single shared block num_blocks times
            h = x_projected
            for _ in range(self._n_apply):
                h = self.res_blocks(h)
            x_residual = h
        else:
            x_residual = self.res_blocks(x_projected)
        x_sum = x_projected + x_residual  # Global skip connection
        raw = self.output_head(x_sum)

        if self.hurdle_mode:
            m = self._activation(raw[:, :self.output_dim])
            logit_pi = raw[:, self.output_dim:]
            shape = (nn.functional.softplus(self.log_shape_param) + 0.05).clamp(max=100.0)
            return {"m": m, "shape": shape, "logit_pi": logit_pi, "prediction": m}

        if self.pi_head_mode:
            logit_pi = self.pi_head(self.pi_head_input_dropout(x_sum))
            return {"prediction": raw, "logit_pi": logit_pi}

        return raw


def _infer_config_from_state_dict(state_dict: Dict) -> Dict:
    """Infer model config from state_dict weights shapes."""
    # input_dim and hidden_dim from input_layer
    input_dim = state_dict["input_layer.weight"].shape[1]
    hidden_dim = state_dict["input_layer.weight"].shape[0]
    
    # output_dim from output_head (last linear layer)
    # output_head is Sequential: [BatchNorm, GELU, Linear, ReLU]
    output_dim = state_dict["output_head.2.weight"].shape[0]
    
    # num_blocks: count unique block indices in res_blocks
    block_indices = set()
    for k in state_dict.keys():
        if k.startswith("res_blocks."):
            idx = int(k.split(".")[1])
            block_indices.add(idx)
    num_blocks = len(block_indices)
    
    # se_reduction: infer from SE attention_computer Linear layer
    # attention_computer.1 is Linear(channel, channel//reduction)
    se_key = "res_blocks.0.se.attention_computer.1.weight"
    if se_key in state_dict:
        reduced_dim = state_dict[se_key].shape[0]
        se_reduction = hidden_dim // reduced_dim
    else:
        se_reduction = 16  # default
    
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "dropout_rate": 0.3,  # can't infer from weights
        "se_reduction": se_reduction,
    }


def load_decoder(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    config_checkpoint_path: Optional[str] = None,
) -> Tuple[nn.Module, List[str], Dict[str, int], Dict]:
    """
    Load Occam Decoder model and gene metadata from checkpoint.

    Supports both standalone decoder checkpoints and unified encoder+decoder
    checkpoints (with ``is_unified_checkpoint`` flag).

    Args:
        checkpoint_path: Path to the .pt/.pth checkpoint file (weights)
        device: Device to load model onto
        config_checkpoint_path: Optional separate checkpoint for gene metadata only.
                               Model config is always inferred from weights checkpoint.

    Returns:
        Tuple of (model, gene_names, gene_to_index, model_config)
        model_config includes 'target_sum' key (value None if not found in checkpoint)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get state_dict from weights checkpoint
    # Unified checkpoints use 'decoder_state_dict'; fall back to 'model_state_dict'
    if isinstance(checkpoint, dict) and "decoder_state_dict" in checkpoint:
        state_dict = checkpoint["decoder_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  # state_dict only checkpoint
    
    # Use explicit decoder_config if present (unified checkpoints), else infer from weights
    if isinstance(checkpoint, dict) and "decoder_config" in checkpoint:
        model_config = dict(checkpoint["decoder_config"])
    else:
        model_config = _infer_config_from_state_dict(state_dict)

    # Extract target_sum from checkpoint if available
    target_sum = None
    if "target_sum" in checkpoint:
        target_sum = checkpoint["target_sum"]
    elif "library_size" in checkpoint:
        target_sum = checkpoint["library_size"]
    elif "training_config" in checkpoint and isinstance(checkpoint["training_config"], dict):
        target_sum = checkpoint["training_config"].get("target_sum")
    model_config["target_sum"] = target_sum
    
    # Load gene metadata from config checkpoint if provided, else from weights checkpoint
    if config_checkpoint_path:
        config_checkpoint = torch.load(config_checkpoint_path, map_location=device, weights_only=False)
        gene_names = config_checkpoint["gene_names"]
        gene_to_index = config_checkpoint["gene_to_index"]
    else:
        gene_names = checkpoint["gene_names"]
        gene_to_index = checkpoint["gene_to_index"]
    
    # Detect architecture from state_dict keys
    uses_hybrid = any("res_blocks" in k for k in state_dict.keys())
    
    if uses_hybrid:
        model = HybridNormGlobalSkipDecoder(
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_blocks=model_config["num_blocks"],
            dropout_rate=model_config["dropout_rate"],
            se_reduction=model_config["se_reduction"],
            weight_tie=model_config.get("weight_tie", False),
            output_activation=model_config.get("output_activation", "rectified_sigmoid"),
            bottleneck_k=model_config.get("bottleneck_k", 1),
            monarch_blocks=model_config.get("monarch_blocks", 0),
            monarch_mup=model_config.get("monarch_mup", False),
            kron_rank=model_config.get("kron_rank", 0),
            glu_mode=model_config.get("glu_mode", False),
            glu_ratio=model_config.get("glu_ratio", 0.667),
            sparsity_24=model_config.get("sparsity_24", False),
            sparsity_24_decay=model_config.get("sparsity_24_decay", 2e-4),
        ).to(device)
    else:
        model = SEResNetDecoder(
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_blocks=model_config["num_blocks"],
            dropout_rate=model_config["dropout_rate"],
            se_reduction=model_config["se_reduction"],
        ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, gene_names, gene_to_index, model_config


def verify_gene_order(
    checkpoint_genes: List[str],
    query_genes: List[str],
) -> Tuple[bool, Optional[List[int]], str]:
    """
    Verify if query genes match checkpoint genes and compute reorder indices if needed.
    
    Args:
        checkpoint_genes: Gene names from the trained checkpoint (defines expected order)
        query_genes: Gene names from the data to be used for inference
        
    Returns:
        Tuple of (is_match, reorder_indices, message)
        - is_match: True if genes can be aligned (exact match or reorderable)
        - reorder_indices: If reordering needed, indices to reorder query to match checkpoint.
                          None if exact match or incompatible.
        - message: Description of the verification result
    """
    checkpoint_set = set(checkpoint_genes)
    query_set = set(query_genes)
    
    # Exact match in order
    if checkpoint_genes == query_genes:
        return True, None, "Gene order matches exactly"
    
    # Same genes, different order
    if checkpoint_set == query_set and len(checkpoint_genes) == len(query_genes):
        # Build mapping from query index to checkpoint index
        query_gene_to_idx = {g: i for i, g in enumerate(query_genes)}
        reorder_indices = [query_gene_to_idx[g] for g in checkpoint_genes]
        return True, reorder_indices, f"Genes match but need reordering ({len(checkpoint_genes)} genes)"
    
    # Gene mismatch
    missing = checkpoint_set - query_set
    extra = query_set - checkpoint_set
    msg_parts = []
    if missing:
        msg_parts.append(f"Missing {len(missing)} genes (e.g., {list(missing)[:3]})")
    if extra:
        msg_parts.append(f"Extra {len(extra)} genes (e.g., {list(extra)[:3]})")
    if len(checkpoint_genes) != len(query_genes):
        msg_parts.append(f"Count mismatch: checkpoint={len(checkpoint_genes)}, query={len(query_genes)}")
    
    return False, None, "; ".join(msg_parts)


def run_inference(
    model: nn.Module,
    embeddings: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Run inference on embeddings.
    
    Args:
        model: Trained decoder model
        embeddings: Input embeddings, shape (n_samples, input_dim)
        device: Device to run on
        batch_size: Batch size for inference
        
    Returns:
        Predictions in log1p space, shape (n_samples, n_genes)
    """
    model.eval()
    n_samples = embeddings.shape[0]
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            X_batch = torch.from_numpy(embeddings[i : i + batch_size]).float().to(device)
            pred_batch = model(X_batch).cpu().numpy()
            all_preds.append(pred_batch)
    
    return np.concatenate(all_preds, axis=0)


