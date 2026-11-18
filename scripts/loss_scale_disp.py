#!/usr/bin/env python
"""Measured magnitude of the F-disp1 dispersion terms, and the SCALE-MATCHED weight for each.

WHY THIS EXISTS
    Q02's terms are scale-matched, not guessed: `dcos = 1 - cos(...)` is dimensionless and
    ~100x `smooth_l1` in log1p space (F-loss5), so a weight only means something next to the
    term's measured size. The three dispersion terms are dimensionless too, but on a different
    scale again, and a guessed weight would repeat the F6 confound (comparing loss
    CONFIGURATIONS while believing you are comparing mechanisms).

    The reference this script matches to is **the contribution DCOS makes in Q02 itself**:
        reference = delta_cos_weight * dcos_value = 0.5 * dcos(W23)
    A weight at 1x parity therefore adds a term of the same size as the term that currently
    dominates Q02's gradient.

WHAT IT MEASURES (on REAL single control cells, not the control mean -- dispersion is
per-cell by definition and a mean input has none)
    smooth_l1 per element · dcos · dcos_conc · radial_spread · gene_var
    plus the realised dispersion ratios, which is the F-disp1 diagnostic.

⚠️ The two moment-matching terms are `(m2 - 1)^2` where m2 is a NORMALISED second moment, so
they are ~1.0 whenever the prediction is collapsed (m2 -> 0) and 0.0 when the dispersion is
matched. Their steady-state starting value is therefore ~1 BY CONSTRUCTION; the measured
number below confirms it rather than discovering it.

    python scripts/loss_scale_disp.py <checkpoint> [n_perts] [n_ctrl]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CKPT = sys.argv[1]
N_PERT = int(sys.argv[2]) if len(sys.argv) > 2 else 256
N_CTRL = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SRC = "hepg2"
H5AD = "data/eval/pert/nadig_2024_hepg2.h5ad"

from eval_splithalf import load_model                                    # noqa: E402
from occam.vcell.dataset import _load_file_metadata                      # noqa: E402
from occam.vcell.stream_v2 import densify_csr, load_chunk_sparse, make_col_map  # noqa: E402

tr, ck = load_model(CKPT, "cuda")
model = tr.model
model.eval()
p2i = tr.pert2id
genes = [str(g) for g in ck["gene_names"]]
g2i = {g: i for i, g in enumerate(genes)}

zm = np.load(f"cache/pert_means/{SRC}.npz", allow_pickle=True)
zd = np.load(f"cache/pert_disp/{SRC}.npz", allow_pickle=True)
assert [str(k) for k in zm["keys"]] == [str(k) for k in zd["keys"]], \
    "means and disp tables disagree on group order -- rebuild the disp table"
perts = [str(k).split("|", 1)[1] for k in zm["keys"]]
gmeans = torch.from_numpy(np.asarray(zm["means"], dtype=np.float32)).cuda()
cmeans = torch.from_numpy(np.asarray(zm["ctrl_means"], dtype=np.float32)).cuda()
gline = np.asarray(zm["group_line"])
gmask = torch.from_numpy(np.asarray(zm["gene_mask"], dtype=bool)).cuda()
gvar = torch.from_numpy(np.asarray(zd["gvar"], dtype=np.float32)).cuda()
rad_m = torch.from_numpy(np.asarray(zd["rad_mean"], dtype=np.float32)).cuda()
rad_s = torch.from_numpy(np.asarray(zd["rad_std"], dtype=np.float32)).cuda()
cos_m = torch.from_numpy(np.asarray(zd["cos_mean"], dtype=np.float32)).cuda()

# ---- REAL control cells in model space, the same way the training stream builds them -------
target_sum = float(ck.get("config", {}).get("target_sum", 2000.0) or 2000.0)
fm = _load_file_metadata(H5AD, genes, g2i, gene_name_col="gene_name",
                         pert_gene_col="gene_name", batch_col=None)
is_c = np.flatnonzero(np.asarray(fm.is_control, dtype=bool))
col_map = make_col_map(fm.n_vars, fm.src_indices, fm.dst_indices)
take = is_c[:N_CTRL]
lo, hi = int(take.min()), int(take.max()) + 1
v, c, rp, rs = load_chunk_sparse(H5AD, lo, hi, fm.n_vars, np.arange(hi - lo), col_map)
X = densify_csr(torch.from_numpy(v).cuda(), torch.from_numpy(c).cuda(),
                torch.from_numpy(rp).cuda(), torch.from_numpy(rs).cuda(),
                len(genes), target_sum, False, False)
ctrl_cells = X[torch.from_numpy(take - lo).cuda()]
print(f"real control cells: {tuple(ctrl_cells.shape)} (target_sum {target_sum:g})")

sel = [i for i, p in enumerate(perts) if p in p2i][:N_PERT]
rows = np.repeat(np.asarray(sel), N_CTRL)
ci = np.tile(np.arange(N_CTRL), len(sel))
ctrl = ctrl_cells[torch.from_numpy(ci).cuda()]
gi = torch.from_numpy(rows).cuda()
with torch.no_grad():
    pred = model(ctrl, [[p2i[perts[i]]] for i in rows])["prediction"]

gmf = gmask.float().unsqueeze(0)
n_el = (gmf.expand_as(pred)).sum().clamp(min=1)
cm = cmeans[torch.from_numpy(gline[rows]).cuda()]
gm = gmeans[gi]
dp = (pred - cm) * gmf
dt = (gm - cm) * gmf

sl1 = (F.smooth_l1_loss(pred, gm, reduction="none") * gmf).sum() / n_el
dcos = (1 - F.cosine_similarity(dp, dt, dim=-1)).mean()
cos_pc = F.cosine_similarity(dp, dt, dim=-1)
dcos_conc = ((cos_pc - cos_m[gi]) ** 2).mean()

# radial: centre on the model's OWN mean radius per group (what the EMA converges to)
r = dp.norm(dim=-1)
r_by_g = torch.zeros(len(perts), device="cuda").index_add_(0, gi, r)
n_by_g = torch.zeros(len(perts), device="cuda").index_add_(0, gi, torch.ones_like(r))
r_own = (r_by_g / n_by_g.clamp(min=1))[gi]
u_r = (r - r_own) / rad_s[gi].clamp(min=1e-3)
m2_r = (u_r ** 2).mean()
rad_term = (m2_r - 1.0) ** 2

# per-gene: centre on the model's OWN per-group mean profile
sum_by_g = torch.zeros(len(perts), pred.shape[1], device="cuda").index_add_(0, gi, pred)
mu_own = (sum_by_g / n_by_g.clamp(min=1).unsqueeze(1))[gi]
sd = gvar[gi].clamp(min=1e-6).sqrt()
w = gmf * (sd > 1e-2).float()
m2_g = (((pred - mu_own) / sd) ** 2 * w).sum() / w.sum().clamp(min=1)
var_term = (m2_g - 1.0) ** 2

print(f"\nn = {len(sel)} perts x {N_CTRL} real control cells = {len(rows)} rows, "
      f"{int(gmask.sum())} present genes")
print(f"  smooth_l1 (per element)                   {sl1.item():.6f}")
print(f"  dcos = 1 - cos(dpred, dtrue)              {dcos.item():.6f}")
ref = 0.5 * dcos.item()
print(f"  REFERENCE = Q02's dcos contribution       {ref:.6f}   (delta_cos_weight 0.5)")
print(f"\n  MEASURED DISPERSION (this is F-disp1, re-measured per-cell here)")
print(f"  cos(pred delta, group delta)              {cos_pc.mean().item():.4f}"
      f"   REAL {cos_m[gi].mean().item():.4f}")
print(f"  radial  std_pred/std_real                 {m2_r.clamp(min=0).sqrt().item():.4f}"
      f"   (1.0 = matched)")
print(f"  per-gene std_pred/std_real                {m2_g.clamp(min=0).sqrt().item():.4f}"
      f"   (1.0 = matched)")
print(f"\n  TERM VALUES and the SCALE-MATCHED weight (weight = reference / term value)")
for name, val in (("dcos_conc_weight", dcos_conc.item()),
                  ("radial_spread_weight", rad_term.item()),
                  ("gene_var_weight", var_term.item())):
    w1 = ref / max(val, 1e-12)
    print(f"  {name:<22} term {val:.6f}   1x parity {w1:.4g}   5x {5 * w1:.4g}")
print("\n  ⚠️ 1x parity means the new term contributes as much as DCOS does in Q02 TODAY.")
