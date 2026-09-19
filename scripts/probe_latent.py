"""delta_z_ratio on a trained VC checkpoint: has the model collapsed to the identity?

`delta_z_ratio = mean(||delta_z|| / ||z_ctrl||)`. Near 0 means the perturbation barely moves
the latent -- the model reproduces its control input, which scores at the identity floor and
reads as "a weak model" rather than a broken one (config.py, OCCAM_STATE_HYBRID.md section 8).

Inputs are REAL control means on the encoder axis, taken from cache/pert_means (the table the
delta losses use), so this is measured on the model's own training space.
"""
import sys, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_splithalf import load_model

CKPT = sys.argv[1]
trainer, ck = load_model(CKPT, "cuda")
model = trainer.model; model.eval()
p2i = trainer.pert2id

z = np.load("cache/pert_means/hepg2.npz", allow_pickle=True)
ctrl = np.asarray(z["ctrl_means"], dtype=np.float32)          # (1, 44728) HepG2 control mean
keys = [str(k) for k in z["keys"]]                            # "<line>|<pert>"
perts = [k.split("|", 1)[1] for k in keys]
have = [p for p in perts if p in p2i][:256]
print(f"probing {len(have)} real HepG2 perturbations against the real control mean")

X = torch.from_numpy(np.repeat(ctrl[:1], len(have), axis=0)).cuda()
pid = [[p2i[p]] for p in have]            # list-of-lists: forward() takes a per-row pert LIST
with torch.no_grad():
    out = model(X, pid)
print("forward keys:", sorted(out))
dz, zc = out.get("delta_z"), out.get("z_ctrl")
if dz is None or zc is None:
    print("⛔ this forward path does not expose delta_z / z_ctrl"); raise SystemExit(1)
d = dz.norm(dim=-1); c = zc.norm(dim=-1).clamp(min=1e-8); r = d / c
print(f"||delta_z||   mean {d.mean():.4f}")
print(f"||z_ctrl||    mean {c.mean():.4f}")
print(f"delta_z_ratio mean {r.mean():.6f}  median {r.median():.6f}  "
      f"p10 {r.quantile(0.1):.6f}  p90 {r.quantile(0.9):.6f}")
# do different perturbations actually move the latent DIFFERENTLY?
dzc = dz - dz.mean(0, keepdim=True)
print(f"perturbation-SPECIFIC share: ||dz - mean(dz)|| / ||dz|| = "
      f"{float((dzc.norm(dim=-1) / d.clamp(min=1e-8)).mean()):.4f}")
print()
print("READ: delta_z_ratio ~0 => COLLAPSED to identity (scores at the identity floor).")
print("      pert-specific share ~0 => one shared shift for every perturbation.")
