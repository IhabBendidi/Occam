"""Measured magnitude of each loss term on a trained checkpoint.

`dcos = 1 - cos(...)` is bounded [0,2] and DIMENSIONLESS. `smooth_l1` in log1p space is a
per-gene error whose scale is set by the data. They are not comparable a priori, so the
configured weight alone says nothing about which term dominates the gradient.

⚠️ Normalisation is the whole point here: both terms must be per-ELEMENT means over the same
present-gene mask, or the ratio is off by the batch size.
"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"src"); sys.path.insert(0,"scripts")
from eval_splithalf import load_model

CKPT=sys.argv[1]
tr,ck=load_model(CKPT,"cuda"); model=tr.model; model.eval(); p2i=tr.pert2id
z=np.load("cache/pert_means/hepg2.npz",allow_pickle=True)
perts=[str(k).split("|",1)[1] for k in z["keys"]]
gmeans=np.asarray(z["means"],dtype=np.float32); cmeans=np.asarray(z["ctrl_means"],dtype=np.float32)
gline=np.asarray(z["group_line"]); gmask=torch.from_numpy(np.asarray(z["gene_mask"],dtype=bool)).cuda()
sel=[i for i,p in enumerate(perts) if p in p2i][:512]
ctrl=torch.from_numpy(cmeans[gline[sel]]).cuda(); gm=torch.from_numpy(gmeans[sel]).cuda()
with torch.no_grad(): pred=model(ctrl,[[p2i[perts[i]]] for i in sel])["prediction"]

gmf=gmask.float().unsqueeze(0).expand_as(pred)      # (B, G) -- NOT the 1-D mask
n_el=gmf.sum().clamp(min=1)                          # B * n_present_genes
dp=(pred-ctrl)*gmf; dt=(gm-ctrl)*gmf
dcos=(1-F.cosine_similarity(dp,dt,dim=-1)).mean()
sl1=(F.smooth_l1_loss(pred,gm,reduction="none")*gmf).sum()/n_el
mae=((pred-gm).abs()*gmf).sum()/n_el
eff=((gm-ctrl).abs()*gmf).sum()/n_el
print(f"\nn={len(sel)} perts x {int(gmask.sum())} present genes = {int(n_el)} elements")
print(f"  smooth_l1 (per element)                {sl1.item():.6f}")
print(f"  dcos = 1 - cos(dpred, dtrue)           {dcos.item():.6f}")
print(f"  RATIO dcos / smooth_l1                 {dcos.item()/max(sl1.item(),1e-12):.1f}x  <-- DCOS is this many times LARGER")
for w in (0.25,0.5,1.0):
    c=w*dcos.item()
    print(f"  at delta_cos_weight={w:<5} -> {100*c/(c+sl1.item()):5.1f}% of total loss "
          f"({c:.4f} vs expr {sl1.item():.4f})")
print(f"\n  MAE |pred - group_mean|                {mae.item():.6f}")
print(f"  true effect |group_mean - ctrl|        {eff.item():.6f}")
print(f"  MAGNITUDE RATIO ||dpred||/||dtrue||    "
      f"{(dp.norm(dim=-1)/dt.norm(dim=-1).clamp(min=1e-8)).mean().item():.3f}   (1.0 = correct size)")
print(f"  mean pred {pred.mean().item():.4f} vs mean group_mean {gm.mean().item():.4f} vs ctrl {ctrl.mean().item():.4f}")
