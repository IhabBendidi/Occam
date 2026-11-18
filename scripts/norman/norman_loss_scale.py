#!/usr/bin/env python
"""Rescale M03 / M06 / Q02's loss weights for norman_combo, preserving their rhj RATIOS.

WHY THIS EXISTS
    The three bases are SCALE-MATCHED configurations, not arbitrary numbers: M03 has
    `dcos` at parity with `smooth_l1`, M06 has it ~50x above (direction-dominated), and Q02
    adds a latent MSE at parity. Those RATIOS are the hypothesis. But the terms have
    different units -- `dcos` is dimensionless and bounded [0,2] while `smooth_l1` is a
    per-element error in log1p space -- so moving to a dataset with a different target scale
    changes every ratio unless the weights move with it.

    Measured: norman's per-gene target scale is 0.859x rhj's and its effect size is 1.349x,
    so the weights CANNOT be copied across. This script measures each term at the INIT POINT
    -- the released encoder+decoder with an untrained interaction, which is the state
    training actually starts from -- on both benchmarks, and solves for the weights that
    reproduce each base's contribution shares.

    w_norman = w_rhj * |term|_rhj / |term|_norman

⚠️ MEASURED AT INIT, NOT ON A TRAINED CHECKPOINT. `dcos` shrinks as the model improves
(1 - cos goes from ~1 toward ~0.4), so a ratio read off a trained arm describes the end of
training, not the beginning, and would set the weights for a regime the optimiser has to
reach first. The init point is the only state both datasets share.

    python scripts/norman/norman_loss_scale.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

# The three historical loss configurations trained on rhj_fewshot.
BASES = {
    "M03": {"delta_cos_weight": 0.01, "delta_mag_weight": 0.0,
            "latent_loss_type": "none", "latent_loss_weight": 0.0},
    "M06": {"delta_cos_weight": 0.5, "delta_mag_weight": 0.01363,
            "latent_loss_type": "none", "latent_loss_weight": 0.0},
    "Q02": {"delta_cos_weight": 0.5, "delta_mag_weight": 0.01363,
            "latent_loss_type": "delta_z", "latent_loss_weight": 0.0044},
}


def _dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def measure(panel_dir, ck, device, n_perts=256, n_ctrl=8, seed=0):
    """The four term magnitudes at init, on real (control cell, perturbation) pairs.

    At init the perturbation conditioning is untrained, so `prediction` is essentially the
    control's reconstruction: `smooth_l1` is then the reconstruction error against the
    perturbed group mean, `dcos` is ~1 (an unaligned delta), `delta_mag` is the squared
    relative size error, and the latent MSE is the squared distance to the target delta_z.
    Those are exactly the magnitudes the weights have to balance on step 1.
    """
    z = np.load(os.path.join(panel_dir, "panel.npz"), allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)
    lab = np.array([str(x) for x in z["labels"]])
    ens = [_dec(x) for x in z["axis_ensembl"]]
    enc = [_dec(x) for x in ck["gene_names"]]
    pos = {g: i for i, g in enumerate(enc)}
    cols = np.array([pos[g] for g in ens], dtype=np.int64)

    ctrl_rows = np.flatnonzero(lab == "non-targeting")
    perts = [p for p in sorted(set(lab)) if p != "non-targeting"]
    rng = np.random.default_rng(seed)
    perts = list(rng.permutation(perts))[:n_perts]
    cm = X[ctrl_rows].mean(0)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/release/occam_small_encoder_decoder.pth")
    ap.add_argument("--rhj", default="data/benchmarks/rhj_fewshot/splithalf")
    ap.add_argument("--norman", default="data/benchmarks/norman_combo/splithalf")
    ap.add_argument("--n-perts", type=int, default=256)
    ap.add_argument("--n-ctrl", type=int, default=8)
    ap.add_argument("--out", default="data/benchmarks/norman_combo/loss_scales.json")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    enc_names = [_dec(x) for x in ck["gene_names"]]
    pos = {g: i for i, g in enumerate(enc_names)}
    n_enc = len(enc_names)

    # The released encoder + decoder, instantiated exactly as
    # `VirtualCellTrainer._load_from_unified_checkpoint` does -- same classes, same configs
    # from the checkpoint, and `output_activation` READ FROM THE CHECKPOINT rather than
    # defaulted (a softplus-trained decoder loads clean under the rectified_sigmoid default
    # and silently runs the wrong nonlinearity).
    from occam.decoder.external import HybridNormGlobalSkipDecoder
    from occam.model import AutoEncoder
    ec, dc = ck["encoder_config"], ck["decoder_config"]
    full = AutoEncoder(num_genes=len(enc_names), model_size=ec.get("model_size", "small"),
                       hidden_size=ec.get("hidden_size", 512))
    cleaned = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v
               for k, v in ck["encoder_state_dict"].items()}
    full.load_state_dict(cleaned, strict=False)
    encoder = full.encoder.to(dev).eval()
    decoder = HybridNormGlobalSkipDecoder(
        input_dim=dc["input_dim"], output_dim=dc["output_dim"], hidden_dim=dc["hidden_dim"],
        num_blocks=dc["num_blocks"], dropout_rate=dc.get("dropout_rate", 0.3),
        se_reduction=dc.get("se_reduction", 16),
        output_activation=dc.get("output_activation", "rectified_sigmoid"),
    )
    decoder.load_state_dict({k.replace("module.", "", 1): v
                             for k, v in ck["decoder_state_dict"].items()}, strict=False)
    decoder = decoder.to(dev).eval()
    print(f"[scale] encoder {ec.get('model_size')} hidden {ec.get('hidden_size')} · "
          f"decoder {dc['hidden_dim']}x{dc['num_blocks']} act "
          f"{dc.get('output_activation')} · {len(enc_names):,} genes on {dev}")

    terms = {}
    for name, pdir in (("rhj_fewshot", a.rhj), ("norman_combo", a.norman)):
        z = np.load(os.path.join(pdir, "panel.npz"), allow_pickle=True)
        X = np.asarray(z["X"], dtype=np.float32)
        lab = np.array([str(x) for x in z["labels"]])
        ens = [_dec(x) for x in z["axis_ensembl"]]
        cols = torch.as_tensor([pos[g] for g in ens], dtype=torch.long, device=dev)
        ctrl = np.flatnonzero(lab == "non-targeting")
        cm = X[ctrl].mean(0)
        perts = [p for p in sorted(set(lab)) if p != "non-targeting"]
        rng = np.random.default_rng(0)
        perts = [perts[i] for i in rng.permutation(len(perts))[:a.n_perts]]

        gmeans, ctrl_cells = [], X[ctrl[:a.n_ctrl]]
        for p in perts:
            gmeans.append(X[np.flatnonzero(lab == p)].mean(0))
        gm = torch.as_tensor(np.repeat(np.stack(gmeans), a.n_ctrl, axis=0), device=dev)
        cc = torch.as_tensor(np.tile(ctrl_cells, (len(perts), 1)), device=dev)
        cmt = torch.as_tensor(cm, device=dev).unsqueeze(0)

        with torch.no_grad():
            full = torch.zeros(cc.shape[0], n_enc, device=dev)
            full[:, cols] = cc
            zc = encoder(full)
            dec = decoder(zc)
            pred_full = dec["prediction"] if isinstance(dec, dict) else dec
            pred = pred_full[:, cols]
            # smooth_l1 per ELEMENT against the perturbed group mean (the training target
            # is a single cell; the group mean is its expectation and removes cell noise
            # from a scale measurement)
            sl1 = float(F.smooth_l1_loss(pred, gm, reduction="mean"))
            dp = pred - cmt
            dt = gm - cmt
            dcos = float((1 - F.cosine_similarity(dp, dt, dim=-1)).mean())
            tn = dt.norm(dim=-1).clamp(min=1e-6)
            mag = float((((dp.norm(dim=-1) / tn) - 1.0) ** 2).mean())
            # the latent MSE's target: the encoder's own delta for the true group mean
            gfull = torch.zeros_like(full)
            gfull[:, cols] = gm
            zt = encoder(gfull)
            lat = float(((zc - zt) ** 2).mean())
        terms[name] = {"smooth_l1": sl1, "dcos": dcos, "delta_mag": mag, "latent_mse": lat,
                       "n_rows": int(cc.shape[0]), "n_perts": len(perts),
                       "n_axis_genes": len(ens)}
        t = terms[name]
        print(f"\n[{name}]  {t['n_perts']} perts x {a.n_ctrl} controls = {t['n_rows']} rows, "
              f"{t['n_axis_genes']:,} genes")
        for k in ("smooth_l1", "dcos", "delta_mag", "latent_mse"):
            print(f"    {k:12} {t[k]:.6f}")

    r, n = terms["rhj_fewshot"], terms["norman_combo"]
    print("\n=== term magnitude RATIOS (norman / rhj) at init ===")
    ratio = {k: n[k] / r[k] for k in ("smooth_l1", "dcos", "delta_mag", "latent_mse")}
    for k, v in ratio.items():
        print(f"    {k:12} {v:.4f}x")

    # ⚠️ TWO DEFENSIBLE RESCALINGS, and the first one I wrote was the wrong one.
    #
    #   (a) preserve each term's ABSOLUTE contribution:  w * |t|  constant
    #       -> w_new = w_old / ratio_t
    #   (b) preserve each term's contribution RELATIVE TO smooth_l1:  w * |t| / |sl1|  constant
    #       -> w_new = w_old * ratio_sl1 / ratio_t
    #
    # (b) is what "keep the same ratios" means here, because the ratios that DEFINE the three
    # bases are ratios to the reconstruction term: M03 is "dcos at parity with smooth_l1",
    # M06 is "dcos ~50x smooth_l1". `smooth_l1` itself is 0.60x on norman, so under (a) the
    # dcos:smooth_l1 balance would silently grow by 1/0.60 = 1.67x and M03 would no longer be
    # the balanced arm it is defined to be. (a) is reported alongside for comparison only.
    print("\n=== RESCALED WEIGHTS ===")
    print("    (b) PRIMARY: preserve each term's ratio TO smooth_l1 -- the ratio that")
    print("        defines the bases.   w_new = w_old * ratio_sl1 / ratio_term")
    out = {"measured_at": "init (released encoder+decoder, untrained interaction)",
           "terms": terms, "ratios": ratio, "bases": {}, "bases_absolute_variant": {},
           "rescaling": "w_new = w_old * ratio_smooth_l1 / ratio_term (ratio = norman/rhj)"}
    for base, w in BASES.items():
        new, alt = dict(w), dict(w)
        for key, term in (("delta_cos_weight", "dcos"),
                          ("delta_mag_weight", "delta_mag"),
                          ("latent_loss_weight", "latent_mse")):
            if float(w.get(key, 0.0)) > 0:
                new[key] = float(w[key]) * ratio["smooth_l1"] / ratio[term]
                alt[key] = float(w[key]) / ratio[term]
        out["bases"][base] = new
        out["bases_absolute_variant"][base] = alt
        print(f"\n  {base}")
        for key, term in (("delta_cos_weight", "dcos"), ("delta_mag_weight", "delta_mag"),
                          ("latent_loss_weight", "latent_mse")):
            if float(w.get(key, 0.0)) > 0:
                x_r = float(w[key]) * r[term] / r["smooth_l1"]
                x_n = float(new[key]) * n[term] / n["smooth_l1"]
                print(f"    {key:20} {w[key]:.6g}  ->  {new[key]:.6g}"
                      f"      x smooth_l1: {x_r:.2f} -> {x_n:.2f}   "
                      f"(absolute-variant weight {alt[key]:.6g})")
        if new.get("latent_loss_type", "none") != "none":
            print(f"    latent_loss_type     {new['latent_loss_type']}")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
