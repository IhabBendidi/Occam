#!/usr/bin/env python
"""Verify an Occam checkpoint against a recorded expectation.

Two profiles, because there are two artifacts with different evidence behind
them:

``--profile release`` (default)
    ``checkpoints/release/occam_small_encoder_decoder.pth`` -- the model:
    stage-2 encoder + the softplus/NB decoder, checked against
    ``checkpoints/release_meta.json``.
``--profile legacy``
    ``checkpoints/legacy/occam_small_encoder_decoder_rectified_sigmoid.pth`` --
    the superseded rectified_sigmoid/smooth-L1 decoder, checked against
    ``golden/release_meta.json``. This is the PORT-EQUIVALENCE gate: the golden
    fixtures were captured from the research tree, so this profile is what
    proves the port reproduces it bit-exactly. It is kept for exactly that
    reason and is not shipped as a model.

    python scripts/verify_checkpoint.py                    # the release
    python scripts/verify_checkpoint.py --profile legacy   # the port gate
    python scripts/verify_checkpoint.py --checkpoint X --meta Y
    python scripts/verify_checkpoint.py --write-expect     # re-record the release

⚠️ What the release profile does and does not prove. Its expectation file is
generated FROM the promoted checkpoint, so agreement with it shows the file has
not drifted or been swapped -- not that the weights are correct. Two checks in
that profile are not circular, and they are the ones that matter:

* the **golden latent** must reproduce. The encoder half of the release is
  byte-identical to the legacy one's, and the golden fixture came from the
  research tree, so this still pins the published encoder to the weights the
  port was verified against.
* the release's ``encoder_state_dict`` digest must equal the legacy one's.

The golden **recon** is NOT checked in the release profile: it is the legacy
decoder's output, and the release decoder is a different network (81 tensors,
128,456,385 params vs 99 and 137,393,859).

Checks, in order (later ones assume the earlier ones passed):

1.  file size and sha256 against ``checkpoints/MANIFEST.json``
2.  top-level key inventory
3.  ``encoder_config`` / ``decoder_config`` / ``target_sum`` / ``epoch`` /
    ``best_val_loss`` / gene-name digest against ``golden/release_meta.json``
4.  decoder loads ``strict=True`` with the expected tensor and parameter counts
5.  encoder loads ``strict=True`` via :class:`occam.model.OccamEncoder`
6.  the golden forward pass reproduces
7.  the output activation is determined EMPIRICALLY, because the released
    checkpoint does not record it

On (7): ``rectified_sigmoid`` is exactly 0 for every non-positive
pre-activation and its gradient there is exactly 0 too, so a large fraction of
genes come out constant. ``softplus`` is strictly positive everywhere. The two
are trivially separable from a single forward pass, so this reports evidence
rather than trusting the code default -- a softplus-trained decoder loaded under
the ``rectified_sigmoid`` default still loads cleanly and silently emits ~99%
exact zeros.

On (6): the forward pass is NOT bit-stable across torch versions. Replaying the
fixture under torch 2.1.2 rather than the 2.4.1 it was captured with shifts
~52% of latent elements by up to 1.4e-6. So bit-exactness is required only when
the torch version matches the fixture; otherwise a tolerance is used and the
substitution is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _ensure_src_on_path() -> None:
    """Make `occam` importable from a source checkout without PYTHONPATH."""
    import os as _os
    _src = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    if _os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)


_ensure_src_on_path()

import numpy as np
import torch

DEFAULT_GOLDEN = "golden"
DEFAULT_MANIFEST = "checkpoints/MANIFEST.json"
LEGACY_CHECKPOINT = "checkpoints/legacy/occam_small_encoder_decoder_rectified_sigmoid.pth"

PROFILES = {
    # profile -> (checkpoint, expectation file, manifest asset key)
    "release": ("checkpoints/release/occam_small_encoder_decoder.pth",
                "checkpoints/release_meta.json", "stage3_release"),
    "legacy": (LEGACY_CHECKPOINT, "golden/release_meta.json", "stage3_legacy"),
}

# Tolerance used when the torch version differs from the fixture's.
CROSS_VERSION_ATOL = 1e-4
CROSS_VERSION_RTOL = 1e-3

# A decoder head that leaves at least this fraction of genes constant across
# cells is rectified_sigmoid, not softplus. Measured separation is ~0.84 vs 0.0.
DEAD_GENE_FRACTION_THRESHOLD = 0.20


class Failure(Exception):
    pass


class Check:
    """Collects pass/fail results so one run reports every problem, not just the first."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))

    def fail(self, label: str, detail: str) -> None:
        self.failures.append(f"{label}: {detail}")
        print(f"  FAIL  {label}\n          {detail}")

    def expect(self, label: str, got, want, detail: str = "") -> None:
        if got == want:
            self.ok(label, detail or f"{got}")
        else:
            self.fail(label, f"expected {want!r}, got {got!r}")

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  NOTE  {text}")


def file_sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_list(items) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode())
        h.update(b"\0")
    return h.hexdigest()


def state_dict_sha256(state_dict) -> str:
    """Content digest of a state dict: key names and raw tensor bytes, key-sorted."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write_expectation(ckpt_path: str, meta_path: str, profile: str) -> str:
    """Record a promoted checkpoint's structure as this profile's expectation.

    Deliberately NOT automatic: an expectation regenerated to silence a failing
    check would turn this gate into a no-op. Run it once, at promotion.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gene_names = list(ck["gene_names"])
    esd, dsd = ck["encoder_state_dict"], ck["decoder_state_dict"]
    meta = {
        "profile": profile,
        "recorded_from": os.path.basename(ckpt_path),
        "source_checkpoint_bytes": os.path.getsize(ckpt_path),
        "source_checkpoint_sha256": file_sha256(ckpt_path),
        "top_level_keys": sorted(ck.keys()),
        "encoder_config": dict(ck["encoder_config"]),
        "decoder_config": dict(ck["decoder_config"]),
        "target_sum": ck.get("target_sum"),
        "epoch": ck.get("epoch"),
        "best_val_loss": ck.get("best_val_loss"),
        "n_genes": len(gene_names),
        "gene_names_sha256": sha256_list(gene_names),
        "encoder_state_dict": {
            "n_params": count_params(esd), "n_tensors": len(esd),
            "keys": sorted(esd), "sha256": state_dict_sha256(esd),
        },
        "decoder_state_dict": {
            "n_params": count_params(dsd), "n_tensors": len(dsd),
            "sha256": state_dict_sha256(dsd),
        },
        "model_state_dict_is_decoder": state_dict_sha256(
            ck.get("model_state_dict", {})) == state_dict_sha256(dsd),
        "torch_version": torch.__version__,
        # The golden recon fixture is the LEGACY decoder's output; only that
        # profile may compare against it.
        "golden_recon_applies": profile == "legacy",
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return (f"decoder {meta['decoder_state_dict']['n_tensors']} tensors / "
            f"{meta['decoder_state_dict']['n_params']:,} params, "
            f"epoch {meta['epoch']}, activation "
            f"{meta['decoder_config'].get('output_activation')}")


def count_params(state_dict) -> int:
    total = 0
    for v in state_dict.values():
        n = 1
        for d in tuple(v.shape):
            n *= int(d)
        total += n
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--profile", default="release", choices=sorted(PROFILES))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--meta", default=None, help="expectation JSON; overrides the profile")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--skip-hash", action="store_true", help="skip the ~1 GB sha256 pass")
    ap.add_argument("--write-expect", action="store_true",
                    help="record this checkpoint's structure as the profile's "
                         "expectation file and exit. Run it once, when a new "
                         "checkpoint is promoted -- never to make a failure go away.")
    args = ap.parse_args()

    repo = args.repo
    prof_ckpt, prof_meta, manifest_key = PROFILES[args.profile]
    ckpt_path = args.checkpoint or os.path.join(repo, prof_ckpt)
    meta_path = args.meta or os.path.join(repo, prof_meta)
    golden_dir = args.golden or os.path.join(repo, DEFAULT_GOLDEN)
    manifest_path = args.manifest or os.path.join(repo, DEFAULT_MANIFEST)

    print(f"Occam checkpoint verification  [profile: {args.profile}]")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  expectation: {meta_path}")
    print(f"  golden     : {golden_dir}")
    print(f"  torch      : {torch.__version__}")
    print()

    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}\n"
              f"Run scripts/bootstrap_assets.sh first.", file=sys.stderr)
        return 2

    fwd_path = os.path.join(golden_dir, "release_forward.npz")

    if args.write_expect:
        written = write_expectation(ckpt_path, meta_path, args.profile)
        print(f"wrote {meta_path}\n  {written}")
        return 0

    if not os.path.isfile(meta_path):
        print(f"ERROR: expectation file not found: {meta_path}\n"
              f"  For --profile legacy, run scripts/porting/capture_goldens.py from\n"
              f"  the source repo. For --profile release, record the promoted\n"
              f"  checkpoint with --write-expect.", file=sys.stderr)
        return 2
    with open(meta_path) as fh:
        meta = json.load(fh)

    c = Check()

    # ---- 1. bytes ------------------------------------------------------
    print("[1] file identity")
    c.expect("byte size", os.path.getsize(ckpt_path), meta["source_checkpoint_bytes"])
    expected_sha = meta.get("source_checkpoint_sha256")
    if args.skip_hash:
        c.note("sha256 skipped (--skip-hash)")
    elif not expected_sha:
        c.note("golden metadata carries no sha256; cannot verify bytes")
    else:
        c.expect("sha256", file_sha256(ckpt_path), expected_sha, detail=expected_sha[:16] + "...")

    if os.path.isfile(manifest_path):
        with open(manifest_path) as fh:
            man = json.load(fh)
        entry = man.get("assets", {}).get(manifest_key, {})
        if entry.get("sha256") and expected_sha:
            c.expect("manifest agrees with golden", entry["sha256"], expected_sha,
                     detail="both records match")
    else:
        c.note(f"no manifest at {manifest_path}")

    # ---- 2/3. structure ------------------------------------------------
    print("\n[2] key inventory and recorded config")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c.expect("top-level keys", sorted(ck.keys()), meta["top_level_keys"],
             detail=f"{len(ck)} keys")
    c.expect("is_unified_checkpoint", bool(ck.get("is_unified_checkpoint")), True)
    c.expect("encoder_config", dict(ck["encoder_config"]), meta["encoder_config"])
    c.expect("decoder_config", dict(ck["decoder_config"]), meta["decoder_config"])
    c.expect("target_sum", ck.get("target_sum"), meta["target_sum"])
    c.expect("epoch", ck.get("epoch"), meta["epoch"])
    c.expect("best_val_loss", ck.get("best_val_loss"), meta["best_val_loss"])

    gene_names = list(ck["gene_names"])
    c.expect("n_genes", len(gene_names), meta["n_genes"])
    c.expect("gene_names digest", sha256_list(gene_names), meta["gene_names_sha256"],
             detail=meta["gene_names_sha256"][:16] + "...")
    gti = ck.get("gene_to_index", {})
    inverse_ok = len(gti) == len(gene_names) and all(
        gti.get(g) == i for i, g in enumerate(gene_names)
    )
    c.expect("gene_to_index is the inverse of gene_names", inverse_ok, True)

    if meta.get("model_state_dict_is_decoder"):
        c.note("model_state_dict is byte-identical to decoder_state_dict, NOT the whole "
               "model -- anything treating it as the model gets a silent encoder-shaped hole")

    # ---- 4. decoder strict=True ----------------------------------------
    print("\n[3] decoder loads strict=True")
    from occam.decoder.external import HybridNormGlobalSkipDecoder

    dcfg = dict(ck["decoder_config"])
    decoder = HybridNormGlobalSkipDecoder(
        input_dim=dcfg["input_dim"],
        output_dim=dcfg["output_dim"],
        hidden_dim=dcfg["hidden_dim"],
        num_blocks=dcfg["num_blocks"],
        dropout_rate=dcfg["dropout_rate"],
        se_reduction=dcfg["se_reduction"],
        # Use the recorded head when there is one. Building a softplus decoder
        # under the rectified_sigmoid default loads strict=True and then emits
        # ~99% exact zeros, so the forward checks below would be meaningless.
        **({"output_activation": dcfg["output_activation"]}
           if dcfg.get("output_activation") else {}),
    )
    dsd = ck["decoder_state_dict"]
    try:
        missing, unexpected = decoder.load_state_dict(dsd, strict=True)
        c.expect("decoder missing keys", sorted(missing), [])
        c.expect("decoder unexpected keys", sorted(unexpected), [])
    except RuntimeError as exc:
        c.fail("decoder strict load", str(exc).splitlines()[0])

    c.expect("decoder tensor count", len(dsd), meta["decoder_state_dict"]["n_tensors"])
    c.expect("decoder param count", count_params(dsd), meta["decoder_state_dict"]["n_params"])
    # num_blocks=5 is NOT the code default (4). res_blocks.4.* is the proof.
    last_block = f"res_blocks.{dcfg['num_blocks'] - 1}."
    c.expect(f"deepest residual block present ({last_block}*)",
             any(k.startswith(last_block) for k in dsd), True,
             detail=f"num_blocks={dcfg['num_blocks']}")
    decoder.eval()

    # ---- 5. encoder strict=True ----------------------------------------
    print("\n[4] encoder loads strict=True via OccamEncoder")
    from occam.model import OccamEncoder

    try:
        encoder, info = OccamEncoder.from_checkpoint(ck, strict=True)
        c.expect("encoder missing keys", info["missing"], [])
        c.expect("encoder unexpected keys", info["unexpected"], [])
        c.expect("encoder keys loaded", info["loaded"],
                 sorted(meta["encoder_state_dict"]["keys"]),
                 detail=f"{len(info['loaded'])} tensors")
        c.expect("encoder param count", count_params(ck["encoder_state_dict"]),
                 meta["encoder_state_dict"]["n_params"])
    except Exception as exc:  # noqa: BLE001
        c.fail("encoder strict load", f"{type(exc).__name__}: {exc}")
        encoder = None

    total = meta["encoder_state_dict"]["n_params"] + meta["decoder_state_dict"]["n_params"]
    c.ok("unified param count", f"{total:,}")

    # ---- 6. golden forward ---------------------------------------------
    print("\n[5] golden forward pass")
    if encoder is None:
        c.fail("golden forward", "skipped: encoder failed to load")
    elif not os.path.isfile(fwd_path):
        c.fail("golden forward", f"fixture missing: {fwd_path}")
    else:
        g = np.load(fwd_path)
        x = torch.from_numpy(g["x"])
        with torch.no_grad():
            latent = encoder(x)
            recon = decoder(latent)
        latent_np = latent.numpy()
        recon_np = recon.numpy()

        fixture_torch = meta.get("torch_version")
        same_torch = fixture_torch == torch.__version__
        if not same_torch:
            c.note(f"torch {torch.__version__} != fixture torch {fixture_torch}; "
                   f"using tolerance atol={CROSS_VERSION_ATOL} rtol={CROSS_VERSION_RTOL} "
                   f"instead of bit-exactness")

        pairs = [("latent", latent_np, g["latent"])]
        if meta.get("golden_recon_applies", True):
            pairs.append(("recon", recon_np, g["recon"]))
        else:
            c.note("golden recon NOT checked: the fixture is the legacy decoder's "
                   "output and this release carries a different decoder. The "
                   "latent check above still pins the encoder, which is "
                   "byte-identical between the two.")
            finite = bool(np.isfinite(recon_np).all())
            c.expect("release recon is finite", finite, True)
            c.expect("release recon is strictly positive (softplus head)",
                     bool((recon_np > 0).all()), True,
                     detail=f"min={recon_np.min():.4g}")
        for name, got, want in pairs:
            maxdiff = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
            if same_torch:
                c.expect(f"{name} bit-exact", np.array_equal(got, want), True,
                         detail=f"max|diff|={maxdiff:.3e}")
            else:
                close = np.allclose(got, want, atol=CROSS_VERSION_ATOL, rtol=CROSS_VERSION_RTOL)
                c.expect(f"{name} within cross-version tolerance", close, True,
                         detail=f"max|diff|={maxdiff:.3e}")

    # ---- 6b. the published encoder is still the verified one -----------
    if meta.get("encoder_state_dict", {}).get("sha256"):
        print("\n[5b] encoder identity")
        c.expect("encoder_state_dict digest", state_dict_sha256(ck["encoder_state_dict"]),
                 meta["encoder_state_dict"]["sha256"],
                 detail="same stage-2 encoder the goldens pin")

    # ---- 7. output activation, empirically -----------------------------
    print("\n[6] output activation (determined, not assumed)")
    recorded = dict(ck["decoder_config"]).get("output_activation")
    if recorded:
        c.ok("output_activation recorded in checkpoint", recorded)
        verdict = recorded
        evidence = {"source": "recorded in decoder_config"}
    else:
        c.note("output_activation is NOT recorded in this checkpoint; determining it "
               "from a forward pass. The code default is 'rectified_sigmoid', and a "
               "softplus-trained decoder would load cleanly under it while emitting "
               "~99% exact zeros -- so the default is not evidence.")
        g = np.load(fwd_path)
        x = torch.from_numpy(g["x"])
        evidence = {}
        for activation in ("rectified_sigmoid", "softplus"):
            probe = HybridNormGlobalSkipDecoder(
                input_dim=dcfg["input_dim"], output_dim=dcfg["output_dim"],
                hidden_dim=dcfg["hidden_dim"], num_blocks=dcfg["num_blocks"],
                dropout_rate=dcfg["dropout_rate"], se_reduction=dcfg["se_reduction"],
                output_activation=activation,
            )
            probe.load_state_dict(dsd, strict=True)
            probe.eval()
            with torch.no_grad():
                out = probe(encoder(x)).numpy()
            zero_frac = float((out == 0.0).mean())
            dead = int((out.std(axis=0) == 0.0).sum())
            evidence[activation] = {
                "exact_zero_fraction": zero_frac,
                "zero_variance_genes": dead,
                "zero_variance_fraction": dead / out.shape[1],
                "min": float(out.min()),
                "max": float(out.max()),
            }
            print(f"        {activation:20s} exact-zero {zero_frac:6.4f}  "
                  f"constant genes {dead:6d}/{out.shape[1]} "
                  f"({dead / out.shape[1]:.1%})  range [{out.min():.4f}, {out.max():.4f}]")

        rs = evidence["rectified_sigmoid"]["zero_variance_fraction"]
        sp = evidence["softplus"]["zero_variance_fraction"]
        if rs >= DEAD_GENE_FRACTION_THRESHOLD > sp:
            verdict = "rectified_sigmoid"
        elif sp >= DEAD_GENE_FRACTION_THRESHOLD > rs:
            verdict = "softplus"
        else:
            verdict = None

        if verdict:
            c.ok("output activation determined", verdict)
            c.note(f"weights were trained with '{verdict}': under it "
                   f"{evidence[verdict]['zero_variance_fraction']:.1%} of genes are "
                   f"constant across cells, the dead-unit signature of a "
                   f"threshold-at-zero head. Record this in MODEL_CARD.md.")
        else:
            c.fail("output activation determination",
                   f"inconclusive: rectified_sigmoid {rs:.1%} vs softplus {sp:.1%} "
                   f"constant genes; neither side of the {DEAD_GENE_FRACTION_THRESHOLD:.0%} "
                   f"threshold separates them")

    # ---- report --------------------------------------------------------
    out_path = os.path.join(golden_dir, "verify_report.json")
    with open(out_path, "w") as fh:
        json.dump({
            "checkpoint": ckpt_path,
            "torch_version": torch.__version__,
            "fixture_torch_version": meta.get("torch_version"),
            "checks_passed": c.passed,
            "failures": c.failures,
            "notes": c.notes,
            "output_activation_verdict": verdict,
            "output_activation_evidence": evidence,
        }, fh, indent=2, sort_keys=True)

    print(f"\n{'=' * 70}")
    if c.failures:
        print(f"FAILED  {len(c.failures)} check(s), {c.passed} passed")
        for f in c.failures:
            print(f"  - {f}")
        print(f"report: {out_path}")
        return 1
    print(f"OK  all {c.passed} checks passed")
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
