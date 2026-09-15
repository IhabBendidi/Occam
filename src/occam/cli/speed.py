"""Measure encoder inference throughput.

    occam-speed --checkpoint checkpoints/release/occam_small_encoder_decoder.pth

Reports samples/sec and the projected wall clock for a million cells, which is
the number that decides whether an embedding job is minutes or hours. Accepts
both a plain encoder checkpoint and a unified encoder+decoder one.
"""
import argparse
import os
import time

import torch

from occam.model import AutoEncoder

DEFAULT_CHECKPOINT = "checkpoints/release/occam_small_encoder_decoder.pth"


def load_encoder(checkpoint_path: str, device: str = "cuda") -> AutoEncoder:
    """Load encoder from a regular or unified decoder checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if ckpt.get("is_unified_checkpoint"):
        enc_cfg = ckpt["encoder_config"]
        model = AutoEncoder(
            num_genes=enc_cfg["num_genes"],
            model_size=enc_cfg["model_size"],
            hidden_size=enc_cfg["hidden_size"],
        ).to(device)
        model.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    else:
        from occam.inference.embed import load_trained_model
        model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        pipeline = load_trained_model(model_dir, device=device, checkpoint_path=checkpoint_path)
        model = pipeline.model

    return model


def benchmark(checkpoint_path=DEFAULT_CHECKPOINT, batch_size=32,
              total_samples=100_000, warmup_batches=10, device="cuda"):
    BATCH_SIZE, TOTAL_SAMPLES, WARMUP_BATCHES = batch_size, total_samples, warmup_batches
    model = load_encoder(checkpoint_path, device=device)
    model.eval()

    num_genes = model.num_genes
    hidden_size = model.hidden_size

    autocast_dtype = torch.bfloat16
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Model loaded. Genes: {num_genes}, Hidden size: {hidden_size}, Precision: {autocast_dtype}")
    print(f"Encoder parameters: {encoder_params:,}")

    dummy_data = torch.randn(BATCH_SIZE, num_genes, device=device) * 2

    # Warmup
    print(f"Warming up ({WARMUP_BATCHES} batches)...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
        for _ in range(WARMUP_BATCHES):
            _ = model.feature(dummy_data)
    torch.cuda.synchronize()

    # Benchmark
    num_batches = TOTAL_SAMPLES // BATCH_SIZE
    print(f"Benchmarking {num_batches} batches ({TOTAL_SAMPLES} samples, batch_size={BATCH_SIZE})...")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
        for _ in range(num_batches):
            _ = model.feature(dummy_data)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    samples_per_sec = TOTAL_SAMPLES / elapsed
    ms_per_batch = (elapsed / num_batches) * 1000

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Samples/sec: {samples_per_sec:,.0f}")
    print(f"  ms/batch: {ms_per_batch:.2f}")
    print(f"  Time for 1M samples: {1_000_000 / samples_per_sec:.1f}s")
    print(f"{'='*50}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="occam-speed", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help=f"encoder or unified checkpoint (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--total-samples", type=int, default=100_000)
    ap.add_argument("--warmup-batches", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        ap.error(f"checkpoint not found: {args.checkpoint}\n"
                 f"Run scripts/bootstrap_assets.sh, or pass --checkpoint.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        ap.error("CUDA is not available; pass --device cpu (throughput will not "
                 "be representative)")

    benchmark(checkpoint_path=args.checkpoint, batch_size=args.batch_size,
              total_samples=args.total_samples, warmup_batches=args.warmup_batches,
              device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
