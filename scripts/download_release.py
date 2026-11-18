import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "bendidiihab/Occam"
PERT_FILES = ["nadig_2024_hepg2.h5ad", "nadig_2024_jurkat.h5ad", "replogle_2022_rpe1_e.h5ad"]
BENCHMARKS = {
    "rhj": "rhj_fewshot", "norman": "norman_combo", "h1": "h1_seen",
    "k562": "k562_seen", "k562-h1": "k562_seen_h1set",
}


def download_plan(resource, benchmark=None, seed=42):
    if seed not in range(42, 47):
        raise ValueError("Released training seeds are 42 through 46")
    if resource == "encoder-decoder":
        return [("checkpoints/occam_small_encoder_decoder.pth", "checkpoints/release/occam_small_encoder_decoder.pth")]
    if resource == "virtual-cell":
        if benchmark == "rhj":
            source = f"checkpoints/occam_seeds/rhj_fewshot/all_sources/seed_{seed}/best_model.pth"
        elif benchmark == "norman":
            source = f"checkpoints/vcell/norman_combo_NF32ALLA/seed_{seed}/best_model.pth"
        else:
            raise ValueError("virtual-cell requires --benchmark rhj or --benchmark norman")
        return [(source, f"checkpoints/vcell/{benchmark}/seed_{seed}/best_model.pth")]
    if resource == "knowledge":
        graphs = ["go/go_essential_all.csv", "go/go_essential_top20.csv", "string/v11.5.parquet", "master_gene_set_sorted.csv"]
        embeddings = ["genept_full_occam_rhj_fewshot.pkl", "esm2_t33_650M_UR50D_occam_rhj_fewshot.pkl"]
        return [("graphs/" + name, "data/graphs/" + name) for name in graphs] + [
            ("assets/" + name + suffix, "data/assets/" + name + suffix)
            for name in embeddings for suffix in ("", ".json")
        ]
    if resource == "data":
        if benchmark == "representation":
            files = ["eval/pert/" + name for name in PERT_FILES]
            files += ["eval/scib/pancreas_scib.h5ad", "eval/scib/pbmc_68k.h5ad"]
        elif benchmark in BENCHMARKS:
            name = BENCHMARKS[benchmark]
            files = [f"benchmarks/{name}/splithalf/panel.npz"]
            if benchmark == "rhj":
                files += ["eval/pert/" + name for name in PERT_FILES]
            elif benchmark == "norman":
                files.append("eval/pert/norman_2019_k562.h5ad")
            elif benchmark == "h1":
                files.append("benchmarks/h1_seen/h1_seen_raw.h5ad")
            else:
                files.append("benchmarks/k562_seen/k562_seen_raw.h5ad")
        else:
            raise ValueError("data requires --benchmark rhj, norman, representation, h1, k562, or k562-h1")
        return [(name, "data/" + name) for name in files]
    raise ValueError(f"Unknown resource: {resource}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_file(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and (
            os.path.samefile(source, destination) or sha256(source) == sha256(destination)
        ):
            return
        raise FileExistsError(f"Existing file differs: {destination}. Move it aside before downloading.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".occam-download-", dir=destination.parent)
    os.close(handle)
    os.unlink(temporary)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    p = argparse.ArgumentParser(description="Download only the Occam models or data you select.")
    p.add_argument("resource", choices=["encoder-decoder", "virtual-cell", "data", "knowledge"])
    p.add_argument("--benchmark", choices=list(BENCHMARKS) + ["representation"])
    p.add_argument("--seed", type=int, choices=range(42, 47), default=42)
    p.add_argument("--root", type=Path, default=ROOT, help="Destination repository directory")
    p.add_argument("--revision", default="main", help="Hugging Face revision or commit")
    p.add_argument("--dry-run", action="store_true", help="List paths without connecting or downloading")
    args = p.parse_args(argv)
    try:
        plan = download_plan(args.resource, args.benchmark, args.seed)
    except ValueError as exc:
        p.error(str(exc))
    print(f"Hugging Face: https://huggingface.co/{REPO_ID}")
    for source, destination in plan:
        print(f"  {source}\n    -> {args.root / destination}")
    if args.dry_run:
        print("Dry run: no network access or downloads.")
        return 0
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
        from requests.exceptions import RequestException
    except ImportError:
        print('Install download support first: pip install -e ".[hub]"', file=sys.stderr)
        return 1
    try:
        revision = HfApi().model_info(REPO_ID, revision=args.revision).sha
        print(f"Revision: {revision}")
        for source, destination in plan:
            cached = hf_hub_download(REPO_ID, source, revision=revision)
            install_file(cached, args.root / destination)
        print("Done. Files are in the paths shown above.")
        return 0
    except (OSError, ValueError, HfHubHTTPError, RequestException) as exc:
        print(f"Download stopped: {exc}", file=sys.stderr)
        if isinstance(exc, HfHubHTTPError) and exc.response is not None and exc.response.status_code in (401, 403):
            print("Private repository: use `huggingface-cli login` with an account that has access.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
