#!/usr/bin/env python3
"""Resume a Hugging Face model repository without rechecking local files.

This is useful for large archives with thousands of small files: the Hub
client puts a downloaded file at its final path only after completion.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("local_dir")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def is_rate_limit(error):
    while error is not None:
        if isinstance(error, HfHubHTTPError):
            if error.response is not None and error.response.status_code == 429:
                return True
        error = error.__cause__
    return False


def download_one(repo_id, local_dir, filename, retries):
    attempt = 0
    while attempt < retries:
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=filename,
                local_dir=local_dir,
            )
            return filename, None
        except Exception as error:
            if is_rate_limit(error):
                print("Hub resolver quota reached; waiting five minutes.", flush=True)
                time.sleep(310)
                continue
            attempt += 1
            if attempt == retries:
                return filename, str(error)
            time.sleep(attempt * 5)


def main():
    args = parse_args()
    files = HfApi().list_repo_files(args.repo_id, repo_type="model")
    local_dir = Path(args.local_dir)
    missing = [name for name in files if not (local_dir / name).is_file()]
    print(f"Repository: {len(files)} files; {len(missing)} still missing.", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            lambda name: download_one(args.repo_id, args.local_dir, name, args.retries),
            missing,
        )
        for index, (filename, error) in enumerate(results, start=1):
            if error is not None:
                failures.append((filename, error))
                print(f"FAILED {filename}: {error}", flush=True)
            if index % 100 == 0 or index == len(missing):
                print(f"Processed {index}/{len(missing)} missing files", flush=True)
    if failures:
        print(f"{len(failures)} files failed; rerun this command to resume.", file=sys.stderr)
        return 1
    print(f"Completed {len(files)} files.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
