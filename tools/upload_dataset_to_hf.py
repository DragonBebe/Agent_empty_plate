#!/usr/bin/env python3
"""
Upload a local dataset directory (containing data/ and meta/) to Hugging Face Hub as a dataset repo.

Usage example:
  HF_TOKEN=xxxx python tools/upload_dataset_to_hf.py \
    --repo-id your-username/robot-remove-red-block-v1 \
    --local-dir /home/dragon/empty_plat_train_deploy/demo_data_language \
    --private \
    --message "Initial upload"
    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

Notes:
  - Non-interactive. Reads token from env HF_TOKEN (or ~/.huggingface/token if already logged in).
  - Preserves your folder structure (data/chunk-*/ and meta/*).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder, upload_folder


def ensure_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        # Persist token for subsequent calls
        HfFolder.save_token(token)
    else:
        # If user has previously logged in via CLI, token may already exist
        token = HfFolder.get_token()
    return token


def validate_local_dir(local_dir: Path) -> None:
    if not local_dir.exists() or not local_dir.is_dir():
        print(f"[ERROR] Local dataset directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)
    # Soft checks for expected structure
    data_dir = local_dir / "data"
    meta_dir = local_dir / "meta"
    if not data_dir.exists():
        print(f"[WARN] {data_dir} not found. Continuing anyway.")
    if not meta_dir.exists():
        print(f"[WARN] {meta_dir} not found. Continuing anyway.")


def create_repo_if_needed(api: HfApi, repo_id: str, private: bool) -> None:
    try:
        # Will raise if not exists
        api.dataset_info(repo_id)
        print(f"[INFO] Repo exists: {repo_id}")
    except Exception:
        print(f"[INFO] Creating dataset repo: {repo_id} (private={private})")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)


def main():
    parser = argparse.ArgumentParser(description="Upload local dataset directory to HF Hub (dataset repo)")
    parser.add_argument("--repo-id", required=True, help="e.g., your-username/robot-remove-red-block-v1")
    parser.add_argument("--local-dir", required=True, help="Local dataset folder containing data/ and meta/")
    parser.add_argument("--private", action="store_true", help="Create/keep repo as private")
    parser.add_argument("--message", default="Upload dataset", help="Commit message")
    args = parser.parse_args()

    token = ensure_token()
    if not token:
        print("[ERROR] No HF token found. Set HF_TOKEN env or run: python -c \"from huggingface_hub import login; login()\"", file=sys.stderr)
        sys.exit(1)

    local_dir = Path(args.local_dir).expanduser().resolve()
    validate_local_dir(local_dir)

    api = HfApi()
    create_repo_if_needed(api, args.repo_id, args.private)

    print(f"[INFO] Uploading folder: {local_dir} -> {args.repo_id}")
    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=".",
        commit_message=args.message,
        token=token,
    )
    print(f"[DONE] Pushed dataset to: {args.repo_id}")


if __name__ == "__main__":
    main()


