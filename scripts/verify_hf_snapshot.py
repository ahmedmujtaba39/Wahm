"""Verify every file recorded by cache_hf_snapshot.py."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(audit_path, model_id=None, revision=None):
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if model_id and audit["model_id"] != model_id:
        raise ValueError("cache audit model_id does not match")
    if revision and audit["revision"] != revision:
        raise ValueError("cache audit revision does not match")
    snapshot = Path(audit["snapshot_dir"])
    for row in audit["files"]:
        path = snapshot / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"size mismatch: {path}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
    print(f"verified {len(audit['files'])} files for "
          f"{audit['model_id']}@{audit['revision']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    args = parser.parse_args()
    run(args.audit, args.model_id, args.revision)
