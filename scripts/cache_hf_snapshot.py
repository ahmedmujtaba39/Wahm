"""Download and checksum one immutable Hugging Face model snapshot."""

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


def run(model_id, revision, cache_dir, output):
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=model_id, revision=revision, cache_dir=cache_dir))
    files = []
    for path in sorted(candidate for candidate in snapshot.rglob("*")
                       if candidate.is_file()):
        files.append({
            "path": path.relative_to(snapshot).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    audit = {
        "model_id": model_id,
        "revision": revision,
        "snapshot_dir": str(snapshot),
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"cached {model_id}@{revision}: {len(files)} files")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.model_id, args.revision, args.cache_dir, args.output)
