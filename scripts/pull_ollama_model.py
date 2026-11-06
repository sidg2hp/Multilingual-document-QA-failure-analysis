"""Download an Ollama model directly from the registry with resumable writes.

Ollama's built-in puller can stall on slow/flaky links and does not expose a
simple resume path. This script fetches the registry manifest and blobs with
HTTP Range requests, writes them into an Ollama model store, and lets the local
server discover the model. It is a fallback, not the default installation path.

Usage:
    python scripts/pull_ollama_model.py --model moondream --models-dir .ollama/models
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, Optional

import requests

REGISTRY = "https://registry.ollama.ai"
CHUNK = 1024 * 1024


def split_model(model: str) -> tuple:
    name, _, tag = model.partition(":")
    tag = tag or "latest"
    if "/" not in name:
        name = f"library/{name}"
    return name, tag


def manifest_url(model: str) -> str:
    name, tag = split_model(model)
    return f"{REGISTRY}/v2/{name}/manifests/{tag}"


def blob_url(model: str, digest: str) -> str:
    name, _ = split_model(model)
    return f"{REGISTRY}/v2/{name}/blobs/{digest}"


def digest_hex(digest: str) -> str:
    return digest.split(":", 1)[1]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_blob(
    url: str,
    target: Path,
    expected_size: int,
    expected_digest: str,
    retries: int = 8,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        if sha256_file(target) == expected_digest:
            print(f"  exists {target.name} ({expected_size} bytes, hash OK)", flush=True)
            return
        print(f"  hash mismatch for {target.name}; redownloading", flush=True)
        target.unlink()
    if target.exists() and target.stat().st_size == 0:
        target.unlink()

    temp = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        existing = temp.stat().st_size if temp.exists() else 0
        if existing > expected_size:
            temp.unlink()
            existing = 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                if response.status_code == 416 and existing >= expected_size:
                    break
                response.raise_for_status()
                mode = "ab" if existing else "wb"
                with open(temp, mode) as handle:
                    downloaded = existing
                    last_report = time.time()
                    for chunk in response.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_report > 5:
                            print(
                                f"  {target.name}: {downloaded}/{expected_size} "
                                f"({100 * downloaded / max(1, expected_size):.1f}%)",
                                flush=True,
                            )
                            last_report = time.time()
            if temp.stat().st_size == expected_size:
                if sha256_file(temp) != expected_digest:
                    print(f"  checksum mismatch for {target.name}; retrying", flush=True)
                    temp.unlink()
                    continue
                temp.replace(target)
                print(f"  done {target.name} ({expected_size} bytes, hash OK)", flush=True)
                return
            print(
                f"  short read {target.name}: {temp.stat().st_size}/{expected_size}; retrying",
                flush=True,
            )
        except (requests.RequestException, OSError) as error:
            print(f"  attempt {attempt} failed for {target.name}: {error}", flush=True)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Could not download {target} after {retries} attempts")


def pull(model: str, models_dir: Path) -> None:
    response = requests.get(manifest_url(model), timeout=30)
    response.raise_for_status()
    manifest: Dict[str, object] = response.json()

    blobs_dir = models_dir / "blobs"
    name, tag = split_model(model)
    manifest_path = (
        models_dir / "manifests" / "registry.ollama.ai" / name / tag
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    config = manifest["config"]  # type: ignore[index]
    print(f"manifest for {model}: {len(manifest.get('layers', []))} layers")  # type: ignore[arg-type]
    download_blob(
        blob_url(model, config["digest"]),  # type: ignore[index]
        blobs_dir / f"sha256-{digest_hex(config['digest'])}",  # type: ignore[index]
        int(config["size"]),  # type: ignore[index]
        digest_hex(config["digest"]),  # type: ignore[index]
    )
    for layer in manifest["layers"]:  # type: ignore[index]
        download_blob(
            blob_url(model, layer["digest"]),
            blobs_dir / f"sha256-{digest_hex(layer['digest'])}",
            int(layer["size"]),
            digest_hex(layer["digest"]),
        )

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {manifest_path}")
    print("Run `ollama list` to confirm; the server may need a restart to rescan blobs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Ollama model download")
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-dir", default=".ollama/models")
    args = parser.parse_args()
    pull(args.model, Path(args.models_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
