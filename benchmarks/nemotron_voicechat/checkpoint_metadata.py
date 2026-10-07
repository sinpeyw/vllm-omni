# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Read checkpoint provenance locally without hashing or downloading weights."""

import hashlib
from pathlib import Path


def checkpoint_identity(model_dir: Path) -> dict[str, object]:
    """Record config content and available Hub download metadata for each file.

    Download revision/ETag identify the recorded source, not a fresh content
    check of the large weight file. A copied checkpoint may lack this metadata;
    config SHA256 and file sizes are still recorded in that case.
    """
    files = {}
    for name in ("config.json", "model.safetensors"):
        path = model_dir / name
        identity: dict[str, object] = {"size_bytes": path.stat().st_size}
        if name == "config.json":
            identity["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = model_dir / ".cache/huggingface/download" / f"{name}.metadata"
        if metadata.is_file():
            with metadata.open() as stream:
                revision = stream.readline().strip()
                etag = stream.readline().strip()
            if not revision or not etag:
                raise ValueError(f"Incomplete Hub download metadata: {metadata}")
            identity.update(hub_revision=revision, hub_etag=etag)
        files[name] = identity
    return {"files": files, "weight_content_hashed": False}
