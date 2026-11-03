"""Verify distributed scene inputs before starting GPU work."""

import hashlib
import json
from pathlib import Path


def validate_scene_assets(root, dataset, scene):
    root = Path(root).resolve()
    if (
        dataset not in ("lerf67", "3dovs61")
        or not scene
        or Path(scene).name != scene
        or scene in (".", "..")
    ):
        raise ValueError("Invalid dataset or scene identifier")
    directory = root / "preprocessed" / dataset / scene
    for name in ("scene.ply", "instances.npy", "config.yaml", "runtime.json"):
        if not (directory / name).is_file():
            raise ValueError("Missing scene input: " + name)
    for name in ("images", "sparse", "rendered_instances"):
        if not (directory / name).is_dir():
            raise ValueError("Missing scene input directory: " + name)
    manifest = root / "manifests/assets.json"
    if not manifest.is_file():
        raise ValueError("Missing scene asset manifest: manifests/assets.json")
    records = json.loads(manifest.read_text())["files"]
    prefix = f"preprocessed/{dataset}/{scene}/"
    records = [
        r
        for r in records
        if r["path"].startswith(prefix) or r["path"].startswith("encoders/")
    ]
    if not any(r["path"].startswith(prefix + "rendered_instances/") for r in records):
        raise ValueError("Manifest has no rendered_instances source inventory")
    total = 0
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Scene asset path escapes data root")
        if not path.is_file():
            raise ValueError("Missing scene asset: " + record["path"])
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != record["size"] or digest != record["sha256"]:
            raise ValueError("Scene asset checksum mismatch: " + record["path"])
        total += record["size"]
    return {
        "files": len(records),
        "bytes": total,
        "checksums": "sha256",
        "dataset": dataset,
        "scene": scene,
    }
