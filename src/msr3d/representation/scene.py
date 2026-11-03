"""Portable query-independent representations built from freshly generated pools."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .construction import build_shared_instance_bundle
from .settings import FusionConfig
from .types import FeatureSourcePool, InstanceFeaturePool
from ..features import QueryFeatures

SOURCE_NAMES = {"sam": "sam-mask-masqclip", "render": "render-mask-masqclip"}


def construct_scene(pool_path, output, frame_suffix=".pt"):
    pool_path, output = Path(pool_path), Path(output)
    with np.load(pool_path, allow_pickle=False) as archive:
        features = archive["features"]
        instance_ids = archive["instance_ids"]
        frames = archive["frame_names"]
        sources = archive["sources"]
    keys = list(zip(instance_ids, sources, frames))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate observation identity")
    if features.shape != (len(keys), 768) or not np.isfinite(features).all():
        raise ValueError("Invalid observation feature matrix")
    if not set(sources) <= set(SOURCE_NAMES):
        raise ValueError("Unknown observation source")
    ids = sorted(set(instance_ids.tolist()))
    if not ids:
        raise ValueError("Empty observation inventory")
    output.mkdir(parents=True, exist_ok=False)
    n = len(ids)
    prototypes = np.zeros((n, 6, 768), dtype=np.float32)
    valid = np.zeros((n, 6), dtype=bool)
    attributes = np.zeros((n, 6, 7), dtype=np.float64)
    instance_attributes = np.zeros((n, 4), dtype=np.float64)
    means = np.zeros((n, 768), dtype=np.float32)
    prototype_frames = np.full((n, 6), "", dtype="<U160")
    records = []
    for index, iid in enumerate(ids):
        source_pools = {}
        for source, name in SOURCE_NAMES.items():
            rows = np.flatnonzero((instance_ids == iid) & (sources == source))
            if not len(rows):
                continue
            frame_features = {
                str(frames[j]) + frame_suffix: torch.from_numpy(features[j].copy())
                for j in rows
            }
            source_pools[name] = FeatureSourcePool(
                source_name=name,
                frame_features=frame_features,
                metadata={
                    "source_type": "mask" if source == "sam" else "render",
                    "layout": "direct_instance_dir",
                },
            )
        bundle = build_shared_instance_bundle(
            InstanceFeaturePool(str(iid), source_pools), FusionConfig()
        )
        proto = bundle.prototype_set
        count = len(proto.prototype_indices)
        if count > 6:
            raise ValueError("Prototype budget exceeded")
        prototypes[index, :count] = proto.prototype_features.numpy()
        valid[index, :count] = True
        means[index] = bundle.mean.mean_feature.numpy()
        prototype_frames[index, :count] = proto.prototype_frame_names
        for j, meta in enumerate(proto.prototype_meta):
            attributes[index, j] = [
                meta.support_ratio,
                meta.cross_source_support_ratio,
                meta.proto_member_sim_mean,
                meta.proto_separation,
                meta.geom_bad_fraction,
                meta.source_comp_map.get(SOURCE_NAMES["sam"], 0.0),
                meta.source_comp_map.get(SOURCE_NAMES["render"], 0.0),
            ]
        meta = bundle.instance_meta
        instance_attributes[index] = [
            meta.proto_count,
            meta.top1_support_ratio,
            meta.cross_source_agreement_score,
            meta.bad_geom_proto_ratio,
        ]
        records.append(
            {
                "instance_id": str(iid),
                "prototype_indices": proto.prototype_indices,
                "prototype_frame_names": proto.prototype_frame_names,
                "prototype_meta": [asdict(x) for x in proto.prototype_meta],
                "instance_meta": asdict(bundle.instance_meta),
                "prototype_stats": proto.prototype_stats,
                "source_frames": {
                    name: sorted(pool.frame_features)
                    for name, pool in source_pools.items()
                },
            }
        )
    with (output / "representation.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            instance_ids=np.asarray(ids),
            prototype_features=prototypes,
            prototype_valid=valid,
            prototype_attributes=attributes,
            instance_attributes=instance_attributes,
            shared_means=means,
            prototype_frame_names=prototype_frames,
        )
    with pool_path.open("rb") as handle:
        pool_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "pool_sha256": pool_hash,
                "frame_suffix": frame_suffix,
                "prototype_config": asdict(FusionConfig()),
                "geometry_policy": "Historical absent-statistics defaults; raw rendering geometry stays in pool metadata",
                "read_representation_references": False,
                "instances": records,
            },
            indent=2,
        )
        + "\n"
    )


def make_query_features(
    representation_root, query_feature, s1_adapter, anchors_path=None
):
    root = Path(representation_root)
    with np.load(root / "representation.npz", allow_pickle=False) as archive:
        ids = archive["instance_ids"].tolist()
        prototypes = archive["prototype_features"].copy()
        valid = archive["prototype_valid"].copy()
        attrs = archive["prototype_attributes"].copy()
        instance_attrs = archive["instance_attributes"].copy()
        means = archive["shared_means"].copy()
    query = F.normalize(
        torch.as_tensor(query_feature).detach().cpu().float().view(1, -1), dim=1
    ).squeeze(0)
    matrix = np.zeros((len(ids), 6, 13), dtype=np.float32)
    for i in range(len(ids)):
        indices = np.flatnonzero(valid[i])
        values = (
            (torch.from_numpy(prototypes[i, indices]) @ query.view(-1, 1))
            .view(-1)
            .tolist()
        )
        for j, value in zip(indices, values):
            matrix[i, j] = [
                value,
                *attrs[i, j].tolist(),
                *instance_attrs[i].tolist(),
                1.0,
            ]
    if s1_adapter == "shared_pool_mean":
        anchor_ids = list(ids)
        bank = F.normalize(torch.from_numpy(means), dim=1)
    elif s1_adapter == "geometric_topk":
        if anchors_path is None:
            raise ValueError("Geometric anchor bank is required")
        with np.load(anchors_path, allow_pickle=False) as archive:
            keep = archive["valid"].astype(bool)
            anchor_ids = archive["instance_ids"][keep].tolist()
            bank = F.normalize(
                torch.from_numpy(archive["embeddings"][keep].copy()), dim=1
            )
    else:
        raise ValueError("Unknown S1 adapter")
    anchor_raw = (bank @ query.view(-1, 1)).view(-1).numpy()
    return QueryFeatures(ids, matrix, valid, anchor_ids, anchor_raw).validated()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(16)
    construct_scene(args.pool, args.output)


if __name__ == "__main__":
    main()
