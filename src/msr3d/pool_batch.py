"""Full-scene pool reconstruction with exact multiplexed visibility rendering."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
import yaml
from .frontend.cameras import load_cameras
from .frontend.gaussian import create_from_ply
from .frontend.render import render
from .frontend.encoder import Encoder
from .frontend.crops import CropParams, masks_to_crops
from .pool import read_rgba
from .inputs import validate_scene_assets


@torch.inference_mode()
def build_scene(
    data_root,
    dataset,
    scene,
    output,
    instances=None,
    batch_size=8,
    mask_batch=24,
    reference_precision=False,
):
    root, output = Path(data_root), Path(output)
    validated_assets = validate_scene_assets(root, dataset, scene)
    output.mkdir(parents=True, exist_ok=False)
    if reference_precision:
        batch_size = 1
    start = time.monotonic()
    scene_root = root / "preprocessed" / dataset / scene
    config = yaml.safe_load((scene_root / "config.yaml").read_text())
    contract = json.loads((scene_root / "runtime.json").read_text())
    device = torch.device("cuda")
    model = create_from_ply(
        scene_root / "scene.ply",
        config["model"]["sh_degree"],
        device,
        feature_dim=config["model"]["mask_dim"],
    )
    labels = (
        torch.from_numpy(np.load(scene_root / "instances.npy", allow_pickle=False))
        .view(-1)
        .to(device)
    )
    if len(labels) != model.num_points:
        raise ValueError("Instance labels/PLY mismatch")
    ids = sorted(int(i) for i in labels.unique() if i >= 0)
    if instances is not None:
        ids = [i for i in ids if f"instance_{i:04d}" in instances]
    if not ids:
        raise ValueError("Empty requested physical instance inventory")
    cameras = load_cameras(scene_root, device, config["model"]["resolution"])
    encoder = Encoder(root / "encoders")
    background = 1.0 if config["model"]["white_background"] else 0.0
    bg = torch.full((3,), background, device=device)
    pipe = SimpleNamespace(**config["pipe"])
    zeros = torch.zeros((model.num_points, 3), device=device)
    counts = {label: int((labels == label).sum()) for label in ids}
    camera_rgb = {
        c.name: c.original_image.detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .contiguous()
        for c in cameras
    }
    camera_by_name = {c.name: c for c in cameras}
    train = set(contract["train_frames"])
    all_rows, all_features, all_anchors, anchor_ids, anchor_valid, anchor_meta = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    crops, masks, pending = [], [], []

    def flush():
        if not pending:
            return
        torch.backends.cudnn.allow_tf32 = reference_precision
        values = encoder.images(crops, masks)
        for row, value in zip(pending, values):
            all_rows.append(row)
            all_features.append(value.numpy())
        pending.clear()
        crops.clear()
        masks.clear()

    def crop_one(item):
        row, image, mask = item
        return row, *encoder.crop(
            image, mask, config["lang"]["ratio"], config["lang"]["dynamic_ratio"]
        )

    with ThreadPoolExecutor(max_workers=6) as workers:
        for offset in range(0, len(ids), mask_batch):
            group = ids[offset : offset + mask_batch]
            group_tensor = torch.tensor(group, device=device)
            features = (labels[:, None] == group_tensor[None, :]).float()
            geometry = {label: [] for label in group}
            for camera in cameras:
                package = render(
                    camera,
                    model,
                    pipe,
                    bg,
                    config["model"]["sh_degree"],
                    override_color=zeros,
                    override_features=features,
                    render_color=False,
                    render_features=True,
                )
                # RGB rendering repeats the same coverage in three channels and averages it.
                values = package.features.clamp(0, 1)
                coverage = (values + values + values) / 3.0
                visible_masks = (coverage > contract["pool_visibility_threshold"]).cpu()
                anchor_masks = (
                    coverage > contract["anchor_visibility_threshold"]
                ).cpu()
                visible_labels = labels[package.visibility_filter]
                visible_counts = torch.bincount(
                    visible_labels[visible_labels >= 0], minlength=int(labels.max()) + 1
                ).cpu()
                todo = []
                rgb = camera_rgb[camera.name]
                for index, label in enumerate(group):
                    iid = f"instance_{label:04d}"
                    visible = visible_masks[index]
                    nvisible = int(visible_counts[label])
                    npoints = counts[label]
                    if camera.name in train:
                        amask = anchor_masks[index]
                        geometry[label].append(
                            {
                                "frame": camera.name,
                                "area": int(amask.sum()),
                                "visible_count": nvisible,
                                "label_count": npoints,
                                "mask": amask,
                            }
                        )
                    base = {
                        "instance_id": iid,
                        "frame": camera.name,
                        "visible_count": nvisible,
                        "label_count": npoints,
                        "height": int(visible.shape[0]),
                        "width": int(visible.shape[1]),
                    }
                    if visible.any():
                        todo.append(
                            (
                                {
                                    **base,
                                    "source": "sam",
                                    "mask_area": int(visible.sum()),
                                },
                                rgb,
                                visible,
                            )
                        )
                    rgba = read_rgba(
                        scene_root / "rendered_instances" / iid / (camera.name + ".png")
                    )
                    if rgba is not None:
                        rendered_rgb, rendered_mask = rgba
                        if rendered_mask.any():
                            composited = rgb.clone()
                            composited[:, rendered_mask] = rendered_rgb[
                                :, rendered_mask
                            ]
                            todo.append(
                                (
                                    {
                                        **base,
                                        "source": "render",
                                        "mask_area": int(rendered_mask.sum()),
                                    },
                                    composited.contiguous(),
                                    rendered_mask,
                                )
                            )
                for row, crop, mask in workers.map(crop_one, todo):
                    pending.append(row)
                    crops.append(crop)
                    masks.append(mask)
                    if len(pending) >= batch_size:
                        flush()
            flush()
            if contract["s1_adapter"] == "geometric_topk":
                for label in group:
                    stats = geometry[label]
                    maximum = max(x["area"] for x in stats)
                    anchor_ids.append(f"instance_{label:04d}")
                    if maximum == 0:
                        all_anchors.append(np.zeros(768, dtype=np.float32))
                        anchor_valid.append(False)
                        anchor_meta.append(
                            {"instance_id": anchor_ids[-1], "frames": []}
                        )
                        continue
                    chosen = sorted(
                        stats,
                        key=lambda x: (x["visible_count"] / x["label_count"])
                        * (x["area"] / maximum),
                        reverse=True,
                    )[: contract["anchor_topk"]]
                    acrops, amasks = [], []
                    for stat in chosen:
                        camera = camera_by_name[stat["frame"]]
                        image = (
                            camera.original_image.detach()
                            .cpu()
                            .clamp(0, 1)
                            .mul(255)
                            .to(torch.uint8)
                            .contiguous()
                        )
                        ratio = (
                            config["lang"]["ratio"]
                            if not config["lang"]["dynamic_ratio"]
                            or stat["area"] / stat["mask"].numel() < 0.0075
                            else 0.1
                        )
                        params = CropParams(expansion_ratio=ratio)
                        images, cmasks, _, _ = masks_to_crops(
                            image, stat["mask"].unsqueeze(0), params, encoder
                        )
                        acrops.extend(images[0])
                        amasks.extend(cmasks[0].unsqueeze(1))
                    # Encode all three crops together; select crops only before averaging.
                    torch.backends.cudnn.allow_tf32 = True
                    embeddings = encoder.images(acrops, amasks)
                    if contract["anchor_crop_selection"] == "middle":
                        embeddings = embeddings.view(-1, 3, 768)[:, 1]
                    all_anchors.append(embeddings.mean(dim=0).numpy())
                    anchor_valid.append(True)
                    anchor_meta.append(
                        {
                            "instance_id": anchor_ids[-1],
                            "frames": [s["frame"] for s in chosen],
                        }
                    )
            print(
                "GROUP_DONE",
                dataset,
                scene,
                offset + len(group),
                "/",
                len(ids),
                "observations",
                len(all_rows),
                "seconds",
                round(time.monotonic() - start, 2),
                flush=True,
            )
            del geometry
    if not all_rows:
        raise ValueError("No pool observations generated")
    with (output / "observations.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            features=np.stack(all_features),
            instance_ids=np.asarray([r["instance_id"] for r in all_rows]),
            frame_names=np.asarray([r["frame"] for r in all_rows]),
            sources=np.asarray([r["source"] for r in all_rows]),
        )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "scene": scene,
                "observations": all_rows,
                "physical_ids": [f"instance_{i:04d}" for i in ids],
                "validated_assets": validated_assets,
                "geometry": anchor_meta,
                "read_observation_or_representation_references": False,
                "pool_encoding": "historical singleton, cuDNN TF32 enabled"
                if reference_precision
                else "batch FP32, cuDNN TF32 disabled",
                "anchor_encoding": (
                    "all-three-crop batch, cuDNN TF32 enabled; "
                    + ("middle crop mean" if contract["anchor_crop_selection"] == "middle"
                       else "all-three-crop mean")
                ) if contract["s1_adapter"] == "geometric_topk" else "shared pool mean",
                "seconds": time.monotonic() - start,
            },
            indent=2,
        )
        + "\n"
    )
    if all_anchors:
        with (output / "anchors.npz").open("xb") as handle:
            np.savez_compressed(
                handle,
                instance_ids=np.asarray(anchor_ids),
                embeddings=np.stack(all_anchors),
                valid=np.asarray(anchor_valid),
            )
    queries = [
        q
        for q in json.loads(
            (root / "evaluation/cached-matcher-reference/queries.json").read_text()
        )
        if q["dataset"] == dataset and q["scene"] == scene
    ]
    torch.backends.cudnn.allow_tf32 = True
    text = encoder.text([q["query_text"] for q in queries]).numpy()
    with (output / "queries.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            query_ids=np.asarray([q["query_id"] for q in queries]),
            query_texts=np.asarray([q["query_text"] for q in queries]),
            features=text,
        )
    print(
        "POOL_DONE",
        dataset,
        scene,
        len(all_rows),
        "seconds",
        round(time.monotonic() - start, 2),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset", choices=("lerf67", "3dovs61"), required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--instances", nargs="+")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--reference-precision",
        action="store_true",
        help="Use the historical singleton image-encoding call and cuDNN precision",
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    build_scene(
        a.data_root,
        a.dataset,
        a.scene,
        a.output,
        a.instances,
        a.batch_size,
        reference_precision=a.reference_precision,
    )


if __name__ == "__main__":
    main()
