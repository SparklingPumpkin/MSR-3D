# Scene data format

```text
data/
  manifests/assets.json
  preprocessed/<dataset>/<scene>/
    scene.ply
    instances.npy
    config.yaml
    runtime.json
    images/
    sparse/
    rendered_instances/instance_XXXX/<frame>.png
  encoders/
    clip/ViT-L-14-336px.pt
    masqclip/base_novel.pth
  precomputed/<dataset>/<scene>/
    pool/
    representations/
  evaluation/cached-matcher-reference/
    queries.json
    annotations/
    ground_truth/
    instance_masks/
```

`dataset` is `lerf67` or `3dovs61`. Each PLY vertex has a corresponding integer instance label in `instances.npy`; negative labels denote background. Camera intrinsics/extrinsics use COLMAP binary format under `sparse/`. RGB filenames match the cameras. RGBA instance renders use the same frame names; the alpha channel supplies their masks.

`config.yaml` specifies Gaussian SH degree, image resolution and crop settings. `runtime.json` supplies the training-camera list, visibility thresholds and S1 adapter. With data v1.1.0, both datasets use `s1_adapter="geometric_topk"` and `anchor_topk=5`. LERF retains `anchor_crop_selection="all_three_levels"`; 3DOVS changes from shared-pool-mean anchors to `anchor_crop_selection="middle"`. All three crops per selected view are encoded together before averaging either all crops (LERF) or only the middle crop (3DOVS). Observation features use the middle crop. The `sam` source name denotes the original RGB masked by rendered instance visibility; segmentation/reconstruction has already been performed.

Anchor views are ranked among training cameras by `(visible Gaussian count / instance Gaussian count) * (mask area / maximum mask area)`. Selection takes the first five ranked views (or all available views if fewer than five), without dropping or replacing zero-area candidates. If every training-view mask is empty, the anchor is a zero vector with `valid=false`. Otherwise, any selected zero-area views still enter crop encoding and averaging; the crop helper uses the empty mask's zero bounding box. This fixed top-five convention is distinct from selecting up to five nonempty masks. `anchors.npz` records `instance_ids`, `embeddings` and `valid`; query construction filters invalid anchors and normalizes the retained embeddings for cosine similarity. Its instance inventory may differ from the observation/prototype inventory.

The cold-start entry point reads scene assets and encoder weights, then generates `pool/observations.npz`, `pool/anchors.npz`, `pool/queries.npz`, and `representations/representation.npz` for both datasets. The pool stores all retained `(instance, source, frame)` observations as 768-dimensional vectors. Each representation stores up to six prototypes, their validity/attributes, instance attributes and the shared mean. Geometry measurements are retained in pool metadata; prototype construction preserves the released absent-statistics defaults.

The evaluation directory supplies query membership, physical-instance positive sets, processed GT masks and processed instance-mask banks. Query features and historical reference outputs under that directory are optional comparison assets and are not read by the full-pipeline entry point. `--pool` explicitly selects the shortcut that reuses an existing pool.

Per-query IoU averages annotated frames. Scene-macro metrics first average queries inside each scene, then weight scenes equally. Query-macro metrics are also saved. F1 compares physical-instance ID sets; it is distinct from pixel F1. Missing processed instance masks contribute an empty mask and are enumerated in per-frame results. See `evaluation/cached-matcher-reference/PROTOCOL.md` for the mask-processing convention and cohort membership.

Before GPU work, the input validator verifies SHA-256 and size for every scene asset and encoder checkpoint listed in `manifests/assets.json`. Sparse render views are defined by that inventory; a missing listed file is an error. The manifest records each file as `{ "path": "relative/path", "size": 123, "sha256": "..." }` in its `files` array.

`precomputed/` contains independently rebuilt pools and portable representations in the same formats as a cold run. They provide convenient starting points for downstream use; default cold-start generation reads only scene assets and encoder weights. The [Hugging Face data release](https://huggingface.co/datasets/KaiLai/MSR-3D-Data) provides `tools/prepare.py` to verify archive checksums, attach separately obtained upstream inputs and generate `manifests/assets.json` locally.
