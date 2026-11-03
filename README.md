# MSR-3D

Inference code and pretrained matching weights for **Multi-Mode Semantic Representation for Open-Vocabulary 3D Scene Understanding**.

The pipeline builds multi-view observation pools from reconstructed and segmented scenes, constructs per-instance MSR-3D representations, and matches text queries to instances.

## Install

Python 3.11, an NVIDIA GPU and a CUDA toolkit compatible with PyTorch are required.

```bash
git clone https://github.com/SparklingPumpkin/MSR-3D.git
cd MSR-3D
python -m venv .venv
source .venv/bin/activate
pip install '.[frontend]'
# Select the CUDA toolkit matching torch.version.cuda (12.4 for the tested setup).
export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation ./third_party/diff-gaussian-rasterization
```

Matching weights are bundled with the package. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream components.

## Data

Download the preprocessed scenes and optional observation pools/representations from [KaiLai/MSR-3D-Data on Hugging Face](https://huggingface.co/datasets/KaiLai/MSR-3D-Data). The dataset README links the official LERF images/cameras and CLIP/MasQCLIP checkpoints to obtain separately.

```bash
pip install huggingface_hub
hf download KaiLai/MSR-3D-Data --repo-type dataset --local-dir ../MSR-3D-Data
python ../MSR-3D-Data/tools/prepare.py \
  --output ../msr3d-data \
  --lerf-root /path/to/lerf_ovs \
  --clip-weight /path/to/ViT-L-14-336px.pt \
  --masqclip-weight /path/to/base_novel.pth
```

The data covers four LERF scenes (67 queries, including the 45-query exact-string-unseen subset) and ten 3DOVS scenes (61 scene/query pairs). See [DATA_FORMAT.md](DATA_FORMAT.md) for the input layout.

## Run

```bash
python -m msr3d.pipeline \
  --data-root ../msr3d-data --dataset lerf67 --scene figurines \
  --output outputs/figurines
```

Use `--dataset 3dovs61 --scene sofa` for a 3DOVS scene. Each run writes its reconstructed pool, representations, query features, predictions and detailed evaluation results to a new output directory.

To reuse the supplied observation pool, add `--pool ../msr3d-data/precomputed/lerf67/figurines/pool`. Prototype construction uses 16 CPU threads and matching uses one thread; retain these defaults for numerical comparisons.

For cached-feature matching and a quick CPU benchmark:

```bash
python -m msr3d.benchmark \
  --data-root ../msr3d-data/evaluation/cached-matcher-reference \
  --output outputs/benchmark --check-reference
```

Use `--split lerf45`, `--split lerf67`, or `--split 3dovs61` for one cohort. Results include per-query/per-frame records and both scene and query averages. Scene averages weight each scene equally; mF1 measures physical-instance set retrieval.

## Check

```bash
python -m unittest discover -s tests -v
```
