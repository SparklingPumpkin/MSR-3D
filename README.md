# MSR-3D

Inference code and pretrained weights for **Multi-Mode Semantic Representation for Open-Vocabulary 3D Scene Understanding**.

The matcher takes precomputed query–prototype similarities, prototype/instance attributes, and anchor similarities, and returns matching instance IDs.

## Install

Requires Python 3.11+ on Linux. Model weights are bundled with the package; inference runs on CPU.

```bash
python -m venv .venv
.venv/bin/python -m pip install .
```

## Run

Try a synthetic example:

```bash
mkdir -p outputs
.venv/bin/python -m msr3d example --output outputs/example.npz
.venv/bin/python -m msr3d predict --input outputs/example.npz --output outputs/prediction.json
```

For your own inputs, prepare an NPZ file following [INPUT_FORMAT.md](INPUT_FORMAT.md), then use the CLI or Python API:

```python
from msr3d import Matcher, load_query

model = Matcher()
result = model.predict(load_query("query_features.npz"))
print(result["selected_ids"])
```

## Data

The **MSR-3D benchmark features v1.0.0** bundle provides LERF 67-query inputs (including the 45-query subset), 3DOVS 61-query inputs, and processed evaluation masks. Extract it into `data/`, then run:

```bash
.venv/bin/python -m msr3d.benchmark \
  --data-root data/msr3d-benchmark-v1.0.0 \
  --output outputs/benchmark --check-reference
```

Use `--split lerf45`, `--split lerf67`, or `--split 3dovs61` to evaluate one cohort. Results include per-query and per-frame records, scene averages, and query averages; scene averages give each scene equal weight. mF1 measures physical-instance set retrieval. See the bundle's `PROTOCOL.md` for definitions.

## Check

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

See [MODEL_CARD.md](MODEL_CARD.md) for model details.
