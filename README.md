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

## Check

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

See [MODEL_CARD.md](MODEL_CARD.md) for model details.
