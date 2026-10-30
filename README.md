# MSR-3D

Minimal fixed-weight inference for **Multi-Mode Semantic Representation for Open-Vocabulary 3D Scene Understanding**.

This release contains the full matcher's inference code and learned parameters. It accepts precomputed query–prototype cosine similarities, prototype/instance attributes, and anchor similarities, and returns the selected instance identifiers. Reconstruction, segmentation, text/image encoding, observation-pool construction, prototype extraction, mask rendering and benchmark evaluation are outside this package. It does not turn a text string and a raw scene directly into a prediction.

## Install

Python 3.11 or newer on Linux. The reference numerical environment is Python 3.11, PyTorch 2.5.1, NumPy 2.0.2 and SciPy 1.16.2. Inference runs on CPU and needs no model downloads after installation.

```bash
python -m venv .venv
.venv/bin/python -m pip install .
```

The model parameters ship inside the installed `msr3d` package; the working directory does not need to contain a separate weights folder. NumPy archives load with `allow_pickle=False`. The archive checksum is checked at model construction.

## Run

```bash
mkdir -p outputs
.venv/bin/python -m msr3d example --output outputs/example.npz
.venv/bin/python -m msr3d predict --input outputs/example.npz --output outputs/prediction.json
```

The example is synthetic and demonstrates the interface; it is not a benchmark sample or an accuracy test. Both commands refuse to overwrite existing output files.

For real inputs, prepare the feature arrays specified in [INPUT_FORMAT.md](INPUT_FORMAT.md). The prototype ordering, source ordering, feature definitions and normalizations must match that specification.

```python
from msr3d import Matcher, load_query

model = Matcher()
result = model.predict(load_query("query_features.npz"))
print(result["selected_ids"])
```

`QueryFeatures` also accepts arrays directly. A `Matcher` instance reuses loaded parameters across queries. Predictions contain the instance-aligned final scores, S3 corrections, base margins and corrected margins, together with the base and final instance sets. Scores are decoder outputs, not calibrated relevance probabilities.

## Model

The first semantic channel normalizes the full anchor bank. The second scores prototype rows, aggregates them with instance statistics and estimates a query threshold. A primary residual MLP and a supplementary two-score MLP form the base set. Attribute-conditioned prototype attention and similarity summaries produce the bounded S3 correction. The decoder protects every base selection and adds at most one unselected instance whose corrected margin is positive; ties are resolved by instance identifier.

The released model uses source-scene supervision. This package provides inference, not training. Its numerical outputs were checked against the reference implementation on 128 cached queries: selected instance sets, instance scores, S3 outputs and continuous margins matched exactly on the reference CPU environment. Bitwise identity across different numerical environments is not guaranteed. This consistency check is not evidence that each component improves every benchmark.

The package does not redistribute datasets, pretrained feature encoders, reconstruction systems or segmentation systems. NumPy, SciPy and PyTorch are external dependencies and retain their respective licenses. No license for any external dataset or upstream asset is granted by this repository.

## Check

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

[MODEL_CARD.md](MODEL_CARD.md) describes the release scope and limitations.
