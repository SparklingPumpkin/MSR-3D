# Cached query features

One NPZ file represents one query evaluated against a scene's complete candidate inventory. Use `np.savez_compressed` with exactly the five fields below. Arrays must not use object dtype; pickle is never enabled.

| Field | Shape | Type | Meaning |
|---|---|---|---|
| `instance_ids` | `(N,)` | Unicode strings | Unique physical instance IDs, aligned with the first axis of `matrix` |
| `matrix` | `(N, 6, 13)` | float32 | Query-conditioned prototype evidence, up to six slots per instance |
| `valid` | `(N, 6)` | bool or binary numeric | Valid prototype slots; every instance needs at least one |
| `anchor_instance_ids` | `(M,)` | Unicode strings | Full anchor feature-bank inventory, including IDs outside the matrix inventory |
| `anchor_similarities` | `(M,)` | float32 | Raw query/anchor cosine similarities before query-level normalization |

N and M must be positive. IDs must be nonempty and unique within each bank, without path separators or control characters. All numeric features must be finite. Instance IDs absent from the anchor bank receive a zero normalized anchor response; do not truncate the full anchor bank to the common-ID subset before normalization. The constant-bank normalization retains its deterministic first-argmax convention.

## Matrix columns

The prototype scorer uses columns 0–7. The instance aggregator uses columns 8–11 from the first valid prototype row. The S3 branch uses similarity statistics and the documented attribute subset across valid rows. Column12 is reserved for the homogeneous bias coordinate and is not read by this checkpoint.

| Column | Feature | Convention |
|---:|---|---|
| 0 | query/prototype cosine similarity | Uncalibrated cosine in the shared embedding space |
| 1 | prototype support ratio | Fraction of raw observations assigned to the prototype |
| 2 | cross-source support ratio | Number of sources represented among assigned observations, divided by two; zero for an empty assignment |
| 3 | mean prototype/member similarity | Mean cosine to the supporting observations |
| 4 | prototype separation | Feature-space separation statistic |
| 5 | poor-geometry observation fraction | Prototype geometry-quality statistic |
| 6 | 2D segmentation source proportion | Source composition of the prototype |
| 7 | rendered 3D segmentation source proportion | Source composition of the prototype |
| 8 | prototype count | Number of retained prototypes for this instance |
| 9 | dominant prototype support ratio | Support fraction of the dominant prototype |
| 10 | cross-source agreement | Sum over prototypes of support ratio times cross-source support ratio |
| 11 | poor-geometry prototype ratio | Fraction of prototypes whose column5 value exceeds 0.5 |
| 12 | homogeneous bias coordinate | 1 for valid rows; reserved, unused in this checkpoint |

These are precomputed, unstandardized attributes in the representation's feature convention, not arbitrary substitute quality scores. The supplied matcher applies its own saved transformations. An upstream adapter must construct the same prototype representation and attribute definitions; this minimal inference release does not implement that adapter. The schema alone does not reconstruct features from images or establish benchmark reproducibility for a new feature extractor. Zero padding is recommended for invalid slots. Do not replace a missing source with the opposite source under the same source label.

The instance attributes should be repeated on each valid prototype row. Prototype/source ordering and physical IDs must be preserved by the feature adapter. Candidate IDs are never used as learned features.

## Output

`selected_ids` is the final sorted set. `base_selected_ids` is retained in full. At most one extra ID is added. The other arrays follow `instance_ids` order:

- `scores`: sigmoid-transformed decoded margins; used for selection, not calibrated probabilities.
- `ambiguity_scores`: bounded nonlinear S3 correction in (-2,2).
- `base_margins`: continuous margin whose sign encodes the base set.
- `corrected_margins`: base margin plus S3 for every instance, before the protected single-add decoder. A positive value alone does not guarantee selection because the decoder allows at most one addition.
