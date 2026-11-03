# MSR-3D 1.2.0

**Task:** fixed-weight open-vocabulary instance matching from reconstructed, segmented scenes or cached semantic evidence. The matching parameters are unchanged from version 1.0.0.

**Contents:** 27 numerical arrays for the semantic row scorer, instance aggregator, query-threshold estimator, primary boundary, supplementary classifier, S3 readout/attention and feature scalers. Only parameters used by the full inference path are included.

**Input boundary:** the full pipeline starts from reconstructed Gaussian PLY, aligned instance labels, posed RGB images and segmented instance renders. It rebuilds observation pools and instance representations, encodes text queries and predicts instance sets. The data release supplies processed evaluation mask banks. CLIP and MasQCLIP checkpoints are obtained separately from the official links in its README. Cached query/prototype similarities and attributes remain supported by the CPU matcher.

**Supervision:** learned using the source scene Waldo Kitchen. Some modules were fitted and monitored on overlapping source-query sets; this release does not claim an independent training/validation split or provide training reproducibility. Target benchmarks were used during exploratory development. Source annotations and per-query training records are not distributed here.

**Input revision:** paired data v1.1.0 changes the 3DOVS S1 anchor from the shared pool mean to geometric top-five views with middle-crop averaging. LERF keeps geometric top-five views with all-three-crop averaging. Matching weights and source supervision are unchanged. This is a target-benchmark-informed configuration choice, not an independently held-out estimate. In the configuration experiment on all ten 3DOVS scenes (61 scene/query pairs), scene-macro IoU/F1 changed from 79.8566% / 81.3492% to 80.9799% / 83.0238%.

**Behavior:** a base selection from two normalized semantic channels, followed by bounded S3 correction and protected single addition. The S3 branch is nonlinear and its coefficients are not independent effects of the original attributes.

**Limitations:** performance depends on upstream representations, instance inventory, feature normalization and scene domain. Missing, fragmented or merged instances cannot be repaired by this matcher. S3 has not demonstrated a consistent benefit across held-out scenes or datasets; numerical agreement with a reference does not establish such a benefit.

**Reproducibility:** code v1.2.0 with data v1.1.0 was checked from newly extracted release archives. All ten 3DOVS scenes were rerun from preprocessed scene assets through observation pools, representations and matching; final instance selections, per-frame counts and per-query/aggregate metrics matched the revised references. The supplied-pool shortcut matched on the same ten scenes. Cached evaluation passed on all 128 scene/query pairs, retaining the LERF67/45 results. All ten rebuilt 3DOVS anchor banks were bitwise equal to the supplied anchors. The CPU suite passed 23 tests. The earlier release also completed full three-stage validation on all 14 scenes. These checks used the same machine and an existing offline CUDA dependency environment; a fresh online installation on other hardware has not been verified. Near-tied prototype choices can differ; retain 16 CPU threads for prototype construction and one for matching.

**Data:** [MSR-3D-Data](https://huggingface.co/datasets/KaiLai/MSR-3D-Data) supplies preprocessed scenes, observation pools, instance representations, cached matching inputs and evaluation data. Metric definitions are documented in `evaluation/cached-matcher-reference/PROTOCOL.md` after preparation.

**Rights:** this distribution does not add a permissive license or change ownership of the code or learned parameters. No rights to third-party assets are granted. The project rights holder determines licensing for redistribution and reuse.
