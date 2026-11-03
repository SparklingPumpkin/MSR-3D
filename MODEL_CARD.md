# MSR-3D 1.1.0

**Task:** fixed-weight open-vocabulary instance matching from reconstructed, segmented scenes or cached semantic evidence. The matching parameters are unchanged from version 1.0.0.

**Contents:** 27 numerical arrays for the semantic row scorer, instance aggregator, query-threshold estimator, primary boundary, supplementary classifier, S3 readout/attention and feature scalers. Only parameters used by the full inference path are included.

**Input boundary:** the full pipeline starts from reconstructed Gaussian PLY, aligned instance labels, posed RGB images and segmented instance renders. It rebuilds observation pools and instance representations, encodes text queries and predicts instance sets. The data release supplies processed evaluation mask banks. CLIP and MasQCLIP checkpoints are obtained separately from the official links in its README. Cached query/prototype similarities and attributes remain supported by the CPU matcher.

**Supervision:** learned using the source scene Waldo Kitchen. Some modules were fitted and monitored on overlapping source-query sets; this release does not claim an independent training/validation split or provide training reproducibility. Target benchmarks were used during exploratory development. Source annotations and per-query training records are not distributed here.

**Behavior:** a base selection from two normalized semantic channels, followed by bounded S3 correction and protected single addition. The S3 branch is nonlinear and its coefficients are not independent effects of the original attributes.

**Limitations:** performance depends on upstream representations, instance inventory, feature normalization and scene domain. Missing, fragmented or merged instances cannot be repaired by this matcher. S3 has not demonstrated a consistent benefit across held-out scenes or datasets; numerical agreement with a reference does not establish such a benefit.

**Reproducibility:** all three inference stages were independently installed and rerun locally across 14 scenes and 128 scene/query pairs. Final instance selections and evaluation metrics matched the references. Some near-tied prototype choices differ; retain 16 CPU threads for prototype construction and one for matching. This check used the same machine with offline dependency provisioning; a fresh online installation on other hardware has not been verified.

**Data:** [MSR-3D-Data](https://huggingface.co/datasets/KaiLai/MSR-3D-Data) supplies preprocessed scenes, observation pools, instance representations, cached matching inputs and evaluation data. Metric definitions are documented in `evaluation/cached-matcher-reference/PROTOCOL.md` after preparation.

**Rights:** this distribution does not add a permissive license or change ownership of the code or learned parameters. No rights to third-party assets are granted. The project rights holder determines licensing for redistribution and reuse.
