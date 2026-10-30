# MSR-3D matcher 1.0.0

**Task:** fixed-weight open-vocabulary instance matching from cached semantic evidence.

**Contents:** 27 numerical arrays for the semantic row scorer, instance aggregator, query-threshold estimator, primary boundary, supplementary classifier, S3 readout/attention and feature scalers. Only parameters used by the full inference path are included.

**Input boundary:** cached query/prototype similarities and representation statistics, plus a complete anchor similarity bank. No image encoder or representation-construction weights are included. Output is an instance set; generating a pixel mask requires an external renderer.

**Supervision:** learned using the source scene Waldo Kitchen. Some modules were fitted and monitored on overlapping source-query sets; this release does not claim an independent training/validation split or provide training reproducibility. Target benchmarks were used during exploratory development. Source annotations and per-query training records are not distributed here.

**Behavior:** a base selection from two normalized semantic channels, followed by bounded S3 correction and protected single addition. The S3 branch is nonlinear and its coefficients are not independent effects of the original attributes.

**Limitations:** performance depends on upstream representations, instance inventory, feature normalization and scene domain. Missing, fragmented or merged instances cannot be repaired by this matcher. S3 has not demonstrated a consistent benefit across held-out scenes or datasets; numerical agreement with a reference does not establish such a benefit. Fixed-seed CPU parity was checked on cached inputs, not an independent raw-image evaluation.

**Assets:** no benchmark examples, ground-truth masks, raw images, external encoder weights, training checkpoints or optimization state are distributed. The included example generator produces synthetic numerical arrays only.

**Rights:** this distribution does not add a permissive license or change ownership of the code or learned parameters. No rights to third-party assets are granted. The project rights holder determines licensing for redistribution and reuse.
