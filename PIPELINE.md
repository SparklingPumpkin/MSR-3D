# Full-method inference contract

## Preprocessed scene assets

For each scene: trained Gaussian PLY, physical instance labels aligned to the Gaussians, posed RGB images, 2D segmentation masks, camera intrinsics/extrinsics, and explicit frame/instance mappings. Encoder weights and image-crop, geometry and filtering parameters are part of the runtime contract.

## Stages

1. **Observation pool:** derive retained per-view semantic features from the 2D and rendered-instance sources and attach geometry and source/frame metadata. Verify the full inventory, selection/filter decisions and numeric features against the distributed observation reference.
2. **Instance representation:** construct prototype embeddings, role assignments, prototype attributes, instance statistics and semantic anchors from the new pool. Verify every instance against the distributed representation reference.
3. **Matching:** encode query text, compute similarities with the new representation, run the fixed matching networks, and recompute mask and instance-set metrics.

Observation and representation caches serve as convenient entry points. The full cold-start run recomputes both stages from the preprocessed scene assets and uses the caches only for comparison.

## Acceptance

Run from an independently installed repository and a portable data copy. Each stage records its actual inputs, outputs and comparisons. Report exact agreement and numerical tolerances separately. All 14 scenes and LERF67/LERF45/3DOVS61 cohort definitions must be accounted for.
