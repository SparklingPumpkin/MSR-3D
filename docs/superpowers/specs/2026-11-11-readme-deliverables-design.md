# MSR-3D README and Poster Deliverables Design

## Goal

Make the repository root README a polished English entry point to the released MSR-3D code, data, project page, overview video, and poster, while retaining accurate installation and reproduction guidance.

## Audience and presentation

The README is for research readers and practitioners discovering the NeurIPS 2026 paper and release. Use the repository's existing `docs/assets/og.jpg` as the hero image, followed by concise links to the project page, dataset, video section, and poster. Keep the visual language consistent with the existing dark project page; avoid adding badges or claims that are not already verified.

## README sections

1. **Project header:** paper title, NeurIPS 2026, existing social-preview image, and direct links to the project page, data v1.1.0, overview video, and poster PDF.
2. **Overview:** explain that MSR-3D preserves a prototype per appearance mode instead of averaging all object views into one feature, then learns to match text queries to 3D instances.
3. **Poster and overview video:** show the poster PNG from the repository and link it to the print PDF; link the video to the existing project-page video section rather than copying a 74 MB MP4 into the README repository.
4. **Results:** summarize the verified scene-averaged mIoU gains from the current project page: +6.69 on LERF-OVS (67 queries), +3.46 on its 45-query unseen-string subset, and +1.57 on 3DOVS (61 queries / 10 scenes). State that these are gains over the strongest compared method and link the full table and caveats to the project page and model card.
5. **Release and data:** identify code v1.2.0 and data v1.1.0; retain the existing Hugging Face data instructions and links to `DATA_FORMAT.md` and `INPUT_FORMAT.md`.
6. **Install, run, and check:** preserve the existing tested Python/CUDA setup, commands, benchmark invocation, and unittest command. Do not imply that a fresh online install or clean-environment run was verified.
7. **Rights and third-party components:** link `MODEL_CARD.md` and `THIRD_PARTY.md`; do not add a license grant or suggest that third-party assets are relicensed.
8. **Citation:** use the exact BibTeX currently published on the project page.

## Assets and file changes

- Modify the repository root `README.md`.
- Add `docs/assets/poster/MSR-3D-poster.png` and `docs/assets/poster/MSR-3D-poster.pdf`, copied unchanged from `/media/usv012/fjn/projects/fjn-projects/MSR-3D_NIPS-released/Poster/deliverables/`.
- Do not modify inference code, model weights, data, project-page content, or third-party files.
- Do not invite or otherwise change access for the `pym96` GitHub account in this task.

## Validation and submission

- Confirm README relative links and image references resolve to files in the repository; confirm external links use the verified project-page and Hugging Face URLs.
- Compare poster copies byte-for-byte with their v5.2 source files and record their SHA-256 values.
- Review the resulting README diff for factual accuracy, especially performance scope and license language.
- Existing unit tests are not a gate for these documentation-only changes. The baseline invocation from the isolated worktree had 20/23 passing and three unrelated `numpy.bitwise_count` errors because the current interpreter's NumPy differs from the version pinned in `pyproject.toml` (2.0.2). Do not change code or dependencies to address that in this task.
- Commit the README and poster assets on the isolated branch and push that branch to GitHub. Do not directly alter `main`; present the pushed branch/PR path for integration.
