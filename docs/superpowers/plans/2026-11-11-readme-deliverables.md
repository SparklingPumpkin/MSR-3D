# MSR-3D README and Poster Deliverables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a polished, accurate English repository README with the poster assets and verified links to the MSR-3D project page, video, and dataset.

**Architecture:** Keep the existing inference package and documentation source of truth unchanged. Copy the v5.2 poster PDF and PNG into `docs/assets/poster/`, then update the root README to reuse the existing project OG image and point readers to the released artifacts and existing code/data/model-card documentation.

**Tech Stack:** Markdown, Git, shell `cp`/`cmp`/`sha256sum`, Python standard library for local-link validation, and the repository's existing `unittest` suite as a non-blocking baseline reference.

---

### Task 1: Add the approved poster artifacts

**Files:**
- Create: `docs/assets/poster/MSR-3D-poster.pdf`
- Create: `docs/assets/poster/MSR-3D-poster.png`

- [ ] **Step 1: Create the poster asset directory**

Run in `/media/usv012/fjn/projects/3dgs/MSR-3D-readme-polish`:

```bash
mkdir -p docs/assets/poster
```

Expected: `docs/assets/poster/` exists.

- [ ] **Step 2: Copy the released v5.2 poster files unchanged**

```bash
cp -p /media/usv012/fjn/projects/fjn-projects/MSR-3D_NIPS-released/Poster/deliverables/MSR-3D-poster.pdf docs/assets/poster/MSR-3D-poster.pdf
cp -p /media/usv012/fjn/projects/fjn-projects/MSR-3D_NIPS-released/Poster/deliverables/MSR-3D-poster.png docs/assets/poster/MSR-3D-poster.png
```

Expected: both files exist and are non-empty.

- [ ] **Step 3: Verify byte identity and record asset hashes**

```bash
cmp /media/usv012/fjn/projects/fjn-projects/MSR-3D_NIPS-released/Poster/deliverables/MSR-3D-poster.pdf docs/assets/poster/MSR-3D-poster.pdf
cmp /media/usv012/fjn/projects/fjn-projects/MSR-3D_NIPS-released/Poster/deliverables/MSR-3D-poster.png docs/assets/poster/MSR-3D-poster.png
sha256sum docs/assets/poster/MSR-3D-poster.pdf docs/assets/poster/MSR-3D-poster.png
```

Expected: both `cmp` commands exit 0; record both printed SHA-256 values for final reporting.

### Task 2: Rewrite the repository entry-point README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the short header with the project entry point**

Use the existing project OG image `docs/assets/og.jpg` as the first image. Set the title to `MSR-3D: Multi-Mode Semantic Representation for Open-Vocabulary 3D Scene Understanding` and identify NeurIPS 2026. Add direct links to:

```markdown
[Project page](https://SparklingPumpkin.github.io/MSR-3D/)
[Interactive 3D demo](https://SparklingPumpkin.github.io/MSR-3D/#demo)
[Overview video](https://SparklingPumpkin.github.io/MSR-3D/#video)
[Data v1.1.0](https://huggingface.co/datasets/KaiLai/MSR-3D-Data/tree/v1.1.0)
[Poster PDF](docs/assets/poster/MSR-3D-poster.pdf)
```

Keep the capitalization/case of these URLs exactly as shown in the existing project page and README links.

- [ ] **Step 2: Add the concise method summary and poster preview**

Explain that MSR-3D retains a prototype per appearance mode instead of averaging all views of an object into a single feature, then learns to match a text query across prototypes and select 3D instances. Include a clickable poster preview:

```markdown
[![MSR-3D poster](docs/assets/poster/MSR-3D-poster.png)](docs/assets/poster/MSR-3D-poster.pdf)
```

State that the PDF is the print version and the PNG is the 150-DPI preview. Link the overview video to the existing project-page video section; do not copy the large MP4 into the code repository.

- [ ] **Step 3: Add the verified result summary and release versions**

Use scene-averaged mIoU gains from the current project page and name OpenSplat3D as the strongest compared method for these columns:

| Evaluation set | Scope | MSR-3D mIoU | OpenSplat3D mIoU | Gain |
|---|---|---:|---:|---:|
| LERF-OVS | 67 queries | 64.70 | 58.01 | +6.69 |
| LERF-OVS unseen-string subset | 45 queries | 63.15 | 59.69 | +3.46 |
| 3DOVS | 61 queries, 10 scenes | 80.98 | 79.41 | +1.57 |

Label the gains as percentage points; note that all methods use the same reconstruction and object candidates; link to the project-page full table and `MODEL_CARD.md`; and retain the model-card caveat that release validation was on the same machine and that a fresh online clean-environment installation was not verified. State code v1.2.0 and data v1.1.0.

- [ ] **Step 4: Retain and organize reproducibility instructions**

Preserve the current README's Python 3.11, NVIDIA GPU/CUDA, install, data download, pipeline, cached benchmark, and test commands. Keep links to `DATA_FORMAT.md`, `INPUT_FORMAT.md`, `PIPELINE.md`, `MODEL_CARD.md`, and `THIRD_PARTY.md`. Do not claim a permissive code or weight license; say rights and upstream notices are described in the linked model card and third-party document.

Use the exact citation from the project page:

```bibtex
@inproceedings{fu2026msr3d,
  title     = {{MSR-3D}: Multi-Mode Semantic Representation for
               Open-Vocabulary 3D Scene Understanding},
  author    = {Jianan Fu and Ting Wu and Rui Yang and Jiacheng Sun and
               Xinghong Zhou and Chenghai Mao and Pan Jiaxiang and Xiaomao Li},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Expected: README links to all listed local guides and poster assets; method, versions, metrics, and BibTeX match their sources.

### Task 3: Validate the documentation and poster package

**Files:**
- Validate: `README.md`
- Validate: `docs/assets/og.jpg`
- Validate: `docs/assets/poster/MSR-3D-poster.pdf`
- Validate: `docs/assets/poster/MSR-3D-poster.png`

- [ ] **Step 1: Check Markdown links and referenced local files**

Run from the worktree root:

```bash
python - <<'PY'
import re
from pathlib import Path

root = Path.cwd()
text = (root / "README.md").read_text()
targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
html_images = re.findall(r'<img[^>]+src="([^"]+)"', text)
for target in [*targets, *images, *html_images]:
    if target.startswith(("https://", "http://", "#", "mailto:")):
        continue
    path = (root / target.split("#", 1)[0]).resolve()
    assert path.is_relative_to(root.resolve()), f"path escapes repository: {target}"
    assert path.is_file(), f"missing local README target: {target}"
print(f"Validated {len(targets)} Markdown links, {len(images)} Markdown images, and {len(html_images)} HTML images.")
PY
```

Expected: exit 0 and every local Markdown target/image resolves inside the repository.

- [ ] **Step 2: Confirm the poster files still match their release sources**

Run both `cmp` commands from Task 1 again.

Expected: both exit 0.

- [ ] **Step 3: Review README diff and whitespace**

```bash
git diff --check
git diff -- README.md
git status --short
ls -lh docs/assets/poster/MSR-3D-poster.pdf docs/assets/poster/MSR-3D-poster.png
```

Expected: no whitespace errors; the README diff and status show only the approved README and two poster files as deliverable changes, apart from the already committed design and plan documents.

- [ ] **Step 4: Record the existing test baseline without changing code**

The full suite was already attempted from this worktree using `PYTHONPATH=src python -m unittest discover -s tests -v`: 20 of 23 passed; three fail because the current interpreter NumPy lacks `numpy.bitwise_count`, although `pyproject.toml` pins NumPy 2.0.2. Do not edit code or install dependencies in this documentation task. Report this existing environment limitation accurately.

### Task 4: Commit and publish the README update

**Files:**
- Commit: `README.md`
- Commit: `docs/assets/poster/MSR-3D-poster.pdf`
- Commit: `docs/assets/poster/MSR-3D-poster.png`

- [ ] **Step 1: Stage only the approved README and poster files**

```bash
git add README.md docs/assets/poster/MSR-3D-poster.pdf docs/assets/poster/MSR-3D-poster.png
git diff --cached --check
git status --short
```

Expected: staged changes contain the README and two poster files only; design/plan documents are already committed.

- [ ] **Step 2: Commit with the required co-author trailer**

```bash
git commit -m "docs: polish MSR-3D release README" -m "Add the released poster preview and print PDF.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

Expected: a new commit on branch `docs/readme-deliverables`.

- [ ] **Step 3: Push the feature branch**

```bash
git push -u origin docs/readme-deliverables
```

Expected: GitHub contains the branch and commit; `main` remains unchanged. If push authentication fails, report the exact failure without changing the remote URL or using credentials in command arguments.
