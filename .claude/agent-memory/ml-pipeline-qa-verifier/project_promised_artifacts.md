---
name: project-promised-artifacts
description: Full per-model artifact list for `artifacts/{limu_bert,harnet10}/`; the source of truth for QA check (e).
metadata:
  type: project
---

Promised artifacts each training script must produce (per CLAUDE.md spec and per the model scripts' write calls).

**Why:** Check (e) artifact completeness needs an authoritative expected-output list to compare against on-disk results; the plan listed these explicitly.

**How to apply:** After training has run, verify each of these paths exists, is non-empty, parses (for `.json`/`.csv`) or opens (for `.png`), and has an mtime after the training run started. `{model}` ∈ {limu_bert, harnet10}; `{mode}` ∈ {frozen, partial}; `{person}` ∈ {adnan, ahmed, chi, mercy, novel}.

Per model directory `artifacts/{model}/`:

- `checkpoints/{model}_{mode}.pt` — main checkpoint
- `checkpoints/lopo_frozen_{person}.pt` — 5 LOPO checkpoints
- `checkpoints/scaler.pkl` — always written, may pickle `None`
- `checkpoints/config.json`
- `plots/training_curves_{mode}.png`
- `plots/confusion_window_{mode}.png`
- `plots/confusion_file_{mode}.png`
- `plots/roc_{mode}.png`
- `plots/confidence_hist_{mode}.png`
- `plots/lopo/{person}_confusion.png` (×5)
- `plots/lopo_summary.png`
- `metrics/{mode}_metrics.json`
- `metrics/lopo_results.csv`
- `metrics/lopo_summary.json`

Note: LIMU-BERT `--task lopo` does NOT write a partial-mode checkpoint; LOPO only does frozen. HARNet10 mirrors this. Don't flag the absence of `lopo_partial_*.pt` as a failure.

Related: [[project-pipeline-contract]].
