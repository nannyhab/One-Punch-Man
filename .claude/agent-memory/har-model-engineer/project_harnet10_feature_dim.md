---
name: project-harnet10-feature-dim
description: HARNet10 backbone (OxWearables/ssl-wearables) discovered feature dim verified via torch.hub probe.
metadata:
  type: project
---

`OxWearables/ssl-wearables` `harnet10` exposes a 1024-dim feature vector after stripping `model.classifier` and global-average-pooling the (B, C, T) backbone output. The hub model also prints `"131 Weights loaded"` when pretrained=True succeeds — useful sentinel for "real backbone loaded vs. random fallback".

**Why:** Spec said "probably 1024 or 512 — verify at runtime". Verified once via smoke test; saved to checkpoint as `feature_dim` so eval-time reload (which may have torch.hub unreachable) can rebuild the head correctly.

**How to apply:** Whenever rebuilding the model from a checkpoint, call `_load_harnet10_backbone()` first and only fall back to the small `_RandomBackbone` if hub fails. The checkpoint's `feature_dim` field is authoritative — if it doesn't match the discovered dim, that's a version skew worth investigating.
