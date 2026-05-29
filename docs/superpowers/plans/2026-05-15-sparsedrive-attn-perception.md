# SparseDrive Attention Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a HUGSIM script that runs scene-0254 extreme with SparseDriveV2 and writes per-frame deformable-attention perception visualizations.

**Architecture:** A single HUGSIM-side Python script creates a temporary SparseDrive bridge and launcher under the output directory, then runs `closed_loop.py` with environment overrides. The generated bridge monkey-patches SparseDrive's `DeformableFeatureAggregation.forward` at runtime to capture deformable sampling locations and weights, renders camera heatmaps plus existing BEV overlays, and saves raw `.npz` attention data.

**Tech Stack:** Python, pytest, OpenCV, NumPy, PyTorch/SparseDrive runtime, HUGSIM closed-loop FIFO protocol.

---

### Task 1: Script Utilities And Tests

**Files:**
- Create: `scripts/run_sparsedrive_attn_perception.py`
- Create: `tests/test_sparsedrive_attn_perception.py`

- [ ] Write tests for scenario output naming, launcher generation, and camera heatmap rendering.
- [ ] Run `pytest tests/test_sparsedrive_attn_perception.py -v` and confirm the tests fail because the script does not exist.
- [ ] Implement minimal utility functions in `scripts/run_sparsedrive_attn_perception.py`.
- [ ] Run the tests and confirm they pass.

### Task 2: Runtime Bridge Generation

**Files:**
- Modify: `scripts/run_sparsedrive_attn_perception.py`
- Test: `tests/test_sparsedrive_attn_perception.py`

- [ ] Add tests that generated bridge text contains the SparseDrive monkey-patch and expected output subdirectories.
- [ ] Run the targeted tests and confirm they fail.
- [ ] Implement bridge and launcher generation.
- [ ] Run the targeted tests and confirm they pass.

### Task 3: End-To-End Run Command

**Files:**
- Modify: `scripts/run_sparsedrive_attn_perception.py`

- [ ] Add CLI defaults for scene-0254 extreme, checkpoint path, SparseDrive repo, Python binary, and output root.
- [ ] Add command construction for `closed_loop.py` with temporary config and environment overrides.
- [ ] Run static help and unit tests.
- [ ] Run the requested scenario if dependencies/GPU are available.
