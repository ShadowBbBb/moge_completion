# Phase 0 source audit

Audit date: 2026-10-23. Starting checkout: `d7d24c3c73ea92b064c106b4d1bd7e8aeebcb6b3`.
The preceding official MoGe commit is `74fbce0` (`V3 (#158)`). `git diff 74fbce0 -- moge` was empty before implementation. No files under `moge/` are changed by this project.

## Confirmed local interfaces

- `moge.model.import_model_class_by_version` selects v2/v3 lazily.
- v2 `from_pretrained` accepts a checkpoint file or HF repository containing `model.pt`; it loads with `strict=False` and warns about missing/randomly initialized parameters. ZSDC rejects these warnings. A v3 checkpoint must actually contain a refiner for SSR steps.
- v3 `infer` supports `refine_steps`, `return_per_step`, `fov_x`, `apply_mask`, `force_projection` and `use_fp16`. v2 does not accept the v3 refinement/per-step keywords.
- Both versions return normalized camera intrinsics. v3 returns K+1 depth/points/intrinsics maps when per-step output is requested.
- `apply_mask=False` retains numerical predictions outside the official mask; the actual model mask is independently retained in the cache.
- v3 factorized internal coordinates are `(x/z,y/z,logz)`. Voxelization forces fp32 and quantizes logz, not metric XYZ. Official focal/shift recovery runs separately for each step. Phase 0/1 does not touch these internals.
- Models run in eval/inference mode with all parameter gradients disabled. Audits hash the entire state dictionary before and after inference, compare all supported output maps and repeatability, and check pixel-camera unprojection against official points.

## Dependencies and execution status

The pre-existing GPU setup uses pinned utils3d-moge, pipeline and FlexGEMM commits. ZSDC does not change these pins or the existing setup scripts. `requirements-zsdc-cpu.txt` installs only the geometry/report/test dependencies. Package discovery now also includes `zsdc`; PyYAML is exposed through an optional `zsdc` extra.

Local Python: Anaconda 3.13.9 / Windows. NumPy, SciPy, OpenCV, YAML, matplotlib and pytest are available. PyTorch, utils3d-moge, Triton/FlexGEMM, model weights and real RobbySimVal frames have **not** been validated here. Actual package versions, checkpoint file SHA256, source content hashes, CUDA build and device information are written by server audit tools. No checkpoint hash is invented in this document.

The local CPU test suite and analytic synthetic smoke are executable. Official GPU parity is explicitly skipped unless `ZSDC_GPU_CONFIG` is set. This is not evidence that MoGe or SSR executes successfully on this machine.

## Data sources

User-provided lists were inspected read-only from the neighboring project: 29,455 train frames across 16 scenes; 9,521 validation frames across 5 different scenes. No duplicate IDs, malformed columns or mismatched RGB/raw/GT filenames were found. They are deployment assets, not copied automatically into this repository.

The [official Robbyant dataset card](https://huggingface.co/datasets/robbyant/mdm_depth/blob/main/README.md) documents camera-local `intrinsics.txt`, the RobbySimVal filenames and uint16 millimeter depth. It does not establish every invalid-value or range-versus-Z convention needed by this application. The defaults for those conventions are explicit assumptions to verify on the server; see geometry_convention.md.

## Runtime gates

Completion/cache generation requires a passing audit matching the model/checkpoint, inference configuration, source content, and runtime identity. Known/estimated FOV modes are gated separately. Cache-only runs reuse that attested inference identity, check current source content and model settings, and record their separate CPU environment. A missing cache is an error; cache-only mode never loads or downloads a model.
