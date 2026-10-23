# Phase 0/1 experiment protocol

## Fixed inputs

Keep the supplied 16-scene train list and 5-scene validation list. This project performs no training. Train is available for preflight/smoke; validation is used for reported evaluation. Selection is deterministic round-robin over sorted scenes/sample IDs. No method can select samples using errors or GT statistics. Lists, RGB/raw/GT files, sidecar K, source and checkpoint hashes are logged.

The v3 full matrix has 31 sensor conditions per image: one native condition and random/grid at five budgets times three seeds. E0–E4 make 155 outputs per image. E0 is repeated in the tables to preserve the same evaluation masks for each sensor condition, but there is only one RGB forward/cache fetch. Point patterns are constructed once per condition, shared by all methods, and fingerprinted.

Default v2 contrast: first eight deterministically selected images, native plus random-100/seed-0. Set that model's `full_matrix: true` for the complete matrix. v3 uses SSR=3; audits compare SSR=0/1/3 plus any configured refinement count. No measurement enters SSR in Phase 1.

## Audit before benchmark

Audits compare wrapper/official depth, points, K, mask, normals and available per-step arrays; repeated predictions are compared, frozen/eval state is checked and full model state hashes must match. Unmasked maps must be finite/positive. FP32 uses rtol=1e-5, atol=1e-6. FP16 uses a separate fixed rtol/atol=1e-3 and is recorded independently. Tolerances never auto-expand after failure. Source/model/runtime changes require a new audit; solver/sampling changes do not.

Only the FOV modes seen in a passing audit can be evaluated. If a dataset mixes calibrated and uncalibrated cameras, choose audit examples covering both modes or audit the explicit estimated-FOV protocol. Full K in the graph is not equivalent to conditioning MoGe on full K.

## Metrics and errors

Report RMSE, MAE, AbsRel and delta1 on all GT-valid, observed GT-valid and unobserved GT-valid pixels. Unobserved is `GT_valid & !sensor_mask`, even for corrupted observations or rejected low-confidence measurements. Edge bands use a fixed GT log-depth jump threshold and fixed dilation, strictly within the evaluator.

Each region records counts and sufficient statistics; aggregation reports both per-image means and pixel-pooled metrics. Empty regions return null. If any prediction is invalid on a fixed region, that region's metrics and aggregate are undefined and the invalid count is shown. Errors are never computed by silently deleting difficult predicted pixels. Data/prior failures cannot enter the denominator without a target; requested/processed coverage is therefore explicitly reported beside failures.jsonl.

All seeds remain in tables. No best-seed or per-image oracle selection is supported. Plots average seed-specific pooled RMSE values only for visualization, with full rows retained.

## Time, memory and output

CUDA events synchronize around inference; model timing and full prediction timing are saved in the cache. CPU solver logs distinguish graph construction/completion from each linear solve. Actual wall time includes I/O and export. Reconstructed uncached per-condition time adds one saved full-model prediction cost, never the entire RGB cache-generation run. Plot runtime is model+completion and excludes export/I/O; both definitions are explicit.

CPU peak RSS is a process high-water mark. CUDA peak allocated memory is measured around RGB inference. Sparse matrix bytes are recorded separately; they are not claimed to be total peak memory. Cold JIT/first-use time may be present; no unrecorded warm-up is subtracted.

Runs write full YAML, manifest with coverage, per_image.csv, per_image.jsonl, summary.csv/.md, timings.json, failures.jsonl and fixed-order visualizations. Per-condition details are streamed to bound memory; group summaries are checkpointed every 32 images and at completion. Output directories must be empty to prevent mixing runs. Solver sweeps share audited RGB caches and write separate run directories. The graph sweep defaults are depth, XYZ and XYZ+normal ablations.

True-data smoke and full metrics remain **pending server execution**. Synthetic outputs use an analytic prior with an explicit SYNTHETIC label. They validate software/numerics only.
