# Geometry and data conventions

## Coordinates and units

- RGB is HWC float32 in [0,1]. Sensor depth and evaluator GT are separate HW arrays, expressed in meters.
- Internal depth is camera optical-axis Z. Dataset defaults `depth_kind: z` and `gt_depth_kind: z` are **assumptions** pending server verification, not claims established by PNG units alone. `range` explicitly divides ray distance by the norm of `K^-1 [u,v,1]` after normalizing the ray's Z to one. Missing K makes range conversion fail.
- Pixel K uses integer-index pixel centers. Official MoGe/utils3d K is normalized with `u_norm=(u+0.5)/W`, `v_norm=(v+0.5)/H`. Thus `fx_px=W*fx_norm`, `cx_px=W*cx_norm-0.5`, and equivalently for y. Resize preserves normalized K.
- MoGe receives horizontal FOV only; it does not receive full calibrated K, principal point or skew. Auto mode uses K-derived focal FOV when available; otherwise it estimates FOV. Geometry graphs use full input K when available, else pixelized MoGe K. These protocols are logged separately.

## Validity and masks

`sensor_mask` encodes presence, independently of `sensor_confidence`. Construction removes nonfinite/nonpositive measurements. Confidence can downweight or reject present observations without changing the observed/unobserved evaluation partition.

For uint16 input, raw and GT independently default to invalid codes `[0,65535]`. **The 65535 policy is configurable and must be checked with actual sentinel counts on the server.** Raw saturation and GT invalid encodings must not be inferred from GT accuracy. Positive measurements not listed as sentinels remain observations even if erroneous. Files are required to be aligned single-channel uint16 millimeter PNGs. Intrinsics must be a valid 3x3 pixel matrix; unsupported sidecar encodings fail visibly rather than guessing.

The completion API has no GT field. `EvaluationTarget` is returned separately by the loader. Cache/audit/completion actions do not open GT files. Only evaluation and preflight load GT; its hashes and sentinel diagnostics never enter `CompletionInput` or prediction cache.

Official prediction masks are retained but never intersected with sensor masks or GT evaluation masks. E0 is exactly the official unmasked depth array. Nonpositive/nonfinite priors cause the audit to fail. For direct solver API calls, an invalid prior at a valid sensor pixel can use that sensor as its starting value; an unsupported invalid prior elsewhere fails the sample. No GT pixel is silently removed to rescue a prediction.

## Resolution and sampling

Default resizing preserves aspect ratio, never enlarges an image, and limits area to 90,000 pixels. Explicit `[H,W]` can change aspect ratio and is recorded. Original resolution is opt-in. RGB uses area resampling. Sensor coordinates map through pixel centers; observations colliding at a destination pixel use the smallest Z with stable source-index tie breaking. Confidence, when provided to the remapper, follows that same winner. Collision counts and original/resized shapes are logged.

GT and its independent validity mask use the same explicit nearest-neighbor coordinate lookup. No bilinear depth interpolation is used.

`native` means all surviving **raw-derived observations at the experiment resolution**, not preservation of every original-resolution sensor pixel. Random/grid budgets are applied after this common resize. Random budgets are nested for a fixed sample/seed. Grid anchors snap to actual raw observations; duplicate anchors are resolved with seeded raw-only filling. Grid seeds may produce identical sets if no collisions/filling occur. Insufficient support returns all available points and records the actual count.
