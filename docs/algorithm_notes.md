# Phase 1 algorithms

All algorithms operate on the same unmasked frozen-model prior. No training, test-GT parameter selection, sensor copying by default, or model-internal changes are used.

| Method | Behavior |
|---|---|
| E0 / rgb_only | Exact official RGB-only output |
| E1 / scale | exp(median(log(sensor)-log(prior))) |
| E2 / scale_shift | Positive scale+shift using Huber IRLS in relative sensor-depth units |
| E3 / graph2d | Fixed global alignment followed by RGB graph log-residual optimization |
| E4 / graph3d | Same alignment/solver, adding depth/XYZ/optional-normal graph structure |

E2 requires two nondegenerate depth values and checks the whole output for positive depth. It falls back to scale, then RGB-only if no usable observation exists. E3/E4 share `align_mode` (default scale_shift); additional global recalibration is not performed.

For aligned prior Da, innovation e=log(sensor)-log(Da), correction r solves

`min sum_i w_i rho(r_i-e_i) + lambda_graph sum_(ij) a_ij (r_i-r_j)^2 + lambda_prior ||r||^2`.

Each undirected edge is stored once. Its Laplacian is symmetric/nonnegative, uses O(N+E) memory, and `lambda_prior>0` makes the linear system SPD, including unobserved disconnected components. Eight-neighbor candidates are default; four-neighbor is configurable. Graph precision is float64 for stable Laplacian row sums.

RGB weights use exp(-||color_i-color_j||²/sigma_rgb²). The `depth` graph additionally weights log-depth differences and breaks edges beyond a fixed log jump. `xyz` adds squared XYZ distance divided by the smaller endpoint depth. `normal` multiplies by positive normal cosine to a fixed power and rejects unreliable normals. There are no global dense kNN links. Weighted degree maps visualize graph connectivity.

The quadratic solve is `(W+lambda_graph L+lambda_prior I)r = W e`. CG uses Jacobi preconditioning; direct sparse solution is a diagnostic alternative for small problems. Both absolute/relative residuals, iterations, return code, sparse-system bytes and duration are recorded. Huber IRLS downweights sensor innovations starting on the first round and reweights using current residuals. Graph and prior terms stay quadratic. Correction is clamped before `Da*exp(r)`.

Defaults are materialized in the saved YAML: lambda_graph=1, lambda_prior=0.001, sensor_weight=1, Huber delta=0.1, five IRLS rounds, maximum |r|=log(4), CG rtol=1e-6/maxiter=1000; RGB/log-depth/relative-XYZ bandwidths 0.1/0.1/0.05; depth discontinuity limit 0.2. They are fixed engineering defaults, **not validated best research settings**. The full effective configuration is saved for each run.

Nonconvergence or invalid corrected depth returns the aligned prior with an explicit fallback reason. Zero usable observations bypasses graph construction. Hard-copy is a separately logged control; it should never be treated as the principal method on noisy data.
