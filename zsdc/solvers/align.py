import numpy as np


def observation_weights(sample, min_confidence=0.):
    weights = (sample.sensor_mask & np.isfinite(sample.sensor_depth) & (sample.sensor_depth > 0)).astype(np.float64)
    if sample.sensor_confidence is not None:
        c = sample.sensor_confidence
        weights *= np.where(c >= min_confidence, c, 0)
    return weights


def huber_weights(residual, delta):
    if delta <= 0:
        raise ValueError("Huber delta must be positive")
    return np.minimum(1., delta / np.maximum(np.abs(residual), 1e-15))


def align_depth(depth, sample, mode="scale_shift", config=None):
    config = config or {}
    if mode not in ("none", "scale", "scale_shift"):
        raise ValueError(f"Unknown alignment: {mode}")
    depth = np.asarray(depth, dtype=np.float64)
    info = {"requested_mode": mode, "mode": "none", "scale": 1., "shift": 0., "fallback": None}
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Alignment requires a finite, positive prior")
    weights = observation_weights(sample, config.get("min_confidence", 0.))
    usable = weights > 0
    info["alignment_points"] = int(usable.sum())
    if mode == "none":
        return depth.copy(), info
    if not usable.any():
        info["fallback"] = "no_usable_observations"
        return depth.copy(), info
    x, y, w = depth[usable], sample.sensor_depth[usable], weights[usable]
    scale = float(np.exp(np.median(np.log(y) - np.log(x))))
    aligned = depth * scale
    if not np.isfinite(aligned).all() or np.any(aligned <= 0):
        info["fallback"] = "invalid_scale"
        return depth.copy(), info
    info.update(mode="scale", scale=scale)
    if mode == "scale":
        return aligned, info
    if len(x) < 2 or np.ptp(x) <= 1e-8 * max(1., np.max(np.abs(x))):
        info["fallback"] = "insufficient_or_degenerate_affine_observations"
        return aligned, info
    X = np.column_stack([x, np.ones_like(x)])
    # Start from robust scale, then Huber IRLS in relative depth units.
    beta = np.array([scale, 0.])
    for _ in range(config.get("iterations", 5)):
        residual = (X @ beta - y) / np.maximum(y, 1e-6)
        rw = w * huber_weights(residual, config.get("huber_delta", .1)) / np.maximum(y, 1e-6)**2
        Xw, yw = X * np.sqrt(rw[:, None]), y * np.sqrt(rw)
        try:
            beta_new, _, rank, singular = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            info["fallback"] = "affine_linalg_failure"
            return aligned, info
        if rank < 2 or singular[-1] <= singular[0]*1e-10:
            info["fallback"] = "affine_rank_deficient"
            return aligned, info
        beta = beta_new
    candidate = depth * beta[0] + beta[1]
    if beta[0] <= 0 or not np.isfinite(candidate).all() or np.any(candidate <= 0):
        info["fallback"] = "nonpositive_or_invalid_affine_depth"
        return aligned, info
    info.update(mode="scale_shift", scale=float(beta[0]), shift=float(beta[1]))
    return candidate, info
