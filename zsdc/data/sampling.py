import hashlib
from dataclasses import replace

import numpy as np


def sample_sensor(sample, pattern="native", count=100, seed=0):
    if count < 0:
        raise ValueError("Observation count must be nonnegative")
    candidates = np.flatnonzero(sample.sensor_mask)
    n = min(count, len(candidates))
    stable = int.from_bytes(hashlib.sha256(f"{sample.sample_id}:{seed}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(stable)
    if pattern == "native":
        chosen = candidates
    elif pattern == "random":
        chosen = rng.permutation(candidates)[:n]
    elif pattern == "grid":
        # Regular 2D anchors snapped to actual observations. Resolve duplicate
        # anchors and fill remaining budget from a seeded raw-only permutation.
        from scipy.spatial import cKDTree
        if n:
            h, w = sample.sensor_mask.shape
            nx = max(1, min(w, int(np.ceil(np.sqrt(n*w/h)))))
            ny = max(1, int(np.ceil(n/nx)))
            gy, gx = np.meshgrid((np.arange(ny)+.5)*h/ny-.5, (np.arange(nx)+.5)*w/nx-.5, indexing="ij")
            anchors = np.column_stack([gy.ravel(), gx.ravel()])
            # If the rectangular grid exceeds the budget, distribute retained anchors.
            anchors = anchors[np.linspace(0, len(anchors)-1, n).astype(int)]
            y, x = np.unravel_index(candidates, (h,w))
            _, nearest = cKDTree(np.column_stack([y,x])).query(anchors)
            chosen = np.unique(candidates[nearest])
            remaining = np.setdiff1d(candidates, chosen, assume_unique=True)
            chosen = np.r_[chosen, rng.permutation(remaining)[:n-len(chosen)]]
        else:
            chosen = np.array([], dtype=int)
    else:
        raise ValueError(f"Unknown sampling pattern: {pattern}")
    mask = np.zeros(sample.sensor_mask.size, bool)
    mask[chosen] = True
    mask = mask.reshape(sample.sensor_mask.shape)
    return replace(sample, sensor_depth=np.where(mask, sample.sensor_depth, 0), sensor_mask=mask,
                   metadata={**sample.metadata, "sampling": pattern, "seed": seed,
                             "requested_points": None if pattern == "native" else count,
                             "available_points": len(candidates), "actual_points": len(chosen),
                             "observation_hash": hashlib.sha256(mask.tobytes()).hexdigest()})
