from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..data.geometry import unproject


@dataclass
class GeometryGraph:
    i: np.ndarray
    j: np.ndarray
    weights: np.ndarray
    size: int

    def laplacian(self):
        i, j, a = self.i, self.j, self.weights
        return sparse.coo_matrix((np.concatenate([a, a, -a, -a]),
                                  (np.concatenate([i, j, i, j]), np.concatenate([i, j, j, i]))),
                                 shape=(self.size, self.size)).tocsr()


def build_graph(image, depth, K=None, kind="rgb", normal=None, config=None):
    config = config or {}
    if kind not in ("rgb", "depth", "xyz", "normal"):
        raise ValueError(f"Unknown graph kind: {kind}")
    if kind in ("xyz", "normal") and K is None:
        raise ValueError("Metric geometry graph requires K")
    if kind == "normal" and normal is None:
        raise ValueError("Normal ablation requires model normals")
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Graph requires positive finite depth")
    h, w = depth.shape
    ids = np.arange(h*w).reshape(h, w)
    offsets = [(0, 1), (1, 0)]
    connectivity = config.get("connectivity", 8)
    if connectivity == 8:
        offsets += [(1, 1), (1, -1)]
    elif connectivity != 4:
        raise ValueError("connectivity must be 4 or 8")
    sigma_rgb, sigma_depth, sigma_xyz = (config.get(k, default) for k, default in
                                        [("sigma_rgb", .1), ("sigma_depth", .1), ("sigma_xyz", .05)])
    if min(sigma_rgb, sigma_depth, sigma_xyz) <= 0:
        raise ValueError("Graph bandwidths must be positive")
    points = unproject(depth, K) if kind in ("xyz", "normal") else None
    logd = np.log(depth)
    ii, jj, aa = [], [], []
    for dy, dx in offsets:
        sy = slice(0, h-dy)
        ty = slice(dy, h)
        sx, tx = (slice(0, w-dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w+dx))
        s, t = (sy, sx), (ty, tx)
        color_distance = np.sum(np.square(image[s]-image[t], dtype=np.float64), axis=-1)
        a = np.exp(-color_distance / sigma_rgb**2)
        jump = np.abs(logd[s]-logd[t])
        if kind != "rgb":
            a *= np.exp(-(jump / sigma_depth)**2)
            a = np.where(jump > config.get("max_log_jump", .2), 0., a)
        if points is not None:
            distance = np.linalg.norm(points[s]-points[t], axis=-1) / np.maximum(np.minimum(depth[s], depth[t]), 1e-8)
            a *= np.exp(-(distance / sigma_xyz)**2)
        if kind == "normal":
            ns, nt = normal[s], normal[t]
            denom = np.linalg.norm(ns, axis=-1) * np.linalg.norm(nt, axis=-1)
            cosine = np.sum(ns*nt, axis=-1)/np.maximum(denom, 1e-12)
            # Unreliable normals do not create spurious links.
            a *= np.where((denom > 1e-8) & np.isfinite(cosine), np.maximum(cosine, 0)**config.get("normal_power", 4.), 0.)
        keep = np.isfinite(a) & (a > config.get("min_edge_weight", 1e-8))
        ii.append(ids[s][keep]); jj.append(ids[t][keep]); aa.append(a[keep])
    return GeometryGraph(np.concatenate(ii), np.concatenate(jj), np.concatenate(aa), h*w)
