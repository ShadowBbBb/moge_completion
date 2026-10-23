import numpy as np
from scipy.ndimage import binary_dilation


def edge_band(target, log_jump=.1, radius=2):
    gt, valid = target.gt_depth, target.gt_valid_mask
    edge = np.zeros(gt.shape, bool)
    logd = np.log(np.maximum(gt, 1e-12))
    for axis in (0, 1):
        a, b = [slice(None)]*2, [slice(None)]*2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        jump = valid[a] & valid[b] & (np.abs(logd[a]-logd[b]) > log_jump)
        edge[a] |= jump
        edge[b] |= jump
    if radius > 0:
        edge = binary_dilation(edge, iterations=radius)
    return edge & valid


def region_metrics(prediction, gt, mask):
    n = int(mask.sum())
    values = {"count": n, "invalid_predictions": 0, "sse": 0., "sae": 0., "srel": 0., "delta1_hits": 0,
              "rmse": None, "mae": None, "absrel": None, "delta1": None}
    if not n:
        return values
    if prediction is None:
        values["invalid_predictions"] = n
        return values
    if prediction.shape != gt.shape:
        raise ValueError("Prediction/GT shape mismatch")
    p, g = prediction[mask].astype(np.float64), gt[mask].astype(np.float64)
    bad = ~np.isfinite(p) | (p <= 0)
    values["invalid_predictions"] = int(bad.sum())
    if bad.any():
        # Do NOT shrink GT mask and report deceptively good metrics.
        return values
    error = p-g
    values.update(sse=float(np.sum(error**2)), sae=float(np.sum(np.abs(error))),
                  srel=float(np.sum(np.abs(error)/g)), delta1_hits=int((np.maximum(p/g, g/p) < 1.25).sum()))
    values.update(rmse=float(np.sqrt(values["sse"]/n)), mae=values["sae"]/n,
                  absrel=values["srel"]/n, delta1=values["delta1_hits"]/n)
    return values


def evaluate_depth(prediction, sample, target, config=None):
    config = config or {}
    valid = target.gt_valid_mask
    masks = {"all_valid": valid, "observed_valid": valid & sample.sensor_mask,
             "unobserved_valid": valid & ~sample.sensor_mask,
             "edge_valid": edge_band(target, config.get("edge_log_jump", .1), config.get("edge_band_radius", 2))}
    return {name: region_metrics(prediction, target.gt_depth, mask) for name, mask in masks.items()}


def aggregate_regions(regions):
    n = sum(v["count"] for v in regions)
    invalid = sum(v["invalid_predictions"] for v in regions)
    values = {"pixels": n, "invalid_predictions": invalid, "images": len(regions),
              "nonempty_images": sum(v["count"] > 0 for v in regions)}
    for name in ("rmse", "mae", "absrel", "delta1"):
        nonempty = [v for v in regions if v["count"] > 0]
        values["image_mean_"+name] = (float(np.mean([v[name] for v in nonempty]))
                                             if nonempty and not invalid else None)
    values.update(pixel_rmse=None, pixel_mae=None, pixel_absrel=None, pixel_delta1=None)
    if n and not invalid:
        values.update(pixel_rmse=float(np.sqrt(sum(v["sse"] for v in regions)/n)),
                      pixel_mae=sum(v["sae"] for v in regions)/n,
                      pixel_absrel=sum(v["srel"] for v in regions)/n,
                      pixel_delta1=sum(v["delta1_hits"] for v in regions)/n)
    return values
