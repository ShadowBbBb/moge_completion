import time

import numpy as np
from scipy import sparse

from ..data.geometry import normalized_to_pixel
from ..data.schema import CompletionResult
from .align import align_depth, huber_weights, observation_weights
from .graph import build_graph
from .pcg import solve_system


class GeometryCompletion:
    def __init__(self, method="graph3d", config=None):
        if method not in ("rgb_only", "scale", "scale_shift", "graph2d", "graph3d"):
            raise ValueError(f"Unknown completion method: {method}")
        self.method = method
        self.config = config or {}

    def complete(self, sample, prior):
        start = time.perf_counter()
        if prior.depth.shape != sample.sensor_depth.shape:
            raise ValueError("Prior and sensor shapes differ")
        cfg = self.config
        if self.method == "rgb_only":
            return CompletionResult(prior.depth.copy(), {"method": self.method, "fallback": None,
                                     "completion_seconds": time.perf_counter()-start})
        depth = np.asarray(prior.depth, dtype=np.float64).copy()
        finite = np.isfinite(depth) & (depth > 0)
        # Sensor remains available even when the official prediction mask is false.
        repaired = ~finite & sample.sensor_mask
        depth[repaired] = sample.sensor_depth[repaired]
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError("Prior contains invalid values without sensor support; sample must be reported as failed")
        mode = self.method if self.method in ("scale", "scale_shift") else cfg.get("align_mode", "scale_shift")
        aligned, alignment = align_depth(depth, sample, mode, cfg.get("alignment"))
        meta = {"method": self.method, "alignment": alignment, "fallback": alignment["fallback"],
                "repaired_prior_from_sensor": int(repaired.sum()), "linear_solves": []}
        result = aligned.copy()
        graph_degree = None
        correction, weights = np.zeros(depth.shape), observation_weights(sample, cfg.get("min_confidence", 0.))
        if self.method.startswith("graph") and weights.any():
            lp, lg = cfg.get("lambda_prior", .001), cfg.get("lambda_graph", 1.)
            if lp <= 0 or lg < 0:
                raise ValueError("lambda_prior must be >0 and lambda_graph >=0")
            graph_cfg = cfg.get("graph", {})
            kind = "rgb" if self.method == "graph2d" else graph_cfg.get("kind", "xyz")
            K = sample.K if sample.K is not None else normalized_to_pixel(prior.intrinsics, depth.shape)
            meta["graph_camera"] = "sensor_K" if sample.K is not None else "estimated_K"
            graph = build_graph(sample.image, aligned, K, kind, prior.normal, graph_cfg)
            meta["edges"] = len(graph.weights)
            L = graph.laplacian()
            graph_degree = L.diagonal().reshape(depth.shape)
            e = np.zeros_like(depth)
            observed = weights > 0
            e[observed] = np.log(sample.sensor_depth[observed])-np.log(aligned[observed])
            base_weights = weights.ravel().copy() * cfg.get("sensor_weight", 1.)
            if cfg.get("sensor_weight", 1.) < 0:
                raise ValueError("sensor_weight must be nonnegative")
            loss = cfg.get("sensor_loss", "huber")
            if loss not in ("l2", "huber"):
                raise ValueError("sensor_loss must be l2 or huber")
            rounds = cfg.get("irls_iterations", 5) if loss == "huber" else 1
            if rounds < 1:
                raise ValueError("IRLS iterations must be positive")
            flat_e, r = e.ravel(), np.zeros(depth.size)
            current_weights = base_weights.copy()
            for _ in range(rounds):
                # Robustify against current prior on the FIRST round as well.
                if loss == "huber":
                    current_weights = base_weights * huber_weights(r-flat_e, cfg.get("huber_delta", .1))
                A = lg*L + sparse.diags(current_weights + lp)
                try:
                    r_new, log = solve_system(A, current_weights*flat_e, cfg.get("cg"))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    meta["fallback"] = f"solver_error:{exc}"
                    break
                meta["linear_solves"].append(log)
                if not log["converged"]:
                    meta["fallback"] = "solver_not_converged"
                    break
                r = r_new
            else:
                limit = cfg.get("max_abs_log_correction", float(np.log(4)))
                if limit <= 0:
                    raise ValueError("Correction limit must be positive")
                meta["clipped_corrections"] = int((np.abs(r) > limit).sum())
                correction = np.clip(r.reshape(depth.shape), -limit, limit)
                result = aligned * np.exp(correction)
                weights = current_weights.reshape(depth.shape)
                if not np.isfinite(result).all() or np.any(result <= 0):
                    meta["fallback"] = "invalid_corrected_depth"
                    result, correction = aligned.copy(), np.zeros_like(aligned)
        elif self.method.startswith("graph"):
            meta["fallback"] = "no_usable_observations"
        if cfg.get("hard_copy_sensor", False):
            result[sample.sensor_mask] = sample.sensor_depth[sample.sensor_mask]
            meta["hard_copy_sensor"] = True
        usable = weights > 0
        meta["sensor_log_residual_rmse"] = (float(np.sqrt(np.mean((np.log(result[usable])-np.log(sample.sensor_depth[usable]))**2))) if usable.any() else None)
        meta["completion_seconds"] = time.perf_counter()-start
        return CompletionResult(result, meta, correction, weights, graph_degree)
