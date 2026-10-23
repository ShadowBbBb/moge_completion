from copy import deepcopy
from pathlib import Path

import yaml


DEFAULTS = {
    "data": {"root": None, "train_list": None, "val_list": None, "split": "val", "raw_invalid": [0, 65535],
             "gt_invalid": [0, 65535], "depth_kind": "z", "gt_depth_kind": "z",
             "resolution": {"max_pixels": 90000, "original": False, "size": None}},
    "models": [{"version": "v3", "checkpoint": None, "device": "cuda:0", "num_tokens": 1200, "resolution_level": 0,
                "refine_steps": 3, "use_fp16": False, "fov_policy": "auto", "fov_x": None, "force_projection": True,
                "full_matrix": True},
               {"version": "v2", "checkpoint": None, "device": "cuda:0", "num_tokens": 1200, "resolution_level": 0,
                "refine_steps": 0, "use_fp16": False, "fov_policy": "auto", "fov_x": None, "force_projection": True,
                "full_matrix": False}],
    "methods": ["rgb_only", "scale", "scale_shift", "graph2d", "graph3d"],
    "sampling": {"patterns": ["native", "random", "grid"], "counts": [8, 32, 100, 500, 1200], "seeds": [0, 1, 2]},
    "completion": {"align_mode": "scale_shift", "min_confidence": 0., "lambda_prior": .001, "lambda_graph": 1.,
                   "sensor_weight": 1., "sensor_loss": "huber", "huber_delta": .1, "irls_iterations": 5,
                   "max_abs_log_correction": 1.3862943611198906, "hard_copy_sensor": False,
                   "alignment": {"iterations": 5, "huber_delta": .1, "min_confidence": 0.},
                   "graph": {"connectivity": 8, "kind": "xyz", "sigma_rgb": .1, "sigma_depth": .1,
                             "sigma_xyz": .05, "max_log_jump": .2, "min_edge_weight": 1e-8, "normal_power": 4.},
                   "cg": {"solver": "cg", "rtol": 1e-6, "maxiter": 1000}},
    "evaluation": {"limit": None, "v2_smoke_limit": 8, "visualize_limit": 8, "edge_log_jump": .1, "edge_band_radius": 2},
    "audit": {"limit": 8, "steps": [0, 1, 3], "rtol_fp32": 1e-5, "atol_fp32": 1e-6,
              "rtol_fp16": 1e-3, "atol_fp16": 1e-3},
    "output": "results/zsdc", "cache": "results/baseline_cache", "audit_dir": "results/model_audit", "cache_only": False,
    "sweep": [{"completion.graph.kind": "depth"}, {"completion.graph.kind": "xyz"}, {"completion.graph.kind": "normal"}],
}


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if key in result and isinstance(result[key], dict) and isinstance(value, dict) else deepcopy(value)
    return result


def set_value(config, dotted_key, value):
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f"Unknown configuration key: {dotted_key}")
        target = target[part]
    if parts[-1] not in target:
        raise ValueError(f"Unknown configuration key: {dotted_key}")
    target[parts[-1]] = value


def load_config(filename=None, overrides=()):
    config = merge(DEFAULTS, yaml.safe_load(Path(filename).read_text(encoding="utf-8")) or {}) if filename else deepcopy(DEFAULTS)
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("--set expects dotted.key=YAML_VALUE")
        set_value(config, key, yaml.safe_load(value))
    validate_config(config)
    return config


def validate_config(config):
    if config["data"]["split"] not in ("train", "val"):
        raise ValueError("split must be train or val")
    if any(n <= 0 for n in config["sampling"]["counts"]):
        raise ValueError("Point budgets must be positive")
    if any(p not in ("native", "random", "grid") for p in config["sampling"]["patterns"]):
        raise ValueError("Unknown sampling pattern")
    if config["completion"]["lambda_prior"] <= 0:
        raise ValueError("lambda_prior must be positive")
    for key in ("limit", "v2_smoke_limit"):
        v = config["evaluation"].get(key)
        if v is not None and v <= 0:
            raise ValueError(f"{key} must be positive")
    if config["audit"]["limit"] <= 0:
        raise ValueError("audit.limit must be positive")
    completion = config["completion"]
    if completion["lambda_graph"] < 0 or completion["sensor_weight"] < 0:
        raise ValueError("Graph and sensor weights must be nonnegative")
    if completion["sensor_loss"] not in ("l2", "huber") or completion["irls_iterations"] < 1 or completion["huber_delta"] <= 0:
        raise ValueError("Invalid sensor loss/IRLS configuration")
    if completion["alignment"]["iterations"] < 1 or completion["alignment"]["huber_delta"] <= 0:
        raise ValueError("Invalid alignment configuration")
    if not 0 <= completion["min_confidence"] <= 1 or not 0 <= completion["alignment"]["min_confidence"] <= 1:
        raise ValueError("Confidence thresholds must be in [0,1]")
    graph = completion["graph"]
    if graph["kind"] not in ("rgb","depth","xyz","normal") or graph["connectivity"] not in (4,8):
        raise ValueError("Invalid graph kind/connectivity")
    if min(graph["sigma_rgb"],graph["sigma_depth"],graph["sigma_xyz"]) <= 0 or graph["max_log_jump"] < 0:
        raise ValueError("Invalid graph bandwidth/edge threshold")
    if config["data"]["depth_kind"] not in ("z","range") or config["data"]["gt_depth_kind"] not in ("z","range"):
        raise ValueError("Depth kind must be z or range")
    allowed = {"rgb_only","scale","scale_shift","graph2d","graph3d"}
    if not config["methods"] or not set(config["methods"]) <= allowed:
        raise ValueError("Unknown or empty completion methods")
    versions = [m["version"] for m in config["models"]]
    if len(set(versions)) != len(versions) or any(v not in ("v2", "v3") for v in versions):
        raise ValueError("Choose at most one configuration per model version per run")
