from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import yaml

from ..data.adapters import check_splits, load_record, read_list, select_records
from ..data.sampling import sample_sensor
from ..models.cache import cache_key, load_prediction, save_prediction
from ..models.moge_adapter import CachedMoGeAdapter, MoGeAdapter, audit_sample
from ..solvers.correction import GeometryCompletion
from ..utils import environment, fingerprint, peak_rss_bytes, sha256_file, source_identity, write_json
from .metrics import evaluate_depth
from .report import StreamingReport
from .visualize import panels, plot_summary


def dataset(config):
    data = config["data"]
    for key in ("train_list", "val_list"):
        if not data.get(key):
            raise ValueError(f"data.{key} is required for split isolation audit")
    train, val = read_list(data["train_list"]), read_list(data["val_list"])
    check_splits(train, val)
    return {"train": train, "val": val}


def preflight(config, limit=None, lists_only=False):
    sets = dataset(config)
    report = {"splits": {name: {"samples": len(rows), "scenes": sorted({r.scene for r in rows}),
                                         "list_sha256": sha256_file(config["data"][name+"_list"])} for name,rows in sets.items()},
              "depth_policy": {k:config["data"][k] for k in ("raw_invalid", "gt_invalid", "depth_kind", "gt_depth_kind")},
              "checked": [], "failures": [], "status": "lists_only" if lists_only else "checked"}
    if lists_only:
        return report
    if not config["data"].get("root"):
        raise ValueError("data.root is required for file inspection")
    for split, rows in sets.items():
        for record in select_records(rows, limit):
            try:
                sample, target, diagnostics = load_record(record, config["data"]["root"], config["data"])
                report["checked"].append({"split": split, "sample_id": record.sample_id, **sample.metadata, **diagnostics,
                                           "sensor_points": int(sample.sensor_mask.sum()), "gt_valid": int(target.gt_valid_mask.sum())})
            except Exception as exc:
                report["failures"].append({"sample_id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"})
    if report["failures"]:
        report["status"] = "failed"
    return report


def start_run(config, action):
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty; choose a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output/"run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {"action": action, "status": "running", "config_sha256": fingerprint(config),
                "source": source_identity(), "environment": environment(),
                "depth_policy": {k:config["data"][k] for k in ("raw_invalid", "gt_invalid", "depth_kind", "gt_depth_kind")},
                "lists": {key: {"path": config["data"].get(key), "sha256": sha256_file(config["data"][key]) if config["data"].get(key) else None} for key in ("train_list", "val_list")},
                "models": []}
    write_json(output/"manifest.json", manifest)
    return output, manifest


def audit_models(config):
    if config.get("cache_only"):
        raise ValueError("audit_model requires the actual MoGe runtime, not --cache-only")
    sets = dataset(config)
    records = select_records(sets[config["data"]["split"]], config["audit"]["limit"])
    audit_root = Path(config["audit_dir"])
    audit_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for model_config in config["models"]:
        version = model_config["version"]
        path = audit_root/version
        path.mkdir(parents=True, exist_ok=True)
        # Invalidate stale passing attestations before attempting another audit.
        report = {"version": version, "passed": False, "status": "running", "samples": [], "failures": [],
                  "fov_protocols": [], "environment": environment(), "config": config}
        write_json(path/"audit.json", report)
        try:
            adapter = MoGeAdapter(model_config)
            report.update(base_identity=adapter.base_identity, base_identity_hash=fingerprint(adapter.base_identity), runtime=adapter.runtime)
            before = adapter.parameter_hash()
            half = model_config.get("use_fp16", False)
            rtol, atol = (config["audit"]["rtol_fp16" if half else "rtol_fp32"], config["audit"]["atol_fp16" if half else "atol_fp32"])
            report.update(rtol=rtol, atol=atol, parameters_before=before)
            for index, record in enumerate(records):
                try:
                    sample, _, _ = load_record(record, config["data"]["root"], config["data"], with_target=False)
                    logs = audit_sample(adapter, sample, sorted(set(config["audit"]["steps"] + [model_config.get("refine_steps", 3)])), rtol, atol)
                    for log in logs:
                        log["input_metadata"] = sample.metadata
                    report["samples"].extend(logs)
                    prediction = adapter.predict(sample)
                    from ..data.schema import CompletionResult
                    panels(path/f"sample_{index:03}.png", sample, prediction, CompletionResult(prediction.depth))
                except Exception as exc:
                    report["failures"].append({"sample_id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"})
            after = adapter.parameter_hash()
            report["parameters_after"] = after
            if before != after:
                report["failures"].append({"error": "Model state hash changed during inference"})
            report["fov_protocols"] = sorted({log["fov_protocol"] for log in report["samples"]})
            report["passed"] = bool(report["samples"] and not report["failures"])
            report["status"] = "passed" if report["passed"] else "failed"
            del adapter
        except Exception as exc:
            report["status"] = "blocked"
            report["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
        write_json(path/"audit.json", report)
        (path/"baseline_audit.md").write_text(f"# {version} RGB-only audit\n\nStatus: **{report['status']}**\n\n"
            f"Per-step checks: {len(report['samples'])}; failures: {len(report['failures'])}.\n\n"
            "See audit.json for numeric differences, tolerances, per-map statistics, environment and state hashes.\n\n"
            + "\n".join(f"- {f['error']}" for f in report["failures"])+"\n", encoding="utf-8")
        reports.append(report)
    return reports


def sampling_conditions(config, full=True):
    if not full:
        return [("native", 0, 0), ("random", 100, 0)]
    sampling = config["sampling"]
    result = []
    for pattern in sampling["patterns"]:
        if pattern == "native":
            result.append((pattern, 0, 0))
        else:
            result.extend((pattern, n, seed) for n in sampling["counts"] for seed in sampling["seeds"])
    return result


def _cached_prior(adapter, sample, config):
    audit_path = Path(config["audit_dir"])/adapter.version/"audit.json"
    if not audit_path.is_file():
        raise RuntimeError(f"Parity audit missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    adapter.verify_audit(audit, sample)
    identity = adapter.identity(sample)
    key = cache_key(sample, identity)
    path = Path(config["cache"])/adapter.version/(key+".npz")
    start = time.perf_counter()
    hit = path.exists()
    if hit:
        prior = load_prediction(path)
        if prior.metadata.get("cache_key") != key or fingerprint(prior.metadata.get("identity")) != fingerprint(identity):
            raise RuntimeError(f"Corrupt cache identity: {path}")
    else:
        prior = adapter.predict(sample)
        prior.metadata.update(cache_key=key)
        save_prediction(path, prior)
    return prior, {"cache_hit": hit, "prior_fetch_seconds": time.perf_counter()-start, "cache_key": key}


def run_benchmark(config, action="evaluate"):
    sets = dataset(config)
    records = select_records(sets[config["data"]["split"]], config["evaluation"]["limit"])
    output, manifest = start_run(config, action)
    failures, timing_records = [], []
    manifest["selection"] = {"split":config["data"]["split"], "sample_ids":[r.sample_id for r in records], "policy":"scene_round_robin_sorted_ids"}
    manifest["coverage"] = {}
    report_writer = StreamingReport(output) if action == "evaluate" else None
    condition_count = 0
    start = time.perf_counter()
    images_written = 0
    for model_config in config["models"]:
        version = model_config["version"]
        full = model_config.get("full_matrix", version == "v3")
        selected = records if full else records[:config["evaluation"]["v2_smoke_limit"]]
        manifest["coverage"][version] = {"planned_priors":len(selected),"processed_priors":0,
                                        "planned_conditions":0 if action == "cache" else len(selected)*len(sampling_conditions(config,full))*len(config["methods"]),
                                        "processed_conditions":0}
        try:
            audit_file = Path(config["audit_dir"])/version/"audit.json"
            audit = json.loads(audit_file.read_text(encoding="utf-8")) if config.get("cache_only") else None
            adapter = CachedMoGeAdapter(model_config, audit) if config.get("cache_only") else MoGeAdapter(model_config)
            manifest["models"].append({"identity": adapter.base_identity, "runtime": adapter.runtime})
            write_json(output/"manifest.json", manifest)
            # Fail the model before expensive I/O if its attestation is missing/mismatched.
            audit_file = Path(config["audit_dir"])/version/"audit.json"
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            if not audit.get("passed") or audit.get("base_identity_hash") != fingerprint(adapter.base_identity):
                raise RuntimeError("Passing parity audit for the exact runtime/config/source is required")
        except Exception as exc:
            failures.append({"model": version, "stage": "model_setup", "status": "blocked", "error": f"{type(exc).__name__}: {exc}"})
            continue
        for index, record in enumerate(selected):
            sample_start = time.perf_counter()
            try:
                sample, target, diagnostics = load_record(record, config["data"]["root"], config["data"], with_target=action == "evaluate")
                load_seconds = time.perf_counter()-sample_start
                prior, fetch = _cached_prior(adapter, sample, config)
                timing_records.append({"model": version, "sample_id": record.sample_id, **fetch, "data_metadata":sample.metadata,
                                       "load_seconds": load_seconds, "model_seconds": prior.metadata["model_seconds"],
                                       "peak_cuda_bytes": prior.metadata.get("peak_cuda_bytes"), **diagnostics})
                manifest["coverage"][version]["processed_priors"] += 1
            except Exception as exc:
                failures.append({"model": version, "sample_id": record.sample_id, "stage": "data_or_prior", "error": f"{type(exc).__name__}: {exc}"})
                continue
            if action == "cache":
                continue
            for pattern, count, seed in sampling_conditions(config, full):
                condition_start = time.perf_counter()
                observed = sample_sensor(sample, pattern, count, seed)
                sampling_seconds = time.perf_counter()-condition_start
                for method in config["methods"]:
                    method_start = time.perf_counter()
                    base = {"model": version, "sample_id": record.sample_id, "pattern": pattern,
                            "requested_points": None if pattern == "native" else count, "seed": seed, "method": method,
                            "actual_points": int(observed.sensor_mask.sum()), "observation_hash": observed.metadata["observation_hash"],
                            "collisions": sample.metadata["collisions"], "cache_hit": fetch["cache_hit"],
                            "model_seconds": prior.metadata["model_seconds"], "prior_fetch_seconds": fetch["prior_fetch_seconds"],
                            "load_seconds": load_seconds, "sampling_seconds": sampling_seconds,
                            "fov_protocol": prior.metadata["fov_protocol"], "peak_cuda_bytes": prior.metadata.get("peak_cuda_bytes")}
                    try:
                        result = GeometryCompletion(method, config["completion"]).complete(observed, prior)
                        metrics = evaluate_depth(result.depth, observed, target, config["evaluation"]) if target is not None else {}
                        status = "invalid_prediction" if any(v["invalid_predictions"] for v in metrics.values()) else "ok"
                        row = {**base, "status": status, "fallback": result.metadata.get("fallback"), "metrics": metrics,
                               "completion_metadata": result.metadata}
                        if status != "ok":
                            failures.append({**base, "stage": "metrics", "error": "Invalid predictions on fixed GT mask"})
                        if action == "complete":
                            name = fingerprint({k:base[k] for k in ("model", "sample_id", "pattern", "requested_points", "seed", "method")})
                            np.savez_compressed(output/(name+".npz"), depth=result.depth, metadata_json=np.array(__import__("json").dumps({**base, **result.metadata})))
                        if images_written < config["evaluation"]["visualize_limit"]:
                            panels(output/"visualizations"/f"{images_written:03}_{version}_{method}.png", observed, prior, result, target)
                            images_written += 1
                    except Exception as exc:
                        failures.append({**base, "stage": "completion", "error": f"{type(exc).__name__}: {exc}"})
                        row = {**base, "status": "failed", "fallback": None,
                               "metrics": evaluate_depth(None, observed, target, config["evaluation"]) if target is not None else {}}
                    completion_seconds = time.perf_counter()-method_start
                    # Visualization/serialization are included in condition wall time, but not algorithm time.
                    algorithm_seconds = row.get("completion_metadata", {}).get("completion_seconds", completion_seconds)
                    row.update(completion_seconds=algorithm_seconds, condition_wall_seconds=completion_seconds+sampling_seconds,
                               reconstructed_uncached_seconds=load_seconds+sampling_seconds+prior.metadata.get("prediction_wall_seconds", prior.metadata["model_seconds"])+algorithm_seconds,
                               peak_cpu_rss_bytes=peak_rss_bytes())
                    condition_count += 1
                    manifest["coverage"][version]["processed_conditions"] += 1
                    if report_writer is not None:
                        report_writer.add(row)
            # Save partial results after every image for recoverable batch diagnostics.
            if action == "evaluate" and (index+1) % 32 == 0:
                report_writer.snapshot(failures, timing_records)
        del adapter
    if action == "evaluate":
        summary = report_writer.snapshot(failures, timing_records)
        report_writer.close()
        plot_summary(output/"accuracy_runtime.png", summary)
    else:
        write_json(output/"timings.json", timing_records)
        (output/"failures.jsonl").write_text("".join(json.dumps(f)+"\n" for f in failures), encoding="utf-8")
    manifest.update(status="completed_with_failures" if failures else "completed", failures=len(failures),
                    evaluated_conditions=condition_count, processed_priors=len(timing_records),
                    wall_seconds=time.perf_counter()-start, peak_cpu_rss_bytes=peak_rss_bytes())
    write_json(output/"manifest.json", manifest)
    if action == "evaluate":
        with open(output/"summary.md","a",encoding="utf-8") as f:
            f.write("\nCoverage (data/prior failures cannot enter metric aggregation; inspect failures.jsonl):\n\n")
            for version, coverage in manifest["coverage"].items():
                f.write(f"- {version}: {coverage['processed_priors']}/{coverage['planned_priors']} priors, {coverage['processed_conditions']}/{coverage['planned_conditions']} conditions.\n")
    return manifest


def run_sweep(config):
    from ..config import set_value
    reports = []
    for index, overrides in enumerate(config["sweep"]):
        case = deepcopy(config)
        for key, value in overrides.items():
            set_value(case, key, value)
        case["output"] = str(Path(config["output"])/f"sweep_{index:03}")
        reports.append(run_benchmark(case))
    return reports
