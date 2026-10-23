"""The only CPU/GPU boundary. Importing this module does not import torch."""
import hashlib
from pathlib import Path
import time
import warnings

import numpy as np

from ..data.geometry import horizontal_fov
from ..utils import environment, fingerprint, sha256_file, source_identity
from .output_types import GeometryPrediction


class MoGeAdapter:
    def __init__(self, config):
        import torch
        from moge.model import import_model_class_by_version

        self.torch = torch
        self.config = dict(config)
        self.version = config.get("version", "v3")
        if self.version not in ("v2", "v3"):
            raise ValueError("Only MoGe v2/v3 are supported")
        checkpoint = config.get("checkpoint")
        if not checkpoint:
            raise ValueError("Explicit checkpoint path or Hugging Face model id required")
        checkpoint_path = Path(checkpoint).expanduser()
        revision = config.get("revision")
        if not checkpoint_path.is_file():
            if checkpoint_path.is_absolute() or checkpoint.endswith((".pt", ".pth")):
                raise FileNotFoundError(checkpoint_path)
            from huggingface_hub import hf_hub_download
            checkpoint_path = Path(hf_hub_download(checkpoint, filename="model.pt", revision=revision,
                                                  local_files_only=config.get("offline", False)))
        device = config.get("device", "cuda:0")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use CPU-compatible model/runtime explicitly")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model = import_model_class_by_version(self.version).from_pretrained(str(checkpoint_path))
        load_warnings = [str(w.message) for w in caught]
        if any("absent from the checkpoint" in w or "random initialization" in w for w in load_warnings):
            raise RuntimeError("Checkpoint has missing parameters: " + " | ".join(load_warnings))
        if self.version == "v3" and config.get("refine_steps", 3) > 0 and not hasattr(self.model, "refiner"):
            raise RuntimeError("MoGe-3 checkpoint has no refiner")
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.base_identity = {
            "version": self.version, "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_name": checkpoint, "revision": revision,
            "source": source_identity(), "environment": environment(),
            "inference": {key: config.get(key, default) for key, default in
                          [("num_tokens", 1200), ("resolution_level", 0), ("refine_steps", 3 if self.version == "v3" else 0),
                           ("use_fp16", False), ("force_projection", True), ("fov_policy", "auto"), ("fov_x", None)]},
        }
        self.base_identity["inference"]["apply_mask"] = False
        if self.version == "v2":
            self.base_identity["inference"]["refine_steps"] = 0
        self.runtime = {"checkpoint_path": str(checkpoint_path), "load_warnings": load_warnings,
                        "torch_cuda": torch.version.cuda, "device": str(self.model.device),
                        "gpu": torch.cuda.get_device_name(self.model.device) if self.model.device.type == "cuda" else None}

    def kwargs(self, sample, refine_steps=None):
        config = self.base_identity["inference"]
        policy = config["fov_policy"]
        if policy not in ("auto", "estimated", "known"):
            raise ValueError("fov_policy must be auto, estimated or known")
        fov = config["fov_x"]
        if policy == "estimated":
            fov = None
        elif fov is None and sample.K is not None:
            fov = horizontal_fov(sample.K, sample.sensor_depth.shape)
        elif fov is None and policy == "known":
            raise ValueError("known FOV requires K or explicit fov_x")
        if fov is not None and (not np.isfinite(fov) or not 0 < fov < 180):
            raise ValueError("fov_x must be in (0,180) degrees")
        kwargs = {key: config[key] for key in ("num_tokens", "resolution_level", "use_fp16", "force_projection", "apply_mask")}
        kwargs["fov_x"] = fov
        if self.version == "v3":
            kwargs.update(refine_steps=config["refine_steps"] if refine_steps is None else refine_steps, return_per_step=True)
        return kwargs

    def identity(self, sample):
        return {**self.base_identity, "actual_infer_kwargs": self.kwargs(sample)}

    def tensor(self, sample):
        return self.torch.from_numpy(np.ascontiguousarray(sample.image.transpose(2, 0, 1))).to(self.model.device)

    def parameter_hash(self):
        h = hashlib.sha256()
        for key, tensor in sorted(self.model.state_dict().items()):
            h.update(key.encode())
            h.update(str(tensor.dtype).encode())
            h.update(tensor.detach().contiguous().cpu().reshape(-1).view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def predict(self, sample, refine_steps=None):
        torch = self.torch
        kwargs = self.kwargs(sample, refine_steps)
        cuda = self.model.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.model.device)
            torch.cuda.reset_peak_memory_stats(self.model.device)
            event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        image = self.tensor(sample)
        if cuda:
            event_start.record(torch.cuda.current_stream(self.model.device))
        with torch.inference_mode():
            output = self.model.infer(image, **kwargs)
        if cuda:
            event_end.record(torch.cuda.current_stream(self.model.device))
            torch.cuda.synchronize(self.model.device)
        model_seconds = time.perf_counter()-start
        def cpu(value):
            return value.detach().cpu().numpy().copy()
        values = {key: cpu(output[key]) for key in ("depth", "points", "intrinsics")}
        values["pred_mask"] = cpu(output["mask"]).astype(bool) if "mask" in output else np.isfinite(values["depth"]) & (values["depth"] > 0)
        for key in ("depth_per_step", "points_per_step", "intrinsics_per_step"):
            if key in output:
                values[key] = [cpu(item) for item in output[key]]
        if "normal" in output:
            values["normal"] = cpu(output["normal"])
        values["metadata"] = {"identity": self.identity(sample), "actual_infer_kwargs": kwargs,
                              "model_seconds": model_seconds,
                              "prediction_wall_seconds": time.perf_counter()-start,
                              "cuda_seconds": event_start.elapsed_time(event_end)/1000 if cuda else None,
                              "peak_cuda_bytes": torch.cuda.max_memory_allocated(self.model.device) if cuda else None,
                              "intrinsics_format": "normalized_pixel_centers", "fov_protocol": "known_fov" if kwargs["fov_x"] is not None else "estimated_fov",
                              "mask_source": "official" if "mask" in output else "finite_positive"}
        return GeometryPrediction(**values)

    def verify_audit(self, audit, sample):
        if not audit.get("passed") or audit.get("base_identity_hash") != fingerprint(self.base_identity):
            raise RuntimeError("Missing/mismatched passing parity audit; run audit_model for this model/config/source")
        mode = "known_fov" if self.kwargs(sample)["fov_x"] is not None else "estimated_fov"
        if mode not in audit.get("fov_protocols", []):
            raise RuntimeError(f"Audit did not cover {mode}")


def compare_arrays(actual, expected, rtol, atol):
    if actual.shape != expected.shape:
        raise AssertionError(f"Parity shape mismatch: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite].astype(float)-expected[finite].astype(float)))) if finite.any() else 0.


class CachedMoGeAdapter(MoGeAdapter):
    """Replay an audited cache on a CPU-only machine without importing torch."""
    def __init__(self, config, audit):
        if not audit.get("passed"):
            raise RuntimeError("Cache-only evaluation requires a passing GPU audit")
        self.config = dict(config)
        self.version = config["version"]
        self.base_identity = audit["base_identity"]
        if self.base_identity["version"] != self.version:
            raise RuntimeError("Cache-only model version differs from audit")
        if fingerprint(self.base_identity) != audit.get("base_identity_hash"):
            raise RuntimeError("Audit identity checksum mismatch")
        if source_identity()["source_sha256"] != self.base_identity["source"]["source_sha256"]:
            raise RuntimeError("Cache was audited with different source code")
        expected = self.base_identity["inference"]
        for key, default in [("num_tokens",1200),("resolution_level",0),("refine_steps",3 if self.version == "v3" else 0),
                             ("use_fp16",False),("force_projection",True),("fov_policy","auto"),("fov_x",None)]:
            if config.get(key,default) != expected[key]:
                raise RuntimeError(f"Cache-only config differs from audit: {key}")
        if config.get("checkpoint") != self.base_identity["checkpoint_name"]:
            raise RuntimeError("Cache-only checkpoint identifier differs from audit")
        checkpoint = Path(config["checkpoint"]).expanduser()
        if checkpoint.is_file() and sha256_file(checkpoint) != self.base_identity["checkpoint_sha256"]:
            raise RuntimeError("Local checkpoint contents differ from audited checkpoint")
        self.runtime = {"cache_only":True, "audited_runtime":audit.get("runtime")}

    def predict(self, sample, refine_steps=None):
        raise FileNotFoundError("RGB-only cache missing; run cache_baseline on the audited GPU runtime first")


def audit_sample(adapter, sample, steps, rtol=1e-5, atol=1e-6):
    logs = []
    torch = adapter.torch
    for step in steps if adapter.version == "v3" else [0]:
        wrapped = adapter.predict(sample, step)
        kwargs = adapter.kwargs(sample, step)
        with torch.inference_mode():
            official = adapter.model.infer(adapter.tensor(sample), **kwargs)
        differences = {}
        for field in ("depth", "points", "intrinsics", "mask", "normal", "depth_per_step", "points_per_step", "intrinsics_per_step"):
            if field == "mask" and field not in official:
                continue
            value = wrapped.pred_mask if field == "mask" else getattr(wrapped, field)
            if field not in official:
                if value is not None:
                    raise AssertionError(f"Unexpected wrapper output: {field}")
                continue
            reference = official[field]
            if isinstance(reference, list):
                if len(reference) != step+1 or value is None or len(value) != len(reference):
                    raise AssertionError("Per-step outputs must have K+1 entries")
                differences[field] = [compare_arrays(a, b.detach().cpu().numpy(), rtol, atol) for a,b in zip(value, reference)]
            else:
                differences[field] = compare_arrays(value, reference.detach().cpu().numpy(), rtol, atol)
        repeated = adapter.predict(sample, step)
        compare_arrays(repeated.depth, wrapped.depth, rtol, atol)
        for field in ("points", "intrinsics", "pred_mask", "normal", "depth_per_step", "points_per_step", "intrinsics_per_step"):
            a, b = getattr(repeated, field), getattr(wrapped, field)
            if isinstance(a,list):
                for aa,bb in zip(a,b):
                    compare_arrays(aa,bb,rtol,atol)
            elif a is not None:
                compare_arrays(a,b,rtol,atol)
        if kwargs["force_projection"]:
            from ..data.geometry import normalized_to_pixel, unproject
            differences["pixel_camera_projection"] = compare_arrays(unproject(wrapped.depth,normalized_to_pixel(wrapped.intrinsics,wrapped.depth.shape)), wrapped.points,rtol,atol)
        depth = wrapped.depth
        positive = depth > 0
        logs.append({"sample_id": sample.sample_id, "refine_steps": step, "differences": differences,
                     "nonfinite_pixels": int((~np.isfinite(depth)).sum()), "nonpositive_pixels": int((depth <= 0).sum()),
                     "positive_nonfinite_pixels": int((positive & ~np.isfinite(depth)).sum()),
                     "finite_min": float(depth[np.isfinite(depth)].min()) if np.isfinite(depth).any() else None,
                     "finite_max": float(depth[np.isfinite(depth)].max()) if np.isfinite(depth).any() else None,
                     "fov_protocol": wrapped.metadata["fov_protocol"]})
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise AssertionError("Unmasked prior is not finite/positive; inspect model/sample before completion")
    if any(p.requires_grad for p in adapter.model.parameters()) or adapter.model.training:
        raise AssertionError("Model is not frozen/eval")
    return logs
