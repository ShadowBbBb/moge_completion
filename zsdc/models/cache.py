import os
from pathlib import Path
import tempfile

import numpy as np

from ..utils import canonical, fingerprint
from .output_types import GeometryPrediction


def cache_key(sample, identity):
    import hashlib
    # Intentionally no sensor arrays, sampling metadata, or GT in cache identity.
    return fingerprint({"model": identity, "rgb_sha256": hashlib.sha256(np.ascontiguousarray(sample.image).tobytes()).hexdigest(),
                        "rgb_shape": list(sample.image.shape), "rgb_dtype": str(sample.image.dtype),
                        "preprocessing_version": sample.metadata.get("preprocessing_version", 1)})


def save_prediction(path, prediction):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: getattr(prediction, key) for key in ("depth", "points", "intrinsics", "pred_mask", "normal") if getattr(prediction, key) is not None}
    for key in ("depth_per_step", "points_per_step", "intrinsics_per_step"):
        value = getattr(prediction, key)
        if value is not None:
            arrays[key] = np.stack(value)
    arrays["metadata_json"] = np.array(canonical(prediction.metadata))
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = f.name
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_prediction(path):
    import json
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files if key != "metadata_json"}
        for key in ("depth_per_step", "points_per_step", "intrinsics_per_step"):
            if key in values:
                values[key] = list(values[key])
        values["metadata"] = json.loads(str(archive["metadata_json"]))
    return GeometryPrediction(**values)
