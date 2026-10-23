from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .geometry import range_to_z, resize_K
from .schema import CompletionInput, EvaluationTarget, validate_K


@dataclass(frozen=True)
class Record:
    rgb: str
    raw: str
    gt: str
    scene: str
    sample_id: str


def read_list(filename):
    records, seen = [], set()
    for line_no, line in enumerate(Path(filename).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{filename}:{line_no}: expected RGB/raw/GT columns")
        paths = [PurePosixPath(f.replace("\\", "/")) for f in fields]
        if any(p.is_absolute() or ".." in p.parts or ":" in str(p) for p in paths):
            raise ValueError("Lists must contain paths relative to data_root")
        rgb, raw, gt = map(str, paths)
        if (not rgb.endswith("_rgb.left.jpg") or raw != rgb.replace("_rgb.left.jpg", "_rawdepth.left.png")
                or gt != rgb.replace("_rgb.left.jpg", "_depth_left.png")):
            raise ValueError(f"Unpaired RobbySimVal row at {filename}:{line_no}")
        parts = paths[0].parts
        if len(parts) < 5 or parts[-4] != "val_view" or parts[-5] != "RobbySimVal":
            raise ValueError(f"Unexpected RobbySimVal path: {rgb}")
        if rgb in seen:
            raise ValueError(f"Duplicate sample: {rgb}")
        seen.add(rgb)
        records.append(Record(rgb, raw, gt, parts[-3], rgb))
    if not records:
        raise ValueError(f"Empty list: {filename}")
    return records


def check_splits(train, val):
    overlap = {r.scene for r in train} & {r.scene for r in val}
    if overlap:
        raise ValueError(f"Train/val scene overlap: {sorted(overlap)}")


def select_records(records, limit=None):
    # Deterministic round-robin across scenes: no GT or per-image oracle.
    groups = {}
    for record in sorted(records, key=lambda r: r.sample_id):
        groups.setdefault(record.scene, []).append(record)
    selected = []
    for i in range(max(map(len, groups.values()))):
        for scene in sorted(groups):
            if i < len(groups[scene]):
                selected.append(groups[scene][i])
                if limit is not None and len(selected) >= limit:
                    return selected
    return selected


def output_shape(shape, config):
    h, w = shape
    if config.get("original", False):
        return h, w
    if config.get("size"):
        nh, nw = map(int, config["size"])
        if nh <= 0 or nw <= 0:
            raise ValueError("size must be positive [height,width]")
        return nh, nw
    cap = config.get("max_pixels", 90000)
    if cap is None:
        return h, w
    if cap <= 0:
        raise ValueError("max_pixels must be positive")
    scale = min(1., (cap / (h*w)) ** 0.5)
    return max(1, int(h*scale)), max(1, int(w*scale))


def remap_sensor(depth, mask, shape, confidence=None):
    h, w = depth.shape
    nh, nw = shape
    y, x = np.nonzero(mask)
    yy = np.clip(np.floor((y + .5) * nh/h).astype(int), 0, nh-1)
    xx = np.clip(np.floor((x + .5) * nw/w).astype(int), 0, nw-1)
    index = yy*nw + xx
    # Deterministic nearest-Z winner, confidence follows that same winner.
    order = np.lexsort((y*w+x, depth[y, x], index))
    ordered = index[order]
    first = np.r_[True, ordered[1:] != ordered[:-1]] if len(order) else np.array([], bool)
    winners = order[first]
    out = np.zeros(nh*nw, np.float32)
    out[index[winners]] = depth[y[winners], x[winners]]
    conf = None
    if confidence is not None:
        conf = np.zeros(nh*nw, np.float32)
        conf[index[winners]] = confidence[y[winners], x[winners]]
        conf = conf.reshape(shape)
    valid = out > 0
    return out.reshape(shape), valid.reshape(shape), conf, int(len(index)-valid.sum())


def _depth(path, invalid):
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    data = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(path)
    if data.ndim != 2 or data.dtype != np.uint16:
        raise ValueError(f"Expected uint16 millimeter PNG: {path}")
    mask = (data > 0) & ~np.isin(data, invalid)
    stats = {str(v): int((data == v).sum()) for v in (0, 65535)}
    return data.astype(np.float32) * .001, mask, stats, hashlib.sha256(payload).hexdigest()


def load_record(record, root, config, with_target=True):
    root = Path(root)
    image_payload = (root/record.rgb).read_bytes()
    image = cv2.imdecode(np.frombuffer(image_payload,np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(root / record.rgb)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    raw, mask, raw_stats, raw_hash = _depth(root / record.raw, config.get("raw_invalid", [0, 65535]))
    shape = image.shape[:2]
    if raw.shape != shape:
        raise ValueError("Native RGB/raw shape mismatch")
    sidecar = (root / record.rgb).parent / "intrinsics.txt"
    K = np.loadtxt(sidecar, dtype=np.float64) if sidecar.exists() else None
    if K is not None:
        validate_K(K)
    depth_kind = config.get("depth_kind", "z")
    if depth_kind not in ("z", "range"):
        raise ValueError("depth_kind must be z or range")
    if depth_kind == "range":
        if K is None:
            raise ValueError("Range conversion requires calibrated K")
        raw = range_to_z(raw, K)
    target = None
    if with_target:
        gt, valid, gt_stats, gt_hash = _depth(root / record.gt, config.get("gt_invalid", [0, 65535]))
        if gt.shape != shape:
            raise ValueError("Native RGB/GT shape mismatch")
        gt_kind = config.get("gt_depth_kind", depth_kind)
        if gt_kind not in ("z", "range"):
            raise ValueError("gt_depth_kind must be z or range")
        if gt_kind == "range":
            if K is None:
                raise ValueError("GT range conversion requires calibrated K")
            gt = range_to_z(gt, K)
        target = EvaluationTarget(gt, valid)
    new_shape = output_shape(shape, config.get("resolution", {}))
    raw, mask, _, collisions = remap_sensor(raw, mask, new_shape)
    if new_shape != shape:
        image = cv2.resize(image, new_shape[::-1], interpolation=cv2.INTER_AREA)
        if K is not None:
            K = resize_K(K, shape, new_shape)
        if target is not None:
            # Explicit pixel-center nearest mapping for target and its independent mask.
            iy = np.clip(np.floor((np.arange(new_shape[0])+.5)*shape[0]/new_shape[0]).astype(int), 0, shape[0]-1)
            ix = np.clip(np.floor((np.arange(new_shape[1])+.5)*shape[1]/new_shape[1]).astype(int), 0, shape[1]-1)
            target = EvaluationTarget(target.gt_depth[np.ix_(iy, ix)], target.gt_valid_mask[np.ix_(iy, ix)])
    metadata = {"scene": record.scene, "original_shape": list(shape), "shape": list(new_shape),
                "collisions": collisions, "raw_sentinels": raw_stats, "unit": "meter", "depth_kind": "z",
                "source_depth_kind": depth_kind, "K_source": str(sidecar.relative_to(root)) if K is not None else None,
                "rgb_file_sha256": hashlib.sha256(image_payload).hexdigest(), "raw_file_sha256": raw_hash,
                "K_file_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest() if K is not None else None,
                "input_origin": "published_rawdepth", "preprocessing_version": 1}
    sample = CompletionInput(image, raw, mask, K, sample_id=record.sample_id, metadata=metadata)
    # GT diagnostics are deliberately kept outside CompletionInput metadata.
    return sample, target, ({"gt_sentinels": gt_stats, "gt_file_sha256": gt_hash} if with_target else {})
