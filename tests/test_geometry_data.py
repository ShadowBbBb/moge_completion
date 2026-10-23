from dataclasses import replace
import importlib
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from zsdc.config import load_config
from zsdc.data.adapters import check_splits, load_record, output_shape, read_list, remap_sensor, select_records
from zsdc.data.geometry import normalized_to_pixel, pixel_to_normalized, project, range_to_z, rays, resize_K, unproject
from zsdc.data.sampling import sample_sensor
from zsdc.data.schema import CompletionInput
from zsdc.evaluation.runner import preflight


def test_camera_roundtrip(scene):
    sample, _, _ = scene
    shape = sample.sensor_depth.shape
    np.testing.assert_allclose(normalized_to_pixel(pixel_to_normalized(sample.K, shape), shape), sample.K)
    np.testing.assert_allclose(resize_K(resize_K(sample.K, shape, (12, 16)), (12, 16), shape), sample.K)
    coords = project(unproject(sample.sensor_depth, sample.K), sample.K)
    y, x = np.indices(shape)
    np.testing.assert_allclose(coords, np.stack([x, y], -1), atol=1e-10)


def test_range_z(scene):
    sample, _, _ = scene
    length = np.linalg.norm(rays(sample.sensor_depth.shape, sample.K), axis=-1)
    np.testing.assert_allclose(range_to_z(sample.sensor_depth*length, sample.K), sample.sensor_depth)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., 0.])
def test_invalid_sensor_not_observation(scene, bad):
    sample, _, _ = scene
    depth = sample.sensor_depth.copy(); depth[0, 0] = bad
    sample = replace(sample, sensor_depth=depth)
    assert not sample.sensor_mask[0, 0]


def test_bad_confidence_rejected(scene):
    sample, _, _ = scene
    with pytest.raises(ValueError):
        replace(sample, sensor_confidence=np.full(sample.sensor_depth.shape, 2.))


def test_png_units_sentinels_and_split(triplets):
    root, lists = triplets
    train, val = read_list(lists["train"]), read_list(lists["val"])
    check_splits(train, val)
    sample, target, diag = load_record(val[0], root, {})
    assert sample.sensor_depth[1, 1] == 2.
    assert sample.sensor_mask.sum() == 94
    assert target.gt_valid_mask.sum() == 94
    assert diag["gt_sentinels"] == {"0": 1, "65535": 1}
    assert "gt_sentinels" not in sample.metadata
    with pytest.raises(ValueError, match="overlap"):
        check_splits(train, train)


def test_gt_not_read_for_prediction(triplets):
    root, lists = triplets
    record = read_list(lists["val"])[0]
    (root/record.gt).unlink()
    sample, target, diag = load_record(record, root, {}, with_target=False)
    assert target is None and diag == {} and sample.sensor_mask.any()
    with pytest.raises(FileNotFoundError):
        load_record(record, root, {})


def test_depth_type_and_alignment_errors(triplets):
    root, lists = triplets
    record = read_list(lists["val"])[0]
    cv2.imwrite(str(root/record.raw), np.ones((8,12), np.uint8))
    with pytest.raises(ValueError, match="uint16"):
        load_record(record, root, {})
    cv2.imwrite(str(root/record.raw), np.ones((4,6), np.uint16))
    with pytest.raises(ValueError, match="shape"):
        load_record(record, root, {})


def test_range_requires_K_and_gt_kind_is_independent(triplets):
    root,lists = triplets
    record = read_list(lists["val"])[0]
    z,_,_ = load_record(record,root,{})
    ranged,gt,_ = load_record(record,root,{"depth_kind":"range","gt_depth_kind":"z"})
    assert ranged.sensor_depth[-1,-1] < z.sensor_depth[-1,-1]
    assert gt.gt_depth[-1,-1] == 2.
    (root/record.rgb).parent.joinpath("intrinsics.txt").unlink()
    no_k,_,_ = load_record(record,root,{})
    assert no_k.K is None
    with pytest.raises(ValueError,match="requires calibrated K"):
        load_record(record,root,{"depth_kind":"range"})


@pytest.mark.parametrize("line", ["a b", "../a ../b ../c", "/a /b /c", "C:/a C:/b C:/c", "a b c"])
def test_bad_list(tmp_path, line):
    file = tmp_path/"list.txt"; file.write_text(line)
    with pytest.raises(ValueError):
        read_list(file)


def test_duplicate_list(triplets):
    _, lists = triplets
    f = lists["val"]; f.write_text(f.read_text()*2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_list(f)


def test_resize_sensor_zbuffer():
    depth = np.array([[3., 1.], [4., 2.]])
    out, mask, conf, collisions = remap_sensor(depth, np.ones((2,2), bool), (1,1), depth/4)
    assert collisions == 3 and mask[0,0] and out[0,0] == 1. and conf[0,0] == .25


def test_resize_sparse_no_interpolation():
    depth = np.zeros((4,4)); depth[0,0] = 2
    out, mask, _, collisions = remap_sensor(depth, depth > 0, (8,8))
    assert mask.sum() == 1 and (out > 0).sum() == 1 and collisions == 0


@pytest.mark.parametrize("cfg,expected", [({}, (259,346)), ({"original": True}, (480,640)),
                                         ({"size": [100,200]}, (100,200)), ({"max_pixels": 999999}, (480,640))])
def test_output_shape(cfg, expected):
    assert output_shape((480,640), cfg) == expected


@pytest.mark.parametrize("pattern", ["native", "random", "grid"])
def test_sampling_deterministic_and_raw_only(scene, pattern):
    sample, _, _ = scene
    a = sample_sensor(sample, pattern, 32, 1)
    b = sample_sensor(sample, pattern, 32, 1)
    np.testing.assert_array_equal(a.sensor_mask, b.sensor_mask)
    assert a.sensor_mask.sum() == (sample.sensor_mask.sum() if pattern == "native" else 32)
    assert np.all(a.sensor_depth[a.sensor_mask] == sample.sensor_depth[a.sensor_mask])
    assert np.all(a.sensor_depth[~a.sensor_mask] == 0)


def test_insufficient_points_and_nested_random(scene):
    sample, _, _ = scene
    a = sample_sensor(sample, "random", 8, 0)
    b = sample_sensor(sample, "random", 32, 0)
    assert np.all(~a.sensor_mask | b.sensor_mask)
    assert sample_sensor(a, "random", 100, 0).sensor_mask.sum() == 8
    empty = replace(sample, sensor_mask=np.zeros_like(sample.sensor_mask))
    assert not sample_sensor(empty, "grid", 100, 0).sensor_mask.any()


def test_preflight_read_only(triplets):
    root, lists = triplets
    cfg = load_config()
    cfg["data"].update(root=str(root), train_list=str(lists["train"]), val_list=str(lists["val"]))
    before = sorted(p.as_posix() for p in root.rglob("*"))
    report = preflight(cfg, limit=1)
    assert not report["failures"] and len(report["checked"]) == 2
    assert before == sorted(p.as_posix() for p in root.rglob("*"))


def test_cpu_imports_without_torch():
    script = "import sys; import zsdc.evaluation.runner; import zsdc.scripts.synthetic_smoke; assert 'torch' not in sys.modules; assert 'moge.model.v3' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)
