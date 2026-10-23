from dataclasses import replace
import json

import numpy as np
import pytest

from zsdc.config import load_config
from zsdc.data.sampling import sample_sensor
from zsdc.evaluation.metrics import aggregate_regions, evaluate_depth, region_metrics
from zsdc.evaluation.runner import sampling_conditions
from zsdc.models.cache import cache_key, load_prediction, save_prediction
from zsdc.models.moge_adapter import compare_arrays
from zsdc.scripts.common import main


def test_metric_masks(scene):
    sample,target,prior = scene
    sample = sample_sensor(sample,"random",32,0)
    metrics = evaluate_depth(target.gt_depth, sample,target)
    assert metrics["all_valid"]["rmse"] == 0
    assert metrics["observed_valid"]["count"] == 32
    assert metrics["unobserved_valid"]["count"] == target.gt_depth.size-32
    assert metrics["observed_valid"]["delta1"] == 1


def test_empty_and_invalid_metrics():
    gt = np.ones((2,2))
    empty = region_metrics(gt,gt,np.zeros((2,2),bool))
    assert empty["rmse"] is None and empty["count"] == 0
    bad = gt.copy(); bad[0,0] = np.nan
    m = region_metrics(bad,gt,np.ones((2,2),bool))
    assert m["count"] == 4 and m["invalid_predictions"] == 1 and m["rmse"] is None
    summary = aggregate_regions([m,region_metrics(gt,gt,np.ones((2,2),bool))])
    assert summary["pixel_rmse"] is None and summary["pixels"] == 8


def test_pixel_and_image_aggregation_differ():
    a = region_metrics(np.array([2.]), np.array([1.]), np.array([True]))
    b = region_metrics(np.ones(3),np.ones(3),np.ones(3,bool))
    summary = aggregate_regions([a,b])
    assert summary["pixel_rmse"] == .5
    assert summary["image_mean_rmse"] == .5
    assert summary["pixel_mae"] == .25 and summary["image_mean_mae"] == .5


def test_corrupted_position_stays_observed(scene):
    sample,target,_ = scene
    sample = sample_sensor(sample,"random",8,0)
    sample.sensor_depth[sample.sensor_mask] *= 10
    metrics = evaluate_depth(target.gt_depth,sample,target)
    assert metrics["observed_valid"]["count"] == 8


def test_cache_roundtrip_no_gt(tmp_path,scene):
    sample,_,prior = scene
    prior = replace(prior,depth_per_step=[prior.depth,prior.depth*2], points_per_step=[prior.points], intrinsics_per_step=[prior.intrinsics],metadata={"source":"synthetic"})
    path = tmp_path/"cache.npz"
    save_prediction(path,prior)
    restored = load_prediction(path)
    np.testing.assert_array_equal(restored.depth,prior.depth)
    np.testing.assert_array_equal(restored.depth_per_step[1],prior.depth*2)
    with np.load(path,allow_pickle=False) as archive:
        assert not any("gt" in field or "sensor" in field for field in archive.files)


def test_cache_identity_is_rgb_model_only(scene):
    sample,target,prior = scene
    identity = {"checkpoint":"abc","refine_steps":3,"preprocessing":1}
    key = cache_key(sample,identity)
    sample2 = sample_sensor(sample,"random",8,2)
    assert cache_key(sample2,identity) == key
    target.gt_depth[:] *= 10
    assert cache_key(sample,identity) == key
    assert cache_key(sample,{**identity,"checkpoint":"def"}) != key
    assert cache_key(sample,{**identity,"refine_steps":0}) != key
    assert cache_key(replace(sample,metadata={**sample.metadata,"preprocessing_version":2}),identity) != key
    assert cache_key(replace(sample,image=np.zeros_like(sample.image)),identity) != key


def test_sampling_matrix():
    conditions = sampling_conditions(load_config())
    assert len(conditions) == 31
    assert conditions.count(("native",0,0)) == 1
    assert sampling_conditions(load_config(),False) == [("native",0,0),("random",100,0)]


def test_dry_run_cli_no_writes(triplets,tmp_path,capsys):
    root,lists = triplets
    out = tmp_path/"output"
    rc = main("evaluate",["--data-root",str(root),"--train-list",str(lists["train"]),"--val-list",str(lists["val"]),"--output",str(out),"--dry-run","--limit","1"])
    assert rc == 0 and not out.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "checked"


def test_config_bad_values():
    with pytest.raises(ValueError):
        load_config(overrides=["completion.lambda_prior=0"])
    with pytest.raises(ValueError):
        load_config(overrides=["unknown.key=1"])


def test_parity_does_not_relax_tolerance():
    with pytest.raises(AssertionError):
        compare_arrays(np.array([1.1]),np.array([1.]),1e-5,1e-6)
