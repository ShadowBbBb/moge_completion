from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from zsdc.data.sampling import sample_sensor
from zsdc.solvers.align import align_depth
from zsdc.solvers.correction import GeometryCompletion
from zsdc.solvers.graph import build_graph
from zsdc.solvers.pcg import solve_system


def test_scale_recovery(scene):
    sample, _, _ = scene
    aligned, info = align_depth(sample.sensor_depth/3, sample, "scale")
    np.testing.assert_allclose(aligned, sample.sensor_depth, rtol=1e-6)
    assert info["scale"] == pytest.approx(3, rel=1e-6)


def test_affine_recovery(scene):
    sample, _, _ = scene
    aligned, info = align_depth((sample.sensor_depth-.4)/2, sample)
    np.testing.assert_allclose(aligned, sample.sensor_depth, rtol=1e-6)
    assert info["mode"] == "scale_shift"


@pytest.mark.parametrize("count", [0,1])
def test_alignment_fallback(scene, count):
    sample, _, prior = scene
    sample = sample_sensor(sample, "random", count, 0)
    aligned, info = align_depth(prior.depth, sample)
    assert info["fallback"] is not None
    assert np.isfinite(aligned).all() and np.all(aligned > 0)


def test_degenerate_affine_and_nonpositive_affine(scene):
    sample, _, _ = scene
    _, info = align_depth(np.ones(sample.sensor_depth.shape), sample)
    assert "degenerate" in info["fallback"]
    # Observed affine fit can be positive at anchors but negative elsewhere.
    x = np.linspace(.1, 3, sample.sensor_depth.size).reshape(sample.sensor_depth.shape)
    mask = x > 2
    sensor = np.where(mask, x-1, 0)
    aligned, info = align_depth(x, replace(sample, sensor_depth=sensor, sensor_mask=mask))
    assert info["fallback"] == "nonpositive_or_invalid_affine_depth"
    assert np.all(aligned > 0)


def test_graph_sparse_symmetric_psd(scene):
    sample, _, prior = scene
    graph = build_graph(sample.image, prior.depth, sample.K, "xyz")
    L = graph.laplacian()
    np.testing.assert_allclose((L-L.T).toarray(), 0)
    np.testing.assert_allclose(L @ np.ones(L.shape[0]), 0, atol=1e-14)
    assert L.nnz <= 9*prior.depth.size
    assert np.all(graph.weights >= 0)
    x = np.random.default_rng(0).normal(size=L.shape[0])
    assert x @ (L@x) >= 0


@pytest.mark.parametrize("kind", ["rgb", "depth", "xyz", "normal"])
def test_graph_kinds(scene, kind):
    sample, _, prior = scene
    normal = np.broadcast_to([0.,0.,1.], (*prior.depth.shape,3)).copy()
    graph = build_graph(sample.image, prior.depth, sample.K, kind, normal)
    assert len(graph.weights) > 0


def test_depth_break_blocks_propagation():
    from zsdc.data.schema import CompletionInput
    from zsdc.models.output_types import GeometryPrediction
    h,w = 8, 12
    image = np.ones((h,w,3))*.5
    depth = np.ones((h,w)); depth[:,w//2:] = 4
    sensor = np.zeros_like(depth); sensor[4,2] = 1.5
    sample = CompletionInput(image, sensor, sensor>0, np.eye(3))
    prior = GeometryPrediction(depth, np.zeros((h,w,3)), np.eye(3), np.ones((h,w),bool))
    graph = build_graph(image, depth, np.eye(3), "depth")
    assert not np.any((graph.i%w < w//2) & (graph.j%w >= w//2))
    result = GeometryCompletion("graph3d", {"align_mode":"none", "graph":{"kind":"depth"}, "sensor_loss":"l2"}).complete(sample, prior)
    np.testing.assert_allclose(result.depth[:,w//2:], 4, atol=1e-9)
    assert result.depth[4,2] > 1.1


def test_cg_direct_and_residual(scene):
    sample, _, prior = scene
    L = build_graph(sample.image, prior.depth, sample.K).laplacian()
    A = L+sparse.eye(L.shape[0])*.1
    b = np.random.default_rng(1).normal(size=L.shape[0])
    x, info = solve_system(A, b)
    direct, _ = solve_system(A, b, {"solver":"direct"})
    assert info["converged"] and info["relative_residual"] < 1e-6
    np.testing.assert_allclose(x, direct, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("method", ["rgb_only","scale","scale_shift","graph2d","graph3d"])
def test_no_points_identity(scene, method):
    sample, _, prior = scene
    sample = replace(sample, sensor_mask=np.zeros_like(sample.sensor_mask))
    result = GeometryCompletion(method).complete(sample, prior)
    np.testing.assert_allclose(result.depth, prior.depth)


def test_solver_failure_falls_back(scene):
    sample, _, prior = scene
    sample = sample_sensor(sample, "random", 8, 0)
    aligned, _ = align_depth(prior.depth, sample)
    result = GeometryCompletion("graph2d", {"cg":{"maxiter":1,"rtol":1e-14}}).complete(sample, prior)
    assert result.metadata["fallback"] == "solver_not_converged"
    np.testing.assert_allclose(result.depth, aligned)


def test_huber_is_more_robust_than_l2():
    from zsdc.data.schema import CompletionInput
    from zsdc.models.output_types import GeometryPrediction
    shape = (16,16)
    depth = np.ones(shape)*2
    sensor = depth.copy(); sensor[8,8] = 20
    sample = CompletionInput(np.ones((*shape,3))*.5, sensor, np.ones(shape,bool), np.eye(3))
    prior = GeometryPrediction(depth, np.zeros((*shape,3)), np.eye(3), np.ones(shape,bool))
    cfg = {"align_mode":"none", "lambda_graph":1., "huber_delta":.05}
    l2 = GeometryCompletion("graph2d", {**cfg,"sensor_loss":"l2"}).complete(sample,prior)
    huber = GeometryCompletion("graph2d", {**cfg,"sensor_loss":"huber"}).complete(sample,prior)
    assert np.mean((huber.depth-depth)**2) < np.mean((l2.depth-depth)**2)


def test_log_plane_scale_correction_and_clipping(scene):
    sample, _, prior = scene
    prior = replace(prior, depth=sample.sensor_depth/2)
    cfg = {"align_mode":"none","sensor_loss":"l2","lambda_prior":1e-8, "lambda_graph":1.}
    result = GeometryCompletion("graph2d",cfg).complete(sample,prior)
    np.testing.assert_allclose(result.depth, sample.sensor_depth, rtol=1e-6)
    result = GeometryCompletion("graph2d",{**cfg,"max_abs_log_correction":.01}).complete(sample,prior)
    assert result.metadata["clipped_corrections"] == prior.depth.size
    assert np.isfinite(result.depth).all() and np.all(result.depth>0)


def test_sensor_on_false_prediction_mask_and_hard_copy(scene):
    sample, _, prior = scene
    sample = sample_sensor(sample,"random",8,0)
    prior = replace(prior,pred_mask=np.zeros_like(prior.pred_mask))
    result = GeometryCompletion("graph3d",{"hard_copy_sensor":True}).complete(sample,prior)
    np.testing.assert_array_equal(result.depth[sample.sensor_mask],sample.sensor_depth[sample.sensor_mask])
    assert result.metadata["hard_copy_sensor"]


def test_gt_change_cannot_affect_algorithm(scene):
    sample, target, prior = scene
    before = GeometryCompletion("graph3d").complete(sample,prior).depth
    target.gt_depth[:] = np.nan
    after = GeometryCompletion("graph3d").complete(sample,prior).depth
    np.testing.assert_array_equal(before,after)


def test_confidence_and_large_prior_regularization(scene):
    sample,_,prior = scene
    sample = sample_sensor(sample,"random",32,0)
    cfg = {"align_mode":"none","lambda_prior":1e12,"sensor_loss":"l2"}
    result = GeometryCompletion("graph2d",cfg).complete(sample,prior)
    np.testing.assert_allclose(result.depth,prior.depth,rtol=1e-10)
    # Rejected confidence still means the location is observed for evaluation.
    zero_conf = replace(sample,sensor_confidence=np.zeros_like(sample.sensor_depth))
    result = GeometryCompletion("graph2d").complete(zero_conf,prior)
    np.testing.assert_allclose(result.depth,prior.depth)
    assert zero_conf.sensor_mask.sum() == 32
