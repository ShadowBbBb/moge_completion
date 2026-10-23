import json
from pathlib import Path

import numpy as np
import pytest

from zsdc.config import load_config
from zsdc.data.adapters import load_record, read_list
from zsdc.data.geometry import pixel_to_normalized, unproject
from zsdc.evaluation.metrics import evaluate_depth
from zsdc.evaluation.report import StreamingReport, write_report
from zsdc.evaluation.runner import run_benchmark
from zsdc.models.moge_adapter import CachedMoGeAdapter, MoGeAdapter
from zsdc.models.output_types import GeometryPrediction
from zsdc.solvers.correction import GeometryCompletion
from zsdc.utils import fingerprint, source_identity, write_json


def test_runner_cpu_mock_cache_and_fairness(triplets, tmp_path, monkeypatch):
    """Wiring test only: fake model is never labeled official GPU parity."""
    root, lists = triplets
    calls = []

    class FakeAdapter:
        def __init__(self, config):
            self.version = config["version"]
            self.base_identity = {"version":self.version,"source":"TEST_FAKE_MODEL"}
            self.runtime = {"fake":True}

        def kwargs(self,sample):
            return {"fov_x":60.}

        def verify_audit(self,audit,sample):
            assert audit["passed"] and audit["base_identity_hash"] == fingerprint(self.base_identity)

        def identity(self,sample):
            return self.base_identity

        def predict(self,sample):
            calls.append(sample.sample_id)
            depth = np.full(sample.sensor_depth.shape,1.5)
            return GeometryPrediction(depth,unproject(depth,sample.K),pixel_to_normalized(sample.K,depth.shape),
                                      np.ones(depth.shape,bool),metadata={"identity":self.identity(sample),"model_seconds":.01,
                                      "prediction_wall_seconds":.02,"fov_protocol":"known_fov"})

    monkeypatch.setattr("zsdc.evaluation.runner.MoGeAdapter",FakeAdapter)
    monkeypatch.setattr("zsdc.evaluation.runner.plot_summary",lambda *args:None)
    cfg = load_config()
    cfg["data"].update(root=str(root),train_list=str(lists["train"]),val_list=str(lists["val"]))
    cfg["models"] = [{"version":"v3","checkpoint":"TEST_FAKE_MODEL"}]
    cfg["sampling"] = {"patterns":["native","random","grid"],"counts":[8,32],"seeds":[0,1]}
    cfg["evaluation"].update(visualize_limit=0,limit=1)
    cfg.update(output=str(tmp_path/"run1"),cache=str(tmp_path/"cache"),audit_dir=str(tmp_path/"audit"))
    audit_path = Path(cfg["audit_dir"])/"v3"
    audit_path.mkdir(parents=True)
    write_json(audit_path/"audit.json",{"passed":True,"base_identity_hash":fingerprint(FakeAdapter(cfg["models"][0]).base_identity)})
    result = run_benchmark(cfg)
    assert result["evaluated_conditions"] == 45 and not result["failures"]
    assert len(calls) == 1  # all 45 conditions reuse one RGB-only forward
    rows = [json.loads(line) for line in (Path(cfg["output"])/"per_image.jsonl").read_text().splitlines()]
    for key in {(r["pattern"],r["requested_points"],r["seed"]) for r in rows}:
        subset = [r for r in rows if (r["pattern"],r["requested_points"],r["seed"]) == key]
        assert len({r["observation_hash"] for r in subset}) == 1
    cfg["output"] = str(tmp_path/"run2")
    result2 = run_benchmark(cfg)
    assert not result2["failures"] and len(calls) == 1
    assert all(json.loads(line)["cache_hit"] for line in (Path(cfg["output"])/"per_image.jsonl").read_text().splitlines())


def test_runner_blocks_missing_audit(triplets,tmp_path,monkeypatch):
    root,lists = triplets
    class Fake:
        def __init__(self,config):
            self.base_identity = {"fake":True}
            self.runtime = {}
    monkeypatch.setattr("zsdc.evaluation.runner.MoGeAdapter",Fake)
    monkeypatch.setattr("zsdc.evaluation.runner.plot_summary",lambda *a:None)
    cfg = load_config()
    cfg["data"].update(root=str(root),train_list=str(lists["train"]),val_list=str(lists["val"]))
    cfg["models"] = [{"version":"v3"}]
    cfg.update(output=str(tmp_path/"run"),audit_dir=str(tmp_path/"no_audit"))
    result = run_benchmark(cfg)
    assert result["failures"] == 1 and result["evaluated_conditions"] == 0
    assert "blocked" in (Path(cfg["output"])/"failures.jsonl").read_text()


def test_attestation_config_and_camera_gate(scene):
    sample,_,_ = scene
    adapter = object.__new__(MoGeAdapter)
    adapter.base_identity = {"test":1}
    adapter.kwargs = lambda sample: {"fov_x":60.}
    audit = {"passed":True,"base_identity_hash":fingerprint(adapter.base_identity),"fov_protocols":["known_fov"]}
    adapter.verify_audit(audit,sample)
    with pytest.raises(RuntimeError):
        adapter.verify_audit({**audit,"base_identity_hash":"wrong"},sample)
    with pytest.raises(RuntimeError):
        adapter.verify_audit({**audit,"fov_protocols":["estimated_fov"]},sample)


def test_streaming_matches_batch(tmp_path,scene):
    sample,target,prior = scene
    rows = []
    for seed in range(2):
        sample.sensor_mask[seed,seed] = False
        result = GeometryCompletion("scale").complete(sample,prior)
        rows.append({"model":"synthetic","pattern":"native","requested_points":None,"seed":0,"method":"scale",
                     "sample_id":str(seed),"status":"ok","fallback":None,"completion_seconds":.1,"model_seconds":.2,
                     "metrics":evaluate_depth(result.depth,sample,target)})
    streaming_dir = tmp_path/"stream"; batch_dir = tmp_path/"batch"
    streaming_dir.mkdir(); batch_dir.mkdir()
    writer = StreamingReport(streaming_dir)
    for row in rows:
        writer.add(row)
    streaming = writer.snapshot([],[]); writer.close()
    batch = write_report(batch_dir,rows,[],[])
    assert len(streaming) == len(batch)
    for a,b in zip(streaming,batch):
        for key in b:
            if isinstance(b[key],float):
                assert a[key] == pytest.approx(b[key])
            else:
                assert a[key] == b[key]


def test_cache_only_is_model_free_and_checks_settings(scene):
    sample,_,_ = scene
    cfg = load_config()["models"][0]
    cfg["checkpoint"] = "UNIT_TEST_FAKE_CHECKPOINT"
    identity = {"version":"v3","checkpoint_name":cfg["checkpoint"],"checkpoint_sha256":"test_only",
                "source":source_identity(),"inference":{k:cfg.get(k,d) for k,d in
                 [("num_tokens",1200),("resolution_level",0),("refine_steps",3),("use_fp16",False),
                  ("force_projection",True),("fov_policy","auto"),("fov_x",None)]}}
    identity["inference"]["apply_mask"] = False
    audit = {"passed":True,"base_identity":identity,"base_identity_hash":fingerprint(identity),"fov_protocols":["known_fov"]}
    adapter = CachedMoGeAdapter(cfg,audit)
    assert adapter.kwargs(sample)["refine_steps"] == 3
    adapter.verify_audit(audit,sample)
    with pytest.raises(FileNotFoundError,match="cache missing"):
        adapter.predict(sample)
    with pytest.raises(RuntimeError,match="num_tokens"):
        CachedMoGeAdapter({**cfg,"num_tokens":2400},audit)


def test_exact_rgb_only_including_invalid_values(scene):
    from dataclasses import replace
    sample,_,prior = scene
    depth = prior.depth.copy(); depth[0,0] = np.inf
    prior = replace(prior,depth=depth)
    result = GeometryCompletion("rgb_only").complete(sample,prior)
    np.testing.assert_array_equal(result.depth,prior.depth)


def test_invalid_prediction_preserves_target(scene):
    # Explicit invalid-prediction metrics still retain every valid target pixel.
    sample,target,prior = scene
    depth = prior.depth.copy(); depth[0,0] = np.nan
    result = evaluate_depth(depth,sample,target)
    assert result["all_valid"]["count"] == target.gt_depth.size
    assert result["all_valid"]["invalid_predictions"] == 1
