"""Explicitly synthetic solver diagnostics; these are NOT MoGe benchmark results."""
import argparse
from pathlib import Path

import numpy as np

from ..config import load_config
from ..data.geometry import pixel_to_normalized, unproject
from ..data.sampling import sample_sensor
from ..data.schema import CompletionInput, EvaluationTarget
from ..evaluation.metrics import evaluate_depth
from ..evaluation.report import write_report
from ..evaluation.visualize import panels, plot_summary
from ..models.output_types import GeometryPrediction
from ..solvers.correction import GeometryCompletion
from ..utils import environment, source_identity, write_json


def synthetic_scene(kind="plane", shape=(24, 32)):
    y, x = np.indices(shape)
    gt = 2. + x*.015 + y*.01
    image = np.full((*shape, 3), .3, np.float32)
    if kind == "step":
        gt = gt + (x >= shape[1]//2)*2.
        image[x >= shape[1]//2] = .8
    # Smooth local multiplicative distortion cannot be fully corrected by global alignment.
    prior_depth = gt*np.exp(.15*np.sin(x/6)*np.cos(y/8)) / 1.3
    K = np.array([[40., 0, (shape[1]-1)/2], [0, 40., (shape[0]-1)/2], [0, 0, 1.]])
    sample = CompletionInput(image, gt.astype(np.float32), np.ones(shape, bool), K,
                             sample_id=f"synthetic/{kind}", metadata={"source": "synthetic", "preprocessing_version": 1})
    prior = GeometryPrediction(prior_depth, unproject(prior_depth, K), pixel_to_normalized(K, shape), np.ones(shape, bool),
                               metadata={"source": "analytic_synthetic_prior"})
    return sample, EvaluationTarget(gt, np.ones(shape, bool)), prior


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/phase1_synthetic")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    import yaml
    (output/"run_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    rows, failures, timings = [], [], []
    for kind in ("plane", "step"):
        sample, target, prior = synthetic_scene(kind)
        sample = sample_sensor(sample, "random", 100, 0)
        for method in config["methods"]:
            result = GeometryCompletion(method, config["completion"]).complete(sample, prior)
            row = {"model": "SYNTHETIC_ANALYTIC_PRIOR", "pattern": "random", "requested_points": 100, "seed": 0,
                   "method": method, "sample_id": sample.sample_id, "status": "ok", "fallback": result.metadata.get("fallback"),
                   "model_seconds": 0., "completion_seconds": result.metadata["completion_seconds"],
                   "metrics": evaluate_depth(result.depth, sample, target), "completion_metadata": result.metadata}
            rows.append(row)
            timings.append({"sample_id":sample.sample_id,"method":method,"completion_seconds":result.metadata["completion_seconds"],
                            "linear_solves":result.metadata.get("linear_solves",[])})
            panels(output/f"{kind}_{method}.png", sample, prior, result, target)
    summary = write_report(output, rows, failures, timings)
    plot_summary(output/"accuracy_runtime.png", summary)
    write_json(output/"manifest.json", {"source": source_identity(), "environment": environment(), "status": "executed",
                                        "synthetic_only": True, "real_data_or_checkpoint_used": False})
    with open(output/"summary.md", "r+", encoding="utf-8") as f:
        content = f.read(); f.seek(0)
        f.write("**SYNTHETIC ONLY: analytic prior, not MoGe inference or real-data evidence.**\n\n"+content)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
