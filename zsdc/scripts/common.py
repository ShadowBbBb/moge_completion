import argparse
import json
import sys

from ..config import load_config, validate_config
from ..evaluation.runner import audit_models, preflight, run_benchmark, run_sweep


def main(action="evaluate", argv=None):
    parser = argparse.ArgumentParser(description=f"MoGe-ZSDC {action} (training-free Phase 0/1)")
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--train-list")
    parser.add_argument("--val-list")
    parser.add_argument("--split", choices=["train", "val"])
    parser.add_argument("--output")
    parser.add_argument("--cache")
    parser.add_argument("--audit-dir")
    parser.add_argument("--cache-only", action="store_true", help="Require audited RGB caches; do not import/load MoGe")
    parser.add_argument("--model", choices=["v2", "v3"])
    parser.add_argument("--checkpoint", help="Requires --model; local model.pt or HF model id")
    parser.add_argument("--device")
    parser.add_argument("--methods", help="Comma-separated method names")
    parser.add_argument("--limit", type=int)
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument("--max-pixels", type=int)
    resolution.add_argument("--size", nargs=2, type=int, metavar=("H", "W"))
    resolution.add_argument("--original-resolution", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="DOTTED.KEY=YAML")
    parser.add_argument("--dry-run", action="store_true", help="Read-only data preflight; never loads model or writes files")
    parser.add_argument("--lists-only", action="store_true", help="Check both lists without accessing images/models")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.set)
        for arg, key in [(args.data_root, "root"), (args.train_list, "train_list"), (args.val_list, "val_list"), (args.split, "split")]:
            if arg is not None:
                config["data"][key] = arg
        for key in ("output", "cache", "audit_dir"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if args.model:
            config["models"] = [model for model in config["models"] if model["version"] == args.model]
            if not config["models"]:
                raise ValueError(f"No {args.model} configuration")
        if args.checkpoint:
            if not args.model:
                raise ValueError("--checkpoint requires --model")
            config["models"][0]["checkpoint"] = args.checkpoint
        if args.device:
            for model in config["models"]:
                model["device"] = args.device
        if args.methods:
            config["methods"] = args.methods.split(",")
        if args.cache_only:
            config["cache_only"] = True
        if args.limit is not None:
            config["evaluation"]["limit"] = args.limit
            config["audit"]["limit"] = args.limit
        if args.original_resolution:
            config["data"]["resolution"] = {"original": True}
        elif args.size:
            config["data"]["resolution"] = {"size": args.size}
        elif args.max_pixels is not None:
            config["data"]["resolution"] = {"max_pixels": args.max_pixels}
        validate_config(config)
        if action == "preflight" or args.dry_run or args.lists_only:
            result = preflight(config, args.limit, args.lists_only)
            print(json.dumps(result, indent=2))
            return 1 if result["failures"] else 0
        if action == "audit":
            reports = audit_models(config)
            print(json.dumps([{k:r[k] for k in ("version", "status", "passed", "failures")} for r in reports], indent=2))
            return 0 if all(r["passed"] for r in reports) else 1
        if action == "sweep":
            reports = run_sweep(config)
            print(json.dumps(reports, indent=2))
            return int(any(r["failures"] for r in reports))
        result = run_benchmark(config, action)
        print(json.dumps(result, indent=2))
        return int(bool(result["failures"]))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
