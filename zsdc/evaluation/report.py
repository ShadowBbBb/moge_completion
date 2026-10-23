import csv
from collections import defaultdict
from pathlib import Path

from ..utils import canonical, write_json
from .metrics import aggregate_regions


GROUP_KEYS = ("model", "pattern", "requested_points", "seed", "method")


class StreamingReport:
    """Bounded memory reporting for the full 9k-frame/155-condition matrix."""
    def __init__(self, output):
        self.output = Path(output)
        self.groups = {}
        self.count = 0
        self.csv_file = open(self.output/"per_image.csv", "w", newline="", encoding="utf-8")
        self.json_file = open(self.output/"per_image.jsonl", "w", encoding="utf-8")
        self.writer = None

    def add(self, record):
        self.count += 1
        self.json_file.write(canonical(record)+"\n")
        base = {k:v for k,v in record.items() if k not in ("metrics", "completion_metadata")}
        for region, metrics in record["metrics"].items():
            flat = {**base, "region": region, **metrics}
            if self.writer is None:
                self.writer = csv.DictWriter(self.csv_file, fieldnames=sorted(flat))
                self.writer.writeheader()
            self.writer.writerow(flat)
            key = tuple(record[k] for k in GROUP_KEYS)+(region,)
            if key not in self.groups:
                self.groups[key] = {"pixels":0,"invalid_predictions":0,"images":0,"nonempty_images":0,
                                    "failed_images":0,"fallback_images":0,"completion_seconds":0.,"model_seconds":0.,
                                    **{k:0. for k in ("sse","sae","srel","delta1_hits","rmse","mae","absrel","delta1")}}
            group = self.groups[key]
            group["pixels"] += metrics["count"]
            group["invalid_predictions"] += metrics["invalid_predictions"]
            group["images"] += 1
            group["nonempty_images"] += int(metrics["count"] > 0)
            group["failed_images"] += int(record["status"] != "ok")
            group["fallback_images"] += int(bool(record.get("fallback")))
            group["completion_seconds"] += record.get("completion_seconds",0.)
            group["model_seconds"] += record.get("model_seconds",0.)
            for field in ("sse","sae","srel","delta1_hits","rmse","mae","absrel","delta1"):
                if metrics[field] is not None:
                    group[field] += metrics[field]

    def snapshot(self, failures, timings):
        self.csv_file.flush(); self.json_file.flush()
        summary = []
        for key,g in self.groups.items():
            row = {**dict(zip((*GROUP_KEYS,"region"),key)), **{k:g[k] for k in
                   ("pixels","invalid_predictions","images","nonempty_images","failed_images","fallback_images")},
                   "mean_completion_seconds":g["completion_seconds"]/g["images"],
                   "mean_model_seconds":g["model_seconds"]/g["images"]}
            valid = g["pixels"] > 0 and not g["invalid_predictions"]
            for name,total in (("rmse","sse"),("mae","sae"),("absrel","srel"),("delta1","delta1_hits")):
                value = g[total]/g["pixels"] if valid else None
                row["pixel_"+name] = value**.5 if name == "rmse" and value is not None else value
                row["image_mean_"+name] = g[name]/g["nonempty_images"] if valid else None
            summary.append(row)
        write_csv(self.output/"summary.csv",summary)
        write_json(self.output/"timings.json",timings)
        (self.output/"failures.jsonl").write_text("".join(canonical(f)+"\n" for f in failures),encoding="utf-8")
        write_summary_markdown(self.output,summary,self.count,len(failures))
        return summary

    def close(self):
        self.csv_file.close(); self.json_file.close()


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, records, failures, timings):
    output = Path(output)
    flat, grouped = [], defaultdict(list)
    for record in records:
        base = {k:v for k,v in record.items() if k not in ("metrics", "completion_metadata")}
        for region, metrics in record["metrics"].items():
            flat.append({**base, "region": region, **metrics})
        grouped[tuple(record[k] for k in GROUP_KEYS)].append(record)
    summary = []
    for group, entries in grouped.items():
        for region in ("all_valid", "observed_valid", "unobserved_valid", "edge_valid"):
            summary.append({**dict(zip(GROUP_KEYS, group)), "region": region,
                            **aggregate_regions([e["metrics"][region] for e in entries]),
                            "failed_images": sum(e["status"] != "ok" for e in entries),
                            "fallback_images": sum(bool(e.get("fallback")) for e in entries),
                            "mean_completion_seconds": sum(e.get("completion_seconds", 0) for e in entries)/len(entries),
                            "mean_model_seconds": sum(e.get("model_seconds", 0) for e in entries)/len(entries)})
    write_csv(output/"per_image.csv", flat)
    write_csv(output/"summary.csv", summary)
    write_json(output/"per_image.json", records)
    write_json(output/"timings.json", timings)
    (output/"failures.jsonl").write_text("".join(__import__("json").dumps(f, ensure_ascii=False)+"\n" for f in failures), encoding="utf-8")
    write_summary_markdown(output,summary,len(records),len(failures))
    return summary


def write_summary_markdown(output,summary,count,failure_count):
    lines = ["# Phase 1 depth completion", "", f"Evaluated conditions: {count}; failures: {failure_count}.", "",
             "Metrics are undefined for empty regions or any invalid prediction in that region. No GT pixels are silently removed.", "",
             "| Model | Input | N | Seed | Method | Region | Pixels | RMSE (pixel) | RMSE (image mean) | Failed | Fallback |",
             "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "—" if v is None else f"{v:.6g}"
    for row in summary:
        if row["region"] in ("all_valid", "unobserved_valid"):
            lines.append(f"| {row['model']} | {row['pattern']} | {row['requested_points']} | {row['seed']} | {row['method']} | {row['region']} | {row['pixels']} | {fmt(row['pixel_rmse'])} | {fmt(row['image_mean_rmse'])} | {row['failed_images']} | {row['fallback_images']} |")
    lines += ["", "See summary.csv for MAE, AbsRel, delta1 and all four regions. Seed-specific rows are retained; no best-seed selection.",
              "Model time comes from the RGB-only cache metadata. Actual wall time includes cache I/O; reconstructed uncached time adds the saved model prediction cost per condition.",
              "CPU peak RSS is a process high-water mark, not per-method allocated memory. CUDA peak is measured during model inference."]
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
