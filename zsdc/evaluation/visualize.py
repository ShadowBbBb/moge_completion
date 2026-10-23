from pathlib import Path

import numpy as np


def panels(path, sample, prior, result, target=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sensor = np.where(sample.sensor_mask, sample.sensor_depth, np.nan)
    panels_ = [("RGB", sample.image, None), ("Sensor", sensor, "turbo"),
               ("RGB-only", prior.depth, "turbo"), ("Completion", result.depth, "turbo")]
    if target is not None:
        panels_ += [("GT", np.where(target.gt_valid_mask, target.gt_depth, np.nan), "turbo"),
                    ("Absolute error", np.where(target.gt_valid_mask, np.abs(result.depth-target.gt_depth), np.nan), "magma")]
    if result.correction is not None:
        panels_.append(("Log correction", result.correction, "coolwarm"))
    if result.weights is not None:
        panels_.append(("Observation weights", np.where(sample.sensor_mask, result.weights, np.nan), "viridis"))
    if result.graph_degree is not None:
        panels_.append(("Graph weighted degree", result.graph_degree, "viridis"))
    cols = 4
    fig, axes = plt.subplots((len(panels_)+cols-1)//cols, cols, figsize=(16, 4*((len(panels_)+cols-1)//cols)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, value, cmap) in zip(axes.flat, panels_):
        limits = {}
        if title in ("Sensor","RGB-only","Completion","GT"):
            reference = target.gt_depth[target.gt_valid_mask] if target is not None else prior.depth[np.isfinite(prior.depth)]
            if reference.size:
                limits = {"vmin":float(np.min(reference)),"vmax":float(np.max(reference))}
        elif title == "Log correction":
            extent = max(float(np.max(np.abs(value))),1e-8)
            limits = {"vmin":-extent,"vmax":extent}
        elif title == "Observation weights":
            limits = {"vmin":0.,"vmax":max(1.,float(np.nanmax(result.weights)))}
        artist = ax.imshow(value, cmap=cmap, **limits)
        ax.set_title(title)
        if cmap:
            fig.colorbar(artist, ax=ax, fraction=.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(path, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = [r for r in summary if r["region"] == "unobserved_valid" and r["pixel_rmse"] is not None]
    for model, method, pattern in sorted({(r["model"], r["method"], r["pattern"]) for r in rows}):
        selected = [r for r in rows if (r["model"], r["method"], r["pattern"]) == (model, method, pattern)]
        label = f"{model}/{method}/{pattern}"
        axes[0].scatter([r["mean_model_seconds"]+r["mean_completion_seconds"] for r in selected], [r["pixel_rmse"] for r in selected], label=label, s=12)
        if pattern != "native":
            budgets = sorted({r["requested_points"] for r in selected})
            means = [np.mean([r["pixel_rmse"] for r in selected if r["requested_points"] == n]) for n in budgets]
            axes[1].plot(budgets, means, marker=".", label=label)
    axes[0].set(xlabel="Model + completion seconds", ylabel="Unobserved RMSE (m)")
    axes[1].set(xlabel="Requested observations", ylabel="Unobserved RMSE (m)", xscale="log")
    if rows:
        axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
