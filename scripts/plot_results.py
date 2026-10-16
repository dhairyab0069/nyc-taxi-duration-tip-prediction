"""Make the report figures from runs/<task>/results.json and test_predictions.parquet.

Example:
    python scripts/plot_results.py --out figures
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.metrics import roc_curve

MODELS = ["baseline", "linear", "hgb", "lightgbm", "mlp"]
LABEL = {"baseline": "Baseline", "linear": "Linear", "hgb": "HGB", "lightgbm": "LightGBM", "mlp": "MLP"}

SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
COLOR = {"baseline": NEUTRAL, "linear": "#2a78d6", "hgb": "#eb6834", "lightgbm": "#1baf7a", "mlp": "#eda100"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(task):
    with open(f"runs/{task}/results.json") as f:
        results = json.load(f)["results"]
    preds = pd.read_parquet(f"runs/{task}/test_predictions.parquet")
    return results, preds


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/{name}")


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_line_ends(ax, x, series):
    """Direct-label each line at its last point, nudging labels apart so they do not overlap."""
    ends = sorted(((y[-1], name) for name, y in series.items()), key=lambda t: t[0])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    placed = []
    for y, name in ends:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(LABEL[name], (x[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)


def plot_ladder(dur, tip, out):
    """Validation (January) vs test (February) score for every model, both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    panels = [(axes[0], dur, "mae_min", "Trip duration: MAE (minutes, lower is better)", "{:.2f}"),
              (axes[1], tip, "auroc", "Tip prediction: AUROC (higher is better)", "{:.3f}")]
    for ax, res, key, title, fmt in panels:
        for i, m in enumerate(MODELS):
            v, t = res[m]["val"][key], res[m]["test"][key]
            ax.plot([v, t], [i, i], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
            ax.scatter(v, i, s=60, facecolor=SURFACE, edgecolor=COLOR[m], linewidth=2, zorder=2)
            ax.scatter(t, i, s=60, color=COLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(fmt.format(t), (t, i), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS])
        ax.set_ylim(len(MODELS) - 0.4, -0.6)
        ax.set_title(title)
        ax.tick_params(axis="y", length=0)
        style(ax, "x")
    axes[0].scatter([], [], s=60, facecolor=SURFACE, edgecolor=INK_2, linewidth=2, label="Validation (Jan)")
    axes[0].scatter([], [], s=60, color=INK_2, label="Test (Feb)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    base, best = dur["baseline"]["test"]["mae_min"], dur["lightgbm"]["test"]["mae_min"]
    fig.suptitle(f"LightGBM cuts duration error by {100 * (1 - best / base):.0f}% over the baseline; "
                 "every model scores worse on the later month", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig1_model_ladder.png")


def plot_importance(dur, tip, out, top=8):
    """LightGBM split-gain share of the top features for each task."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, res, title in [(axes[0], dur, "Trip duration"), (axes[1], tip, "Tip prediction")]:
        gain = pd.Series(res["lightgbm"]["feature_importance_gain"])
        share = (gain / gain.sum()).sort_values(ascending=False).head(top)[::-1]
        ax.barh(range(len(share)), share.values * 100, color="#2a78d6", height=0.65)
        for i, v in enumerate(share.values * 100):
            ax.annotate(f"{v:.1f}%", (v, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)
        ax.set_yticks(range(len(share)), share.index)
        ax.set_xlabel("Share of total split gain (%)")
        ax.set_title(title)
        ax.tick_params(axis="y", length=0)
        style(ax, "x")
    fig.suptitle("What LightGBM relies on: distance for duration, rate code and zones for tips",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig2_feature_importance.png")


def plot_pred_vs_actual(preds, out):
    """Density of LightGBM predicted vs actual duration, with the error distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    hb = ax.hexbin(preds.y_true, preds.lightgbm, gridsize=70, extent=(0, 90, 0, 90),
                   cmap=SEQ, norm=LogNorm(), mincnt=1, linewidths=0)
    ax.plot([0, 90], [0, 90], color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("Actual duration (min)")
    ax.set_ylabel("Predicted duration (min)")
    ax.set_title("LightGBM predicted vs actual (test month)")
    fig.colorbar(hb, ax=ax, label="Trips").outline.set_visible(False)

    ax = axes[1]
    bins = np.linspace(-30, 30, 121)
    for m in ["baseline", "lightgbm"]:
        err = preds[m] - preds.y_true
        ax.hist(err, bins=bins, histtype="step", linewidth=2, color=COLOR[m],
                label=f"{LABEL[m]} (median |error| {np.median(np.abs(err)):.1f} min)")
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Prediction error (min), predicted − actual")
    ax.set_ylabel("Trips")
    ax.set_title("Error distribution")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)  # headroom so the legend clears the peak
    ax.legend(fontsize=8.5, loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1)
    style(ax)
    fig.suptitle("Most trips land near the diagonal; long trips are underpredicted",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig3_pred_vs_actual.png")


def mae_by(preds, key, bins=None):
    g = pd.cut(preds[key], bins) if bins is not None else preds[key]
    err = preds[MODELS].sub(preds.y_true, axis=0).abs()
    return err.groupby(g, observed=True).mean(), preds.groupby(g, observed=True).size()


def plot_error_breakdown(preds, out):
    """MAE by trip distance and by pickup hour for every model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    dist_bins = [0, 1, 2, 3, 5, 8, 12, 20, 100]
    by_dist, _ = mae_by(preds, "trip_distance", dist_bins)
    by_hour, _ = mae_by(preds, "hour")
    for ax, table, xlabels, xlabel, title in [
            (axes[0], by_dist, [f"{a}–{b}" if b < 100 else f"{a}+" for a, b in zip(dist_bins, dist_bins[1:])],
             "Trip distance (miles)", "MAE by trip distance"),
            (axes[1], by_hour, [str(h) for h in by_hour.index], "Pickup hour", "MAE by pickup hour")]:
        x = np.arange(len(table))
        for m in MODELS:
            ax.plot(x, table[m].values, color=COLOR[m], linewidth=2, marker="o", markersize=4)
        ax.set_xticks(x[:: 1 if len(x) < 10 else 3], xlabels[:: 1 if len(x) < 10 else 3])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MAE (min)")
        ax.set_title(title)
        style(ax)
        label_line_ends(ax, x, {m: table[m].values for m in MODELS})
        ax.set_xlim(-0.3, len(x) - 0.3 + 1.2)
    fig.suptitle("Error grows with trip distance; LightGBM has the lowest error in every distance bin and every hour",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig4_error_breakdown.png")


def plot_tip_scoring(preds, out):
    """ROC curves and calibration for the tip models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for m in MODELS:
        fpr, tpr, _ = roc_curve(preds.y_true, preds[m])
        ax.plot(fpr, tpr, color=COLOR[m], linewidth=2, label=LABEL[m])
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("False-positive rate (untipped trips flagged as tipped)")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC curves (test month)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax, "both")

    ax = axes[1]
    edges = np.array([0, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0])
    for m in ["linear", "lightgbm", "mlp"]:
        b = pd.cut(preds[m], edges, include_lowest=True)
        g = preds.groupby(b, observed=True).agg(p=(m, "mean"), y=("y_true", "mean"), n=("y_true", "size"))
        g = g[g.n >= 200]
        ax.plot(g.p, g.y, color=COLOR[m], linewidth=2, marker="o", markersize=5, label=LABEL[m])
    ax.plot([0.6, 1], [0.6, 1], color=INK_2, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Predicted P(tipped)")
    ax.set_ylabel("Observed tip rate")
    ax.set_title("Calibration (bins with ≥ 200 trips)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax, "both")
    fig.suptitle(f"Tip models rank moderately (AUROC ≈ 0.75) and are calibrated above 0.85, where most trips fall "
                 f"({preds.y_true.mean():.0%} of card trips tip)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig5_tip_scoring.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (dur, dur_preds), (tip, tip_preds) = load("duration"), load("tip")
    plot_ladder(dur, tip, args.out)
    plot_importance(dur, tip, args.out)
    plot_pred_vs_actual(dur_preds, args.out)
    plot_error_breakdown(dur_preds, args.out)
    plot_tip_scoring(tip_preds, args.out)


if __name__ == "__main__":
    main()
