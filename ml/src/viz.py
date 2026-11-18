"""Figures shared by both pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import FAMILY_COLORS, GRID, PALETTE, SURFACE, TEXT, TEXT_2, setup_mpl

plt = setup_mpl()


def _short(name: str) -> str:
    return name.replace("Unimodal | ", "").replace("Early fusion | ", "Early: ").replace(
        "Late fusion | ", "Late: ").replace("Hybrid fusion | ", "Hybrid: ")


def plot_model_comparison(summary: pd.DataFrame, path, title, metric="roc_auc", label="ROC-AUC", xlim=None):
    s = summary.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(s) + 1.4))
    y = np.arange(len(s))
    colors = [FAMILY_COLORS[f] for f in s["family"]]
    ax.barh(y, s[metric], xerr=s[metric + "_std"], color=colors, height=0.62,
            error_kw=dict(ecolor=TEXT_2, elinewidth=1, capsize=2))
    for yi, v in zip(y, s[metric]):
        ax.text(v + 0.004, yi + 0.18, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y, [_short(n) for n in s["model"]], fontsize=8)
    ax.set_xlabel(f"{label} (mean ± sd across outer folds)")
    lo = xlim[0] if xlim else max(0.0, s[metric].min() - 0.08)
    ax.set_xlim(lo, xlim[1] if xlim else 1.0)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in FAMILY_COLORS.items() if f in set(s["family"])]
    ax.legend(handles, [f for f in FAMILY_COLORS if f in set(s["family"])], loc="lower right", fontsize=8)
    ax.set_title(title, loc="left")
    fig.savefig(path)
    plt.close(fig)


def plot_roc(preds: pd.DataFrame, models: list[str], path, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for i, m in enumerate(models):
        d = preds[preds.model == m]
        fpr, tpr, _ = roc_curve(d.y, d.p)
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=PALETTE[i], lw=2, label=f"{_short(m)}  (pooled AUC {roc_auc_score(d.y, d.p):.3f})")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title(title, loc="left")
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(tp, fp, tn, fn, path, title, labels=("Negative", "Positive")):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap=_blue_cmap())
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({frac:.0%} of row)", ha="center", va="center",
                    color="white" if frac > 0.5 else TEXT, fontsize=10)
    ax.set_xticks([0, 1], [f"Pred. {l}" for l in labels])
    ax.set_yticks([0, 1], [f"True {l}" for l in labels])
    ax.set_title(title, loc="left", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#0d366b"])


def plot_fusion_weights(info: list[dict], views: list[str], path, title):
    w = pd.DataFrame([f["late_weights"] for f in info])[views]
    c = pd.DataFrame([f["stacking_coef"] for f in info])[views]
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.5 * len(views) + 1.8))
    for ax, df, lab in [(axes[0], w, "AUC-weighted voting weight"), (axes[1], c, "Stacking meta-learner coefficient")]:
        y = np.arange(len(views))
        ax.barh(y, df.mean(), xerr=df.std(), color=PALETTE[2], height=0.55,
                error_kw=dict(ecolor=TEXT_2, elinewidth=1, capsize=2))
        ax.set_yticks(y, views)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=TEXT_2, lw=0.8)
    axes[1].set_yticklabels([])
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", color=TEXT)
    fig.savefig(path)
    plt.close(fig)


def plot_hbar(values: pd.Series, path, title, xlabel, color=PALETTE[0], top=None):
    s = values.sort_values(ascending=True)
    if top:
        s = s.iloc[-top:]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(s) + 1.2))
    ax.barh(np.arange(len(s)), s.values, color=color, height=0.6)
    ax.set_yticks(np.arange(len(s)), s.index, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    fig.savefig(path)
    plt.close(fig)
