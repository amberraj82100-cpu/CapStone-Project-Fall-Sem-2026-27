"""Shared paths, metrics and plotting helpers for the screening ML pipelines."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "datasets"
ML_DIR = REPO_ROOT / "ml"
RESULTS_DIR = ML_DIR / "results"
MODELS_DIR = ML_DIR / "models"
CACHE_DIR = ML_DIR / ".cache"

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------- metrics
METRIC_ORDER = [
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "mcc",
    "accuracy",
]


def tune_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Pick the decision threshold that maximises balanced accuracy (Youden's J).

    Always called on *training-side* out-of-fold probabilities, never on test data.
    """
    cands = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    best_t, best_j = 0.5, -1.0
    pos, neg = y == 1, y == 0
    for t in cands:
        yhat = p >= t
        j = yhat[pos].mean() + (~yhat[neg]).mean()
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def compute_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "precision": precision_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 else 0.0,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "threshold": thr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y, p, thr, metric="roc_auc", n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(compute_metrics(y[idx], p[idx], thr)[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- plotting
# Validated categorical palette (fixed order, never cycled) - see docs/ML_METHODOLOGY.md
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_COLORS = {"Unimodal": PALETTE[0], "Early fusion": PALETTE[1], "Late fusion": PALETTE[2], "Hybrid fusion": PALETTE[3]}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
    return plt
