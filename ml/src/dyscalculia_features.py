"""Turn trial-level game logs from the dyscalculia mini-games into per-child features.

This module works the same on synthetic logs and on real logs exported from the game. Expected columns:

    child_id, task, trial, correct (0/1), rt_ms, and task-specific fields:
      subitize   : n_dots, response
      magnitude  : left, right               (non-symbolic comparison; `correct` = chose larger)
      numberline : target, estimate, line_max (game: 0-20)
      sequence   : -
      arithmetic : -

Tasks map to game questions: subitize=Q11, magnitude=Q12, numberline=Q13, sequence=Q14, arithmetic=Q15.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TASKS = ["subitize", "magnitude", "numberline", "sequence", "arithmetic"]
_W_GRID = np.linspace(0.05, 1.2, 116)


def weber_fraction(left: np.ndarray, right: np.ndarray, correct: np.ndarray, lapse=0.02) -> float:
    """Maximum-likelihood ANS Weber fraction w for P(correct) = Phi(|n1-n2| / (w * sqrt(n1^2+n2^2)))."""
    d = np.abs(left - right) / np.sqrt(left ** 2 + right ** 2)
    best, best_ll = np.nan, -np.inf
    for w in _W_GRID:
        p = lapse / 2 + (1 - lapse) * norm.cdf(d / w)
        ll = np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p))
        if ll > best_ll:
            best, best_ll = w, ll
    return float(best)


def _r2(y, yhat):
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else 0.0


def child_features(t: pd.DataFrame) -> dict:
    f = {}
    g = {k: v for k, v in t.groupby("task")}

    # ---- accuracy view
    s = g["subitize"]
    f["acc_subitize_small"] = s.loc[s.n_dots <= 4, "correct"].mean()
    f["acc_subitize_large"] = s.loc[s.n_dots > 4, "correct"].mean()
    f["acc_magnitude"] = g["magnitude"].correct.mean()
    nl = g["numberline"]
    err = np.abs(nl.estimate - nl.target) / nl.line_max
    f["acc_numberline_within10pct"] = float((err <= 0.10).mean())
    f["acc_sequence"] = g["sequence"].correct.mean()
    f["acc_arithmetic"] = g["arithmetic"].correct.mean()

    # ---- response-time view
    for k in TASKS:
        rt = g[k].rt_ms
        f[f"rt_median_{k}"] = float(rt.median())
        f[f"rt_cv_{k}"] = float(rt.std() / rt.mean()) if rt.mean() > 0 else 0.0
    small = s[s.n_dots <= 4]
    f["rt_subitize_slope_ms_per_dot"] = float(np.polyfit(small.n_dots, small.rt_ms, 1)[0]) if small.n_dots.nunique() > 1 else 0.0

    # ---- number-sense profile view
    m = g["magnitude"]
    f["ans_weber_fraction"] = weber_fraction(m.left.values, m.right.values, m.correct.values)
    ratio = np.minimum(m.left, m.right) / np.maximum(m.left, m.right)
    f["ans_ratio_effect"] = float(m.correct[ratio <= 0.7].mean() - m.correct[ratio > 0.7].mean())
    x, e, L = nl.target.values.astype(float), nl.estimate.values.astype(float), float(nl.line_max.iloc[0])
    f["numberline_pae"] = float(np.mean(np.abs(e - x)) / L * 100)  # percent absolute error
    lin = np.polyval(np.polyfit(x, e, 1), x)
    logx = np.log(x + 1)
    logfit = np.polyval(np.polyfit(logx, e, 1), logx)
    f["numberline_linear_r2"] = _r2(e, lin)
    f["numberline_log_minus_linear_r2"] = _r2(e, logfit) - _r2(e, lin)  # >0 = immature logarithmic mapping
    f["numberline_bias"] = float(np.mean(e - x) / L * 100)
    span = 0
    for n in sorted(s.n_dots.unique()):
        if s.loc[s.n_dots == n, "correct"].all():
            span = n
        else:
            break
    f["subitizing_span"] = span
    f["subitize_mean_abs_error"] = float(np.mean(np.abs(s.response - s.n_dots)))
    return f


def build_feature_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows = {cid: child_features(t) for cid, t in trials.groupby("child_id")}
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


ACCURACY = ["acc_subitize_small", "acc_subitize_large", "acc_magnitude", "acc_numberline_within10pct",
            "acc_sequence", "acc_arithmetic"]
RESPONSE_TIME = [f"rt_median_{k}" for k in TASKS] + [f"rt_cv_{k}" for k in TASKS] + ["rt_subitize_slope_ms_per_dot"]
NUMBER_SENSE = ["ans_weber_fraction", "ans_ratio_effect", "numberline_pae", "numberline_linear_r2",
                "numberline_log_minus_linear_r2", "numberline_bias", "subitizing_span", "subitize_mean_abs_error"]
