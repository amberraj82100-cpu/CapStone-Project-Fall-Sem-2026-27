"""Generic multimodal fusion framework (early, late and hybrid fusion).

A *view* (modality) is a named group of feature columns. Given the views, this
module trains and compares:

* Unimodal baselines - one model per view per algorithm.
* Early fusion      - all views concatenated at the feature level, one model.
* Late fusion       - one model per view, their probabilities combined by
    - soft voting (mean),
    - performance-weighted voting (weights ~ inner-CV AUC - 0.5),
    - stacking (logistic-regression meta-learner on out-of-fold probabilities).
* Hybrid fusion     - meta-learner over the per-view probabilities *and* the
                      best early-fusion probability.

All decisions that use labels (choosing the per-view model, fusion weights,
meta-learner, decision thresholds) are made on inner out-of-fold predictions
of the training split only, so the outer test fold is never touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import SEED, compute_metrics, tune_threshold

# A factory receives {view_name: [columns]} and returns an unfitted sklearn estimator
Factory = Callable[[dict], object]


def _oof_and_fit(est, X: pd.DataFrame, y: np.ndarray, cv) -> tuple[np.ndarray, object]:
    oof = np.zeros(len(y), dtype=float)
    for tr, va in cv.split(X, y):
        m = clone(est).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof, clone(est).fit(X, y)


def _meta_learner():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class MultimodalFusion:
    views: dict[str, list[str]]
    unimodal_models: dict[str, Factory]
    early_models: dict[str, Factory]
    inner_splits: int = 5
    seed: int = SEED

    # fitted state
    models_: dict = field(default_factory=dict)
    oof_: dict = field(default_factory=dict)
    family_: dict = field(default_factory=dict)
    thresholds_: dict = field(default_factory=dict)
    inner_auc_: dict = field(default_factory=dict)
    best_view_model_: dict = field(default_factory=dict)
    best_early_: str = ""
    late_weights_: dict = field(default_factory=dict)
    meta_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: np.ndarray, verbose: bool = False):
        y = np.asarray(y)
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.seed)

        # 1) unimodal models
        for v, cols in self.views.items():
            for mname, fac in self.unimodal_models.items():
                name = f"Unimodal | {v} | {mname}"
                oof, full = _oof_and_fit(fac({v: cols}), X[cols], y, cv)
                self._store(name, "Unimodal", oof, full, y)
                if verbose:
                    print(f"    {name:55s} inner AUC={self.inner_auc_[name]:.3f}")

        # 2) early fusion
        all_cols = [c for cols in self.views.values() for c in cols]
        for mname, fac in self.early_models.items():
            name = f"Early fusion | {mname}"
            oof, full = _oof_and_fit(fac(self.views), X[all_cols], y, cv)
            self._store(name, "Early fusion", oof, full, y)
            if verbose:
                print(f"    {name:55s} inner AUC={self.inner_auc_[name]:.3f}")
        early_names = [n for n, f in self.family_.items() if f == "Early fusion"]
        self.best_early_ = max(early_names, key=lambda n: self.inner_auc_[n])

        # 3) late fusion - best algorithm per view chosen on inner OOF AUC
        for v in self.views:
            cands = [f"Unimodal | {v} | {m}" for m in self.unimodal_models]
            self.best_view_model_[v] = max(cands, key=lambda n: self.inner_auc_[n])
        P = self._view_matrix_oof()
        raw_w = np.array([max(self.inner_auc_[self.best_view_model_[v]] - 0.5, 1e-3) for v in self.views])
        w = raw_w / raw_w.sum()
        self.late_weights_ = dict(zip(self.views, w))

        self._store("Late fusion | soft voting", "Late fusion", P.mean(1), None, y)
        self._store("Late fusion | AUC-weighted voting", "Late fusion", P @ w, None, y)

        meta = _meta_learner()
        Z = _logit(P)
        oof_stack, meta_full = _oof_and_fit(meta, pd.DataFrame(Z), y, cv)
        self.meta_["stacking"] = meta_full
        self._store("Late fusion | stacking (LR meta-learner)", "Late fusion", oof_stack, None, y)

        # 4) hybrid fusion - views + best early-fusion model
        Zh = np.column_stack([Z, _logit(self.oof_[self.best_early_])])
        oof_h, meta_h = _oof_and_fit(_meta_learner(), pd.DataFrame(Zh), y, cv)
        self.meta_["hybrid"] = meta_h
        self._store("Hybrid fusion | views + early (LR meta-learner)", "Hybrid fusion", oof_h, None, y)
        return self

    def _store(self, name, family, oof, model, y):
        self.oof_[name] = oof
        self.family_[name] = family
        self.models_[name] = model
        self.inner_auc_[name] = roc_auc_score(y, oof)
        self.thresholds_[name] = tune_threshold(y, oof)

    def _view_matrix_oof(self):
        return np.column_stack([self.oof_[self.best_view_model_[v]] for v in self.views])

    # ------------------------------------------------------------------ predict
    def predict_proba_all(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for name, fam in self.family_.items():
            if fam == "Unimodal":
                v = name.split(" | ")[1]
                out[name] = self.models_[name].predict_proba(X[self.views[v]])[:, 1]
            elif fam == "Early fusion":
                cols = [c for cs in self.views.values() for c in cs]
                out[name] = self.models_[name].predict_proba(X[cols])[:, 1]
        P = np.column_stack([out[self.best_view_model_[v]] for v in self.views])
        w = np.array([self.late_weights_[v] for v in self.views])
        Z = _logit(P)
        out["Late fusion | soft voting"] = P.mean(1)
        out["Late fusion | AUC-weighted voting"] = P @ w
        out["Late fusion | stacking (LR meta-learner)"] = self.meta_["stacking"].predict_proba(pd.DataFrame(Z))[:, 1]
        Zh = np.column_stack([Z, _logit(out[self.best_early_])])
        out["Hybrid fusion | views + early (LR meta-learner)"] = self.meta_["hybrid"].predict_proba(pd.DataFrame(Zh))[:, 1]
        return out

    def view_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-view probabilities feeding late fusion (useful for local explanations)."""
        return pd.DataFrame({
            v: self.models_[self.best_view_model_[v]].predict_proba(X[self.views[v]])[:, 1]
            for v in self.views
        }, index=X.index)

    def stacking_view_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-sample additive contribution (in log-odds) of every view to the stacking decision."""
        Z = _logit(self.view_probabilities(X).values)
        coef = self.meta_["stacking"].coef_[0]
        return pd.DataFrame(Z * coef, columns=list(self.views), index=X.index)


# ---------------------------------------------------------------------- outer evaluation
def nested_evaluate(make_fusion: Callable[[], MultimodalFusion], X: pd.DataFrame, y: np.ndarray,
                    outer_cv, verbose=True):
    """Outer CV loop. Returns (per-fold metrics DataFrame, pooled predictions DataFrame, fold info)."""
    y = np.asarray(y)
    rows, preds, info = [], [], []
    for k, (tr, te) in enumerate(outer_cv.split(X, y)):
        if verbose:
            print(f"  outer fold {k + 1}: train={len(tr)} test={len(te)}")
        fus = make_fusion().fit(X.iloc[tr], y[tr], verbose=False)
        probs = fus.predict_proba_all(X.iloc[te])
        for name, p in probs.items():
            m = compute_metrics(y[te], p, fus.thresholds_[name])
            m.update(model=name, family=fus.family_[name], fold=k, inner_auc=fus.inner_auc_[name])
            rows.append(m)
            preds.append(pd.DataFrame({"fold": k, "model": name, "index": X.index[te], "y": y[te], "p": p,
                                       "threshold": fus.thresholds_[name]}))
        info.append({"fold": k, "best_view_model": fus.best_view_model_, "best_early": fus.best_early_,
                     "late_weights": fus.late_weights_,
                     "stacking_coef": dict(zip(fus.views, fus.meta_["stacking"].coef_[0]))})
        if verbose:
            best = max(probs, key=lambda n: roc_auc_score(y[te], probs[n]))
            print(f"    best on this fold: {best} (AUC={roc_auc_score(y[te], probs[best]):.3f})")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), info


def summarise(per_fold: pd.DataFrame, metrics=("balanced_accuracy", "roc_auc", "pr_auc", "sensitivity",
                                                "specificity", "precision", "f1", "mcc", "accuracy")):
    g = per_fold.groupby(["family", "model"])[list(metrics)]
    mean, std = g.mean(), g.std()
    out = mean.copy()
    for m in metrics:
        out[m + "_std"] = std[m]
    return out.reset_index().sort_values("roc_auc", ascending=False)
