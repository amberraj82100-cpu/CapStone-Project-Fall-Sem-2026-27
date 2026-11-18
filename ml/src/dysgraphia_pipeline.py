"""Dysgraphia risk prediction from handwriting images (Potential vs Low-Potential Dysgraphia).

Views (modalities):
  * Handwriting geometry - 25 interpretable, scale-normalised handwriting measures
  * Deep visual          - EfficientNet-B0 (ImageNet) embeddings of the handwriting strip

Run:  python ml/src/dysgraphia_pipeline.py
"""
from __future__ import annotations

import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

import dysgraphia_features as F
import viz
from common import MODELS_DIR, PALETTE, RESULTS_DIR, SEED, TEXT, ensure_dir, save_json, set_seed, setup_mpl
from fusion import MultimodalFusion, nested_evaluate
from reporting import standard_report

OUT = ensure_dir(RESULTS_DIR / "dysgraphia")
GEO, DEEP = "Handwriting geometry", "Deep visual (EfficientNet-B0)"
N_REPEATS = 3


# ---------------------------------------------------------------------------- models
def _prep(views: dict) -> ColumnTransformer:
    steps = []
    for v, cols in views.items():
        if v == DEEP:  # 1280-d embedding -> PCA keeps the small-data model well conditioned
            steps.append((v, make_pipeline(StandardScaler(), PCA(n_components=32, random_state=SEED)), cols))
        else:
            steps.append((v, StandardScaler(), cols))
    return ColumnTransformer(steps)


def model_factories():
    def logreg(views):
        return make_pipeline(_prep(views), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))

    def svm(views):
        return make_pipeline(_prep(views), SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced",
                                               random_state=SEED))

    def rf(views):
        return make_pipeline(_prep(views), RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                                  class_weight="balanced", n_jobs=8,
                                                                  random_state=SEED))

    def xgb(views):
        return make_pipeline(_prep(views), XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05,
                                                         subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                                                         eval_metric="logloss", n_jobs=8, random_state=SEED))

    unimodal = {"Logistic Regression": logreg, "SVM (RBF)": svm, "Random Forest": rf}
    early = {"Logistic Regression": logreg, "SVM (RBF)": svm, "Random Forest": rf, "XGBoost": xgb}
    return unimodal, early


# ---------------------------------------------------------------------------- XAI
def shap_geometry(final: MultimodalFusion, X: pd.DataFrame):
    import shap

    plt = setup_mpl()
    name = final.best_view_model_[GEO]
    model = final.models_[name]
    cols = final.views[GEO]
    Xg = X[cols]
    f = lambda a: model.predict_proba(pd.DataFrame(a, columns=cols))[:, 1]
    explainer = shap.Explainer(f, shap.maskers.Independent(Xg, max_samples=100), seed=SEED)
    sv = explainer(Xg)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title(f"SHAP - handwriting geometry ({name.split(' | ')[-1]})", loc="left", fontweight="bold")
    plt.savefig(OUT / "xai_shap_geometry_beeswarm.png")
    plt.close("all")
    imp = pd.Series(np.abs(sv.values).mean(0), index=cols).sort_values(ascending=False)
    imp.to_csv(OUT / "xai_geometry_importance.csv", header=["mean_abs_shap"])
    # direction: correlation of feature value with its SHAP value (+ => higher value raises risk)
    direction = {c: float(np.corrcoef(Xg[c], sv.values[:, i])[0, 1]) if Xg[c].std() > 0 else 0.0
                 for i, c in enumerate(cols)}
    return imp, direction


def grouped_permutation_importance(factory, views, X, y, n_rep=20):
    """Drop in held-out AUC when ALL columns of one view are shuffled together (5-fold CV)."""
    rng = np.random.default_rng(SEED)
    cols = [c for cs in views.values() for c in cs]
    drops = {v: [] for v in views}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = factory(views).fit(X.iloc[tr][cols], y[tr])
        Xte = X.iloc[te][cols]
        base = roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])
        for v, vc in views.items():
            for _ in range(n_rep):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[vc] = Xte[vc].values[perm]
                drops[v].append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
    return {v: (float(np.mean(d)), float(np.std(d))) for v, d in drops.items()}


def occlusion_maps(final: MultimodalFusion, X: pd.DataFrame, y: np.ndarray, imgs: pd.DataFrame, n_per_class=3):
    """Occlusion sensitivity for the deep-visual view, on images held out from the explained model."""
    plt = setup_mpl()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [PALETTE[0], "#f0efec", PALETTE[1]])

    algo = final.best_view_model_[DEEP].split(" | ")[-1]
    unimodal, _ = model_factories()
    cols = final.views[DEEP]
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    model = unimodal[algo]({DEEP: cols}).fit(X.iloc[tr][cols], y[tr])
    p_te = model.predict_proba(X.iloc[te][cols])[:, 1]

    chosen = []
    for label in (1, 0):
        idx = [i for i in range(len(te)) if y[te[i]] == label and (p_te[i] >= 0.5) == bool(label)]
        idx = sorted(idx, key=lambda i: -abs(p_te[i] - 0.5))[:n_per_class]
        chosen += [te[i] for i in idx]

    P, S = 56, 28
    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, 1.55 * len(chosen) + 0.6))
    for ax, k in zip(np.atleast_1d(axes), chosen):
        strip = F.to_rgb_strip(F.preprocess(F.read_gray(imgs.path.iloc[k])))
        Hs, Ws = strip.shape[:2]
        starts = F.tile_starts(Ws)
        T = np.stack([strip[:, s:s + F.TILE] if s + F.TILE <= Ws else
                      np.pad(strip[:, s:], ((0, 0), (0, s + F.TILE - Ws), (0, 0)), constant_values=255)
                      for s in starts]).astype("float32")
        E = F.cnn().predict(T, verbose=0, batch_size=32)
        p0 = model.predict_proba(pd.DataFrame(E.mean(0)[None], columns=cols))[:, 1][0]
        heat = np.zeros((Hs, Ws)); cnt = np.zeros((Hs, Ws))
        patches = [(r, c) for r in range(0, Hs - P + 1, S) for c in range(0, Ws - P + 1, S)]
        # skip blank patches (occluding white paper changes nothing)
        patches = [(r, c) for r, c in patches if (strip[r:r + P, c:c + P, 0] < 128).any()]
        batch_tiles, owners = [], []
        for j, (r, c) in enumerate(patches):
            for t_i, s in enumerate(starts):
                if c < s + F.TILE and c + P > s:
                    t = T[t_i].copy()
                    c0, c1 = max(c - s, 0), min(c + P - s, F.TILE)
                    t[r:r + P, c0:c1] = 255
                    batch_tiles.append(t); owners.append((j, t_i))
        Eo = F.cnn().predict(np.stack(batch_tiles), verbose=0, batch_size=64) if batch_tiles else np.zeros((0, 1280))
        sums = {j: E.sum(0).copy() for j in range(len(patches))}
        for (j, t_i), e in zip(owners, Eo):
            sums[j] += e - E[t_i]
        feats = np.stack([sums[j] / len(starts) for j in range(len(patches))]) if patches else np.zeros((0, 1280))
        pj = model.predict_proba(pd.DataFrame(feats, columns=cols))[:, 1] if len(feats) else []
        for (r, c), p in zip(patches, pj):
            heat[r:r + P, c:c + P] += p0 - p  # >0: region pushes towards dysgraphia
            cnt[r:r + P, c:c + P] += 1
        heat = np.divide(heat, cnt, out=np.zeros_like(heat), where=cnt > 0)
        lim = max(np.abs(heat).max(), 1e-3)
        ax.imshow(strip[..., 0], cmap="gray", vmin=0, vmax=255)
        ax.imshow(heat, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), alpha=0.55)
        ax.set_axis_off()
        lab = "Potential dysgraphia" if y[k] == 1 else "Low potential"
        ax.set_title(f"{imgs.file.iloc[k]} - true: {lab} - p(dysgraphia)={p0:.2f}", loc="left", fontsize=9, color=TEXT)
    fig.suptitle("Occlusion sensitivity (deep-visual view). Orange = region raises dysgraphia risk, blue = lowers it",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "xai_occlusion_maps.png")
    plt.close(fig)
    return [imgs.file.iloc[k] for k in chosen]


# ---------------------------------------------------------------------------- main
def main():
    set_seed()
    t0 = time.time()
    imgs = F.list_images()
    print(f"Images: {len(imgs)} (positives={imgs.label.sum()})  - extracting features ...")
    X = F.build_feature_table(imgs)
    y = imgs.label.values
    views = {GEO: F.GEOMETRY_FEATURES, DEEP: [c for c in X.columns if c.startswith(F.DEEP_PREFIX)]}
    unimodal, early = model_factories()
    make = lambda: MultimodalFusion(views, unimodal, early, inner_splits=5)

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    per_fold, preds, info = nested_evaluate(make, X, y, outer)
    summary, best, pooled, rep = standard_report(
        per_fold, preds, info, views, OUT, "Dysgraphia", labels=("Low potential", "Potential dysgraphia"),
        n_outer_desc=f"5-fold CV x {N_REPEATS} repeats")

    print("Fitting final model on all data ...")
    final = make().fit(X, y, verbose=True)
    final.unimodal_models, final.early_models = {}, {}
    joblib.dump({"fusion": final, "views": views, "canonical_letter_height": F.CANONICAL_LETTER_HEIGHT},
                ensure_dir(MODELS_DIR) / "dysgraphia_fusion.joblib")

    print("XAI: SHAP on geometry view ...")
    geo_imp, geo_dir = shap_geometry(final, X)
    viz.plot_hbar(geo_imp, OUT / "xai_geometry_importance.png", "Handwriting-geometry features by mean |SHAP|",
                  "mean |SHAP| (impact on p(dysgraphia))")
    print("XAI: grouped permutation importance ...")
    best_early_algo = summary[summary.family == "Early fusion"].iloc[0]["model"].split(" | ")[-1]
    gpi = grouped_permutation_importance(early[best_early_algo], views, X, y)
    viz.plot_hbar(pd.Series({k: v[0] for k, v in gpi.items()}), OUT / "xai_modality_importance.png",
                  f"Modality importance - early fusion ({best_early_algo})",
                  "drop in held-out ROC-AUC when the modality is shuffled")
    print("XAI: occlusion maps ...")
    occl = occlusion_maps(final, X, y, imgs)

    contrib = final.stacking_view_contributions(X)
    contrib.assign(file=imgs.file.values, label=y).to_csv(OUT / "xai_late_fusion_contributions.csv", index=False)

    save_json({
        "task": "dysgraphia", "n_samples": int(len(y)), "n_positive": int(y.sum()),
        "views": {k: len(v) for k, v in views.items()},
        "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
        **rep,
        "xai": {"geometry_top_features": geo_imp.head(10).to_dict(),
                "geometry_direction_corr": {k: geo_dir[k] for k in geo_imp.head(10).index},
                "grouped_permutation_auc_drop": gpi, "occlusion_examples": occl},
        "runtime_sec": time.time() - t0,
    }, OUT / "summary.json")
    print(f"Done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
