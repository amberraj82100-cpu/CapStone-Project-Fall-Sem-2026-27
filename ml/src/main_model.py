"""Main screening model: one entry point that scores a child for dyslexia, dysgraphia and dyscalculia.

No dataset contains the same children for more than one disorder, so a single jointly-trained network is not
possible yet. Instead, this system wraps each disorder's best validated fusion model, maps its probability to the
backend's risk levels (LOW_RISK / MODERATE_TENDENCY / HIGH_RISK), attaches an explanation, and synthesises an
overall risk with the same rule as AssessmentScoringService.synthesizeOverall(). Each disorder is optional:
a child who skipped the handwriting task still gets the other two scores.

Build (after the three pipelines):   python ml/src/main_model.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from common import MODELS_DIR, RESULTS_DIR, save_json
from model_selection import choose

# which saved bundle + results folder serves each disorder
DISORDERS = {
    "DYSLEXIA": {"bundle": "dyslexia_fusion.joblib", "results": "dyslexia", "synthetic": False},
    "DYSGRAPHIA": {"bundle": "dysgraphia_fusion.joblib", "results": "dysgraphia", "synthetic": False},
    "DYSCALCULIA": {"bundle": "dyscalculia_synthetic_fusion.joblib", "results": "dyscalculia_synthetic", "synthetic": True},
}
HIGH_RISK_SPECIFICITY = 0.95  # HIGH_RISK only when p exceeds what 95 % of true negatives score


@dataclass
class DisorderModel:
    name: str
    fusion: object
    model_name: str
    moderate_threshold: float
    high_threshold: float
    synthetic: bool
    cv_metrics: dict

    def score(self, X: pd.DataFrame) -> list[dict]:
        p = self.fusion.predict_proba_all(X)[self.model_name]
        contrib = self.fusion.stacking_view_contributions(X)
        out = []
        for i in range(len(X)):
            level = ("HIGH_RISK" if p[i] >= self.high_threshold else
                     "MODERATE_TENDENCY" if p[i] >= self.moderate_threshold else "LOW_RISK")
            c = contrib.iloc[i].sort_values(ascending=False)
            out.append({
                "disorder": self.name, "probability": float(p[i]), "risk_level": level,
                "model": self.model_name, "synthetic_training_data": self.synthetic,
                "modality_contributions_logodds": {k: round(float(v), 3) for k, v in c.items()},
                "top_risk_modality": c.index[0] if c.iloc[0] > 0 else None,
            })
        return out


class ScreeningSystem:
    def __init__(self, models: dict[str, DisorderModel]):
        self.models = models

    @classmethod
    def build(cls) -> "ScreeningSystem":
        models = {}
        for name, cfg in DISORDERS.items():
            path = MODELS_DIR / cfg["bundle"]
            if not path.exists():
                print(f"  ! {name}: {path.name} missing - run its pipeline first")
                continue
            fusion = joblib.load(path)["fusion"]
            sel = choose(cfg["results"])
            best = sel["chosen_model"]
            summ = pd.read_csv(RESULTS_DIR / cfg["results"] / "metrics_summary.csv").set_index("model")
            oof = pd.read_csv(RESULTS_DIR / cfg["results"] / "oof_predictions.csv")
            neg = oof[(oof.model == best) & (oof.y == 0)].p
            moderate = float(fusion.thresholds_[best])
            high = max(float(np.quantile(neg, HIGH_RISK_SPECIFICITY)), moderate + 1e-6)
            models[name] = DisorderModel(name, fusion, best, moderate, high, cfg["synthetic"],
                                         {**summ.loc[best].to_dict(), "selection_reason": sel["reason"],
                                          "top_scoring_model": sel["top_model"]})
        return cls(models)

    def screen(self, inputs: dict[str, pd.DataFrame]) -> list[dict]:
        """inputs: {disorder: feature DataFrame (rows = children, same order)}; any disorder may be missing."""
        n = max(len(v) for v in inputs.values())
        per = {d: self.models[d].score(X) for d, X in inputs.items() if d in self.models}
        reports = []
        for i in range(n):
            r = {d: per[d][i] for d in per}
            levels = [x["risk_level"] for x in r.values()]
            high, mod = levels.count("HIGH_RISK"), levels.count("MODERATE_TENDENCY")
            overall = "HIGH_RISK" if high else "MODERATE_TENDENCY" if mod else "LOW_RISK"
            reports.append({"overall_risk_level": overall, "domains": r,
                            "note": "Screening aid only - not a diagnosis."})
        return reports

    def card(self) -> dict:
        return {d: {"model": m.model_name, "moderate_threshold": m.moderate_threshold,
                    "high_threshold": m.high_threshold, "synthetic_training_data": m.synthetic,
                    "cv_roc_auc": m.cv_metrics["roc_auc"], "cv_balanced_accuracy": m.cv_metrics["balanced_accuracy"],
                    "cv_sensitivity": m.cv_metrics["sensitivity"], "cv_specificity": m.cv_metrics["specificity"],
                    "top_scoring_model": m.cv_metrics["top_scoring_model"],
                    "selection_reason": m.cv_metrics["selection_reason"]}
                for d, m in self.models.items()}


def main():
    system = ScreeningSystem.build()
    card = system.card()
    save_json(card, MODELS_DIR / "screening_system_card.json")
    print(json.dumps(card, indent=2))
    # smoke test: score one real dyslexia child and one synthetic dyscalculia child together
    from dyslexia_pipeline import load_data
    import dyscalculia_features as F
    from common import DATA_DIR
    dl = load_data().drop(columns=["Dyslexia"]).iloc[[0]]
    t = pd.read_csv(DATA_DIR / "dyscalculia_synthetic" / "trials.csv")
    c = pd.read_csv(DATA_DIR / "dyscalculia_synthetic" / "children.csv").set_index("child_id")
    dc = F.build_feature_table(t[t.child_id == 0])
    dc["age"], dc["gender_male"] = c.loc[0, "age"], int(c.loc[0, "gender"] == "M")
    print(json.dumps(system.screen({"DYSLEXIA": dl.reset_index(drop=True), "DYSCALCULIA": dc.reset_index(drop=True)}), indent=2))


if __name__ == "__main__":
    main()
