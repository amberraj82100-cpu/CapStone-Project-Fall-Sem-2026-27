"""SYNTHETIC dyscalculia game-telemetry generator.

!!! The data produced here is simulated. It exists to (1) build and test the dyscalculia pipeline end-to-end
!!! before real game data exists and (2) check that the XAI recovers the effects we planted. Accuracy measured
!!! on it says nothing about real children and must never be reported as a clinical result.

Generative model (per child):
  latent abilities z = [ans, symb, speed, attn, subit] ~ N(shift, Sigma), age-standardised
    ans   - approximate number system acuity     symb  - symbolic number / arithmetic fluency
    speed - general processing speed             attn  - sustained attention
    subit - subitizing (small-set enumeration)
  Dyscalculia (DD) shifts, in SD units (assumptions informed by the literature, see docs):
    ans -1.0 (impaired ANS), symb -1.4 (core arithmetic deficit), speed -0.3, attn -0.2,
    subit -0.3 (subitizing largely intact, cf. Decarli et al. 2020)
  Confounders: 8 % of non-DD children have attention difficulties (attn -1.5) -> low scores without DD.
  Label noise: 5 % of labels flipped (screening labels are imperfect).
Trials are simulated through the five game tasks (Q11-Q15) with age-dependent development.

Run:  python ml/src/dyscalculia_synthetic.py      -> datasets/dyscalculia_synthetic/
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from common import DATA_DIR, ensure_dir

PARAMS = {
    "n_children": 3000,
    "prevalence": 0.10,
    "label_noise": 0.05,
    "confounder_rate": 0.08,
    "seed": 2026,
    "dd_shift": {"ans": -1.0, "symb": -1.4, "speed": -0.3, "attn": -0.2, "subit": -0.3},
    "trials": {"subitize": 10, "magnitude": 16, "numberline": 10, "sequence": 6, "arithmetic": 10},
    "numberline_max": 20,
}
LATENTS = ["ans", "symb", "speed", "attn", "subit"]
OUT = DATA_DIR / "dyscalculia_synthetic"


def _cov():
    c = np.full((5, 5), 0.3)
    np.fill_diagonal(c, 1.0)
    c[0, 1] = c[1, 0] = 0.45  # ANS and symbolic skills are linked
    return c


def simulate_child(cid, age, z, rng, P):
    ans, symb, speed, attn, subit = z
    rows = []
    lapse = 0.03 + 0.05 * max(0.0, -attn)
    rt_noise = 0.25 * (1 + 0.3 * max(0.0, -attn))
    w = (0.30 - 0.025 * (age - 6)) * np.exp(-0.30 * ans)  # Weber fraction: develops with age

    # Q11 subitizing
    span = np.clip(3.6 + 0.5 * subit + 0.1 * (age - 8), 2, 6)
    base = (650 + 60 * (11 - age)) * np.exp(-0.15 * speed)
    for k, n in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 4][: P["trials"]["subitize"]]):
        if n <= span and rng.random() > lapse:
            resp = n
        else:
            resp = max(1, int(round(n * np.exp(rng.normal(0, 1.1 * w)))))
        rt = (base + (50 * n if n <= span else 50 * span + 300 * (n - span))) * rng.lognormal(0, rt_noise)
        rows.append(dict(child_id=cid, task="subitize", trial=k, n_dots=n, response=resp, correct=int(resp == n), rt_ms=rt))

    # Q12 non-symbolic magnitude comparison
    ratios = [0.5, 0.6, 0.67, 0.75, 0.8, 0.86, 0.9, 0.92]
    for k in range(P["trials"]["magnitude"]):
        r = ratios[k % len(ratios)]
        big = int(rng.integers(10, 25))
        small = max(1, int(round(big * r)))
        left, right = (big, small) if rng.random() < 0.5 else (small, big)
        d = abs(left - right) / np.sqrt(left ** 2 + right ** 2)
        p = lapse / 2 + (1 - lapse) * norm.cdf(d / w)
        rt = (900 + 80 * (11 - age)) * np.exp(-0.15 * speed) * (1 + 0.6 * r) * rng.lognormal(0, rt_noise)
        rows.append(dict(child_id=cid, task="magnitude", trial=k, left=left, right=right,
                         correct=int(rng.random() < p), rt_ms=rt))

    # Q13 number line 0-L (log-to-linear developmental shift)
    L = P["numberline_max"]
    lam = expit(1.2 * (age - 7.0) + 0.9 * symb + 0.5 * ans)
    sd = 1.6 * np.exp(-0.35 * ans) * (1 + 0.3 * max(0.0, -attn))
    for k, x in enumerate(rng.choice(np.arange(1, L), P["trials"]["numberline"], replace=False)):
        logmap = L * np.log(x + 1) / np.log(L + 1)
        est = float(np.clip(lam * x + (1 - lam) * logmap + rng.normal(0, sd), 0, L))
        rt = 2500 * np.exp(-0.15 * speed) * rng.lognormal(0, rt_noise)
        rows.append(dict(child_id=cid, task="numberline", trial=k, target=int(x), estimate=est, line_max=L,
                         correct=int(abs(est - x) <= 1.5), rt_ms=rt))

    # Q14 sequence completion, Q15 visual arithmetic
    for task, n_tr, a, b, c_age, c_att, rt0, rt_s in [
        ("sequence", P["trials"]["sequence"], 1.4, 1.1, 0.35, 0.4, 4000, 0.20),
        ("arithmetic", P["trials"]["arithmetic"], 1.6, 1.4, 0.40, 0.3, 3000, 0.30),
    ]:
        p = expit(a + b * symb + c_age * (age - 8) + c_att * attn)
        for k in range(n_tr):
            ok = int(rng.random() < p * (1 - lapse))
            rt = rt0 * np.exp(-rt_s * symb - 0.15 * speed) * rng.lognormal(0, rt_noise)
            rows.append(dict(child_id=cid, task=task, trial=k, correct=ok, rt_ms=rt))
    return rows


def generate(P=PARAMS):
    rng = np.random.default_rng(P["seed"])
    n = P["n_children"]
    dd = rng.random(n) < P["prevalence"]
    confound = (~dd) & (rng.random(n) < P["confounder_rate"])
    shift = np.array([P["dd_shift"][k] for k in LATENTS])
    Z = rng.multivariate_normal(np.zeros(5), _cov(), n) + np.outer(dd, shift)
    Z[confound, LATENTS.index("attn")] -= 1.5
    age = np.round(rng.uniform(6, 11.99, n), 2)
    gender = rng.choice(["F", "M"], n)

    trials = []
    for i in range(n):
        trials += simulate_child(i, age[i], Z[i], rng, P)
    flip = rng.random(n) < P["label_noise"]
    label = np.where(flip, ~dd, dd).astype(int)

    children = pd.DataFrame({"child_id": np.arange(n), "age": age, "gender": gender, "dyscalculia": label})
    truth = pd.DataFrame(Z, columns=[f"latent_{k}" for k in LATENTS]).assign(
        child_id=np.arange(n), true_dd=dd.astype(int), attention_confounder=confound.astype(int), label_flipped=flip.astype(int))
    return children, pd.DataFrame(trials), truth


def main():
    ensure_dir(OUT)
    children, trials, truth = generate()
    children.to_csv(OUT / "children.csv", index=False)
    trials.to_csv(OUT / "trials.csv", index=False)
    truth.to_csv(OUT / "ground_truth_DO_NOT_TRAIN.csv", index=False)
    (OUT / "generator_params.json").write_text(json.dumps(PARAMS, indent=2))
    (OUT / "README.md").write_text(
        "# dyscalculia_synthetic - SIMULATED DATA (not real children)\n\n"
        "Generated by `ml/src/dyscalculia_synthetic.py` (seed in `generator_params.json`). "
        "Use only to develop and test the dyscalculia pipeline before real game data exists. "
        "**Never report accuracy on this data as a real-world or clinical result.**\n\n"
        "* `children.csv` - child_id, age, gender, dyscalculia label (5 % label noise)\n"
        "* `trials.csv` - trial-level logs of the five game tasks (Q11 subitize, Q12 magnitude, Q13 numberline, "
        "Q14 sequence, Q15 arithmetic), in the same shape the game should export\n"
        "* `ground_truth_DO_NOT_TRAIN.csv` - latent abilities and true status, used only to verify the XAI\n", encoding="utf-8")
    print(f"children={len(children)} DD={children.dyscalculia.sum()} trials={len(trials)} -> {OUT}")


if __name__ == "__main__":
    main()
