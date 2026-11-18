"""SIMULATED game sessions for testing the scoring + retraining loop end to end.

Produces sessions in the same JSON shape the Java backend stores, including the two new tasks:
  Q16 copy-the-sentence  -> rawData.handwritingImage (a real image from the dysgraphia dataset, as a PNG data URL)
  Q17 number-sense round -> rawData.trials (trial logs from the dyscalculia simulator)
and labels from a "specialist" (70 %) or "parent" (30 %). Never use these to report accuracy.
"""
from __future__ import annotations

import base64
import random
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from scipy.special import expit

import dysgraphia_features as DGF
from dyscalculia_synthetic import LATENTS, PARAMS, simulate_child

PREVALENCE = {"DYSLEXIA": 0.25, "DYSGRAPHIA": 0.45, "DYSCALCULIA": 0.25}


def _png_data_url(path: str, max_w: int = 900) -> str:
    g = DGF.read_gray(path)
    if g.mean() < 127:  # dataset is white ink on black; the game canvas is dark ink on white
        g = 255 - g
    if g.shape[1] > max_w:
        g = cv2.resize(g, (max_w, int(g.shape[0] * max_w / g.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", g)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def simulate(n: int = 160, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    imgs = DGF.list_images()
    pd_imgs, lpd_imgs = list(imgs[imgs.label == 1].path), list(imgs[imgs.label == 0].path)
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    sessions = []
    for i in range(n):
        age = int(rng.integers(6, 12))
        risk = {d: int(rng.random() < p) for d, p in PREVALENCE.items()}
        resp = []
        # Q1-Q5 dyslexia, Q6-Q10 dysgraphia, Q11-Q15 dyscalculia (one item each, as in the game)
        for q in range(1, 16):
            dom = "DYSLEXIA" if q <= 5 else "DYSGRAPHIA" if q <= 10 else "DYSCALCULIA"
            p_ok = expit(1.6 + 0.25 * (age - 8) - 1.6 * risk[dom] + rng.normal(0, 0.6))
            t_ms = 4000 * np.exp(0.45 * risk[dom] - 0.06 * (age - 8) + rng.normal(0, 0.35))
            r = {"questionId": q, "domain": dom, "isCorrect": bool(rng.random() < p_ok), "timeTakenMs": int(t_ms),
                 "motorJitterScore": 0.0, "motorHesitationCount": 0, "reversalDetected": False, "magnitudeDistance": 0.0}
            if dom == "DYSGRAPHIA":
                r["motorJitterScore"] = float(max(0, rng.normal(10 + 8 * risk[dom], 4)))
                r["motorHesitationCount"] = int(rng.poisson(1 + 2 * risk[dom]))
            if q == 1:
                r["reversalDetected"] = bool(rng.random() < 0.15 + 0.4 * risk["DYSLEXIA"])
            if q == 13:
                r["magnitudeDistance"] = float(abs(rng.normal(1.5 + 2.5 * risk["DYSCALCULIA"], 1.2)))
            resp.append(r)
        # Q16 copy-the-sentence (90 % of children complete it)
        if rng.random() < 0.9:
            path = random.choice(pd_imgs if risk["DYSGRAPHIA"] else lpd_imgs)
            resp.append({"questionId": 16, "domain": "DYSGRAPHIA", "isCorrect": True, "timeTakenMs": 30000,
                         "rawData": {"handwritingImage": _png_data_url(path)}})
        # Q17 number-sense round
        z = rng.multivariate_normal(np.zeros(5), np.eye(5) * 0.8 + 0.2)
        if risk["DYSCALCULIA"]:
            z = z + np.array([PARAMS["dd_shift"][k] for k in LATENTS])
        trials = simulate_child(0, age + 0.5, z, rng, PARAMS)
        resp.append({"questionId": 17, "domain": "DYSCALCULIA", "isCorrect": True, "timeTakenMs": 180000,
                     "rawData": {"trials": [{k: (v.item() if hasattr(v, "item") else v) for k, v in t.items()
                                             if k != "child_id"} for t in trials]}})
        done = t0 + timedelta(hours=6 * i)
        labels = {}
        for d in PREVALENCE:
            if rng.random() < 0.85:  # not every child gets a confirmed outcome
                src = "specialist" if rng.random() < 0.7 else "parent"
                val = risk[d] if src == "specialist" or rng.random() > 0.1 else 1 - risk[d]  # parents are noisier
                labels[d] = {"value": int(val), "source": src, "at": (done + timedelta(days=14)).isoformat()}
        sessions.append({"id": f"sim-{i:04d}", "childName": f"Simulated {i}", "childAge": age,
                         "status": "COMPLETED", "completedAt": done.isoformat(), "responses": resp, "labels": labels,
                         "simulated": True})
    return sessions
