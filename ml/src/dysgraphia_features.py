"""Image preprocessing and the two dysgraphia modalities.

1. Handwriting-geometry features - interpretable measures motivated by clinical
   dysgraphia markers (letter-size variability, baseline drift, spacing
   irregularity, slant variability, stroke-width variability, contour tremor).
   All lengths are normalised by the writer's median letter height so that
   image resolution cannot leak into the features.
2. Deep visual features - EfficientNet-B0 (ImageNet) embeddings averaged over
   overlapping square tiles of the handwriting strip.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import CACHE_DIR, DATA_DIR, ensure_dir

IMG_DIR = DATA_DIR / "DATASET DYSGRAPHIA HANDWRITING"
CLASSES = {"Potential Dysgraphia": 1, "Low Potential Dysgraphia": 0}
TILE = 224


# ---------------------------------------------------------------------------- data
def list_images() -> pd.DataFrame:
    rows = []
    for cls, label in CLASSES.items():
        for p in sorted((IMG_DIR / cls).glob("*.jpg")):
            rows.append({"path": str(p), "file": p.name, "label": label})
    return pd.DataFrame(rows)


def read_gray(path: str) -> np.ndarray:
    # np.fromfile + imdecode handles non-ASCII / space-containing Windows paths
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def preprocess(gray: np.ndarray) -> np.ndarray:
    """Return a clean binary mask (ink=1) with ruled lines and specks removed, cropped to the ink."""
    if gray.mean() > 127:  # make ink white on black
        gray = 255 - gray
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape

    # remove long horizontal ruled lines (they are paper, not handwriting)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 8), 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    bw = cv2.subtract(bw, lines)
    # strokes that crossed a ruled line are now cut; bridge gaps up to the line thickness
    lm = lines > 0
    cols = lm.any(0)
    if cols.any():
        starts = (np.diff(lm[:, cols].astype(np.int8), axis=0) == 1).sum(0) + lm[0, cols]
        t = int(np.median(lm[:, cols].sum(0) / np.maximum(starts, 1)))
    else:
        t = 0
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, t + 4))))

    # remove specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(12, 0.00005 * h * w)
    bw = (keep[lab]).astype(np.uint8)

    return normalize_scale(_crop(bw))


def _crop(bw, m=4):
    ys, xs = np.nonzero(bw)
    if len(xs) == 0:
        return bw
    return bw[max(ys.min() - m, 0): ys.max() + m + 1, max(xs.min() - m, 0): xs.max() + m + 1]


CANONICAL_LETTER_HEIGHT = 40


def normalize_scale(bw: np.ndarray, target: int = CANONICAL_LETTER_HEIGHT) -> np.ndarray:
    """Rescale so the median connected-component (letter) height equals `target` px.

    The raw scans have unknown and varying resolution; raw letter height in pixels alone
    separates the classes with AUC ~0.90, which could be a scanning artefact rather than
    handwriting. Canonical scaling removes that shortcut from BOTH modalities.
    """
    _, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if len(st) < 2:
        return bw
    ref = float(np.median(st[1:, cv2.CC_STAT_HEIGHT]))
    s = target / max(ref, 1.0)
    h, w = bw.shape
    out = cv2.resize(bw.astype(np.float32), (max(1, round(w * s)), max(1, round(h * s))),
                     interpolation=cv2.INTER_LINEAR)
    out = cv2.GaussianBlur(out, (3, 3), 0)  # equalise edge smoothness of up- vs down-sampled scans
    return _crop((out > 0.25).astype(np.uint8))


# ---------------------------------------------------------------------------- handcrafted
def _cv(a):
    a = np.asarray(a, float)
    return float(a.std() / a.mean()) if len(a) > 1 and a.mean() > 0 else 0.0


def _line_residual(x, y):
    if len(x) < 3:
        return 0.0, 0.0
    coef = np.polyfit(x, y, 1)
    return float(np.std(y - np.polyval(coef, x))), float(np.degrees(np.arctan(coef[0])))


GEOMETRY_FEATURES = [
    "letter_height_cv", "letter_width_cv", "letter_area_cv", "letter_height_iqr_ratio",
    "baseline_residual", "topline_residual", "midline_residual", "baseline_skew_abs_deg",
    "text_band_height", "letters_per_width", "sentence_length_in_letter_heights", "ink_density",
    "stroke_width", "stroke_width_cv", "gap_count_per_width", "gap_mean", "gap_cv", "gap_max",
    "word_gap_count", "slant_mean_deg", "slant_std_deg", "contour_convexity_mean",
    "contour_convexity_cv", "contour_tremor", "fragmentation",
]


def geometry_features(bw: np.ndarray) -> dict:
    H, W = bw.shape
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    st = stats[1:]
    areas = st[:, cv2.CC_STAT_AREA]
    big = areas >= max(20, 0.08 * np.median(areas)) if len(areas) else np.array([], bool)
    st, cent_b = st[big], cent[1:][big]
    if len(st) < 3:
        return {f: 0.0 for f in GEOMETRY_FEATURES}
    hts, wds = st[:, cv2.CC_STAT_HEIGHT].astype(float), st[:, cv2.CC_STAT_WIDTH].astype(float)
    ref = float(np.median(hts))  # writer's typical letter/blob height -> scale unit
    x_c = cent_b[:, 0]
    tops = st[:, cv2.CC_STAT_TOP].astype(float)
    bottoms = tops + hts

    base_res, base_ang = _line_residual(x_c, bottoms)
    top_res, _ = _line_residual(x_c, tops)
    mid_res, _ = _line_residual(x_c, cent_b[:, 1])

    ys = np.repeat(np.arange(H), bw.sum(1).astype(np.int64))
    band = (np.percentile(ys, 95) - np.percentile(ys, 5)) if len(ys) else 0

    # stroke width from distance transform ridge
    dt = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3)))) & (bw > 0)
    sw = 2 * dt[ridge] if ridge.any() else np.array([1.0])

    # horizontal gaps (column projection)
    colink = bw.sum(0) > 0
    gaps, run = [], 0
    for v in colink:
        if not v:
            run += 1
        else:
            if run:
                gaps.append(run)
            run = 0
    gaps = np.array(gaps, float) / ref if gaps else np.array([0.0])

    # slant & contour shape per component
    slants, convex, tremor = [], [], []
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 0.6 * ref or len(c) < 10:
            continue
        mo = cv2.moments(c)
        if mo["mu20"] + mo["mu02"] > 0:
            theta = 0.5 * np.degrees(np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"]))
            slants.append(90 - abs(theta) if h >= w else np.nan)
        per = cv2.arcLength(c, True)
        hull_per = cv2.arcLength(cv2.convexHull(c), True)
        if per > 0:
            convex.append(hull_per / per)
            approx = cv2.approxPolyDP(c, 1.5, True)
            tremor.append(len(approx) / (per / ref))
    slants = np.array([s for s in slants if not np.isnan(s)]) if slants else np.array([0.0])

    width_units = W / ref
    return {
        "letter_height_cv": _cv(hts),
        "letter_width_cv": _cv(wds),
        "letter_area_cv": _cv(areas[big]),
        "letter_height_iqr_ratio": float((np.percentile(hts, 75) - np.percentile(hts, 25)) / ref),
        "baseline_residual": base_res / ref,
        "topline_residual": top_res / ref,
        "midline_residual": mid_res / ref,
        "baseline_skew_abs_deg": abs(base_ang),
        "text_band_height": band / ref,
        "letters_per_width": len(st) / width_units,
        "sentence_length_in_letter_heights": width_units,
        "ink_density": float(bw.mean()),
        "stroke_width": float(np.median(sw) / ref),
        "stroke_width_cv": _cv(sw),
        "gap_count_per_width": len(gaps) / width_units,
        "gap_mean": float(gaps.mean()),
        "gap_cv": _cv(gaps),
        "gap_max": float(gaps.max()),
        "word_gap_count": float((gaps > 0.5).sum()),
        "slant_mean_deg": float(np.mean(slants)),
        "slant_std_deg": float(np.std(slants)),
        "contour_convexity_mean": float(np.mean(convex)) if convex else 0.0,
        "contour_convexity_cv": _cv(convex) if convex else 0.0,
        "contour_tremor": float(np.mean(tremor)) if tremor else 0.0,
        "fragmentation": float((areas < 0.08 * np.median(areas)).sum() / max(len(areas), 1)),
    }


# ---------------------------------------------------------------------------- deep features
def to_rgb_strip(bw: np.ndarray, height: int = TILE) -> np.ndarray:
    """Black ink on white, letters kept at canonical size, vertically centred in a `height` canvas."""
    img = ((1 - bw) * 255).astype(np.uint8)
    h, w = img.shape
    if h > height:  # unusually tall writing: shrink to fit (keeps aspect)
        img = cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
        h, w = img.shape
    canvas = np.full((height, max(w, height)), 255, np.uint8)
    top = (height - h) // 2
    canvas[top: top + h, :w] = img
    return np.repeat(canvas[..., None], 3, axis=2)


def tile_starts(w: int, stride: int = TILE // 2) -> list[int]:
    starts = list(range(0, max(w - TILE, 0) + 1, stride))
    if starts[-1] + TILE < w:
        starts.append(w - TILE)
    return starts


def tiles(strip: np.ndarray, stride: int = TILE // 2) -> np.ndarray:
    out = []
    for s in tile_starts(strip.shape[1], stride):
        t = strip[:, s: s + TILE]
        if t.shape[1] < TILE:
            t = np.pad(t, ((0, 0), (0, TILE - t.shape[1]), (0, 0)), constant_values=255)
        out.append(t)
    return np.stack(out)


_CNN = None


def cnn():
    global _CNN
    if _CNN is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        _CNN = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")
    return _CNN


def embed_strip(strip: np.ndarray) -> np.ndarray:
    t = tiles(strip).astype("float32")  # EfficientNet in Keras rescales 0-255 internally
    return cnn().predict(t, verbose=0, batch_size=32).mean(0)


DEEP_PREFIX = "cnn_"


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Geometry + deep features for every image, cached on disk."""
    key = hashlib.md5("|".join(df.file).encode()).hexdigest()[:10]
    cache = ensure_dir(CACHE_DIR) / f"dysgraphia_features_{key}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    geo, deep = [], []
    for i, p in enumerate(df.path):
        bw = preprocess(read_gray(p))
        geo.append(geometry_features(bw))
        deep.append(embed_strip(to_rgb_strip(bw)))
        if (i + 1) % 25 == 0:
            print(f"    features: {i + 1}/{len(df)}")
    G = pd.DataFrame(geo, index=df.index)[GEOMETRY_FEATURES]
    D = pd.DataFrame(np.vstack(deep), index=df.index, columns=[f"{DEEP_PREFIX}{j}" for j in range(1280)])
    out = pd.concat([G, D], axis=1)
    out.to_pickle(cache)
    return out
