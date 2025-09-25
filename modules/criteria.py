from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _counts(y: pd.Series) -> np.ndarray:
    vals, counts = np.unique(y.dropna(), return_counts=True)
    return counts.astype(float)


def gini_impurity(y: pd.Series) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    p = _counts(y) / n
    return float(1.0 - np.sum(p**2))


def entropy(y: pd.Series, eps: float = 1e-12) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    p = _counts(y) / n
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def info_gain(parent: pd.Series, left: pd.Series, right: pd.Series) -> float:
    H = entropy(parent)
    n = len(parent)
    w = (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)
    return float(H - w)


def split_info(left_n: int, right_n: int) -> float:
    n = left_n + right_n
    if n == 0 or left_n == 0 or right_n == 0:
        return 0.0
    p_l = left_n / n
    p_r = right_n / n
    return float(-(p_l * np.log2(p_l) + p_r * np.log2(p_r)))


def gain_ratio(parent: pd.Series, left: pd.Series, right: pd.Series) -> float:
    ig = info_gain(parent, left, right)
    si = split_info(len(left), len(right))
    if si == 0.0:
        return 0.0
    return float(ig / si)


def best_split_numeric(x: pd.Series, y: pd.Series, min_leaf: int = 1) -> Optional[Dict]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    df = df.sort_values("x")
    if df["x"].nunique() < 2:
        return None
    xs = df["x"].values
    ys = df["y"].values
    candidates = (xs[:-1] + xs[1:]) / 2.0
    rows = []
    for thr in np.unique(candidates):
        left_idx = xs <= thr
        right_idx = ~left_idx
        if left_idx.sum() < min_leaf or right_idx.sum() < min_leaf:
            continue
        y_left, y_right = ys[left_idx], ys[right_idx]
        g_left = gini_impurity(pd.Series(y_left))
        g_right = gini_impurity(pd.Series(y_right))
        weighted_gini = (len(y_left) * g_left + len(y_right) * g_right) / len(ys)
        ig = info_gain(pd.Series(ys), pd.Series(y_left), pd.Series(y_right))
        gr = gain_ratio(pd.Series(ys), pd.Series(y_left), pd.Series(y_right))
        rows.append({"threshold": thr, "weighted_gini": weighted_gini, "gain": ig, "gain_ratio": gr})
    if not rows:
        return None
    table = pd.DataFrame(rows).sort_values(["weighted_gini", "gain"], ascending=[True, False])
    # Plot preview
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(table["threshold"], table["weighted_gini"], label="Gini ponderado", color="#4F46E5")
    ax2 = ax.twinx()
    ax2.plot(table["threshold"], table["gain"], label="Gain", color="#14B8A6")
    ax.set_xlabel("Umbral")
    ax.set_title("Evaluación de splits numéricos")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return {
        "table": table.reset_index(drop=True),
        "best_gini_threshold": float(table.loc[table["weighted_gini"].idxmin(), "threshold"]),
        "best_gain_threshold": float(table.loc[table["gain"].idxmax(), "threshold"]),
        "best_gain_ratio_threshold": float(table.loc[table["gain_ratio"].idxmax(), "threshold"]),
        "figure": fig,
    }


def best_split_categorical(x: pd.Series, y: pd.Series, min_leaf: int = 1) -> Optional[Dict]:
    df = pd.DataFrame({"x": x.astype(str), "y": y}).dropna()
    cats = df["x"].unique()
    if len(cats) < 2:
        return None
    rows = []
    for cat in cats:
        left = df[df["x"] == cat]["y"]
        right = df[df["x"] != cat]["y"]
        if len(left) < min_leaf or len(right) < min_leaf:
            continue
        g_left = gini_impurity(left)
        g_right = gini_impurity(right)
        weighted_gini = (len(left) * g_left + len(right) * g_right) / len(df)
        ig = info_gain(df["y"], left, right)
        gr = gain_ratio(df["y"], left, right)
        rows.append({"category": cat, "weighted_gini": weighted_gini, "gain": ig, "gain_ratio": gr})
    if not rows:
        return None
    table = pd.DataFrame(rows).sort_values(["weighted_gini", "gain"], ascending=[True, False])
    return {
        "table": table.reset_index(drop=True),
        "best_gini_category": str(table.loc[table["weighted_gini"].idxmin(), "category"]),
        "best_gain_category": str(table.loc[table["gain"].idxmax(), "category"]),
        "best_gain_ratio_category": str(table.loc[table["gain_ratio"].idxmax(), "category"]),
    }


