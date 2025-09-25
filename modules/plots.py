from __future__ import annotations

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def _setup_style() -> None:
    sns.set_theme(style="whitegrid")
    sns.set_palette(["#4F46E5", "#14B8A6", "#F59E0B", "#EF4444", "#10B981"]) 


def plot_numeric_histograms(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, bins: int = 25):
    _setup_style()
    cols = list(columns) if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        return None
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(3, 3.5 * nrows)))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, col in enumerate(cols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
        ax.set_title(col)
    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')
    fig.tight_layout()
    return fig


def plot_categorical_bars(df: pd.DataFrame, columns: Optional[Iterable[str]] = None, top_n: int = 12):
    _setup_style()
    cols = list(columns) if columns else df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cols:
        return None
    n = len(cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(3, 3.8 * nrows)))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, col in enumerate(cols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        vc = df[col].astype(str).value_counts().nlargest(top_n)
        sns.barplot(x=vc.values, y=vc.index, ax=ax, orient="h")
        ax.set_title(col)
        ax.set_xlabel("Frecuencia")
        ax.set_ylabel("")
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    _setup_style()
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    try:
        corr = num.corr(numeric_only=True)
    except TypeError:
        corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=False, cmap="viridis", ax=ax)
    ax.set_title("Matriz de correlación (variables numéricas)")
    fig.tight_layout()
    return fig


# ---- Mixed-type association matrix (numeric + categorical) ----
def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    x = x.astype(str)
    y = y.astype(str)
    tbl = pd.crosstab(x, y)
    if tbl.size == 0:
        return np.nan
    chi2 = chi2_contingency(tbl, correction=False)[0]
    n = tbl.values.sum()
    phi2 = chi2 / n
    r, k = tbl.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1)) if n > 1 else 0.0
    rcorr = r - (r - 1) ** 2 / (n - 1) if n > 1 else r
    kcorr = k - (k - 1) ** 2 / (n - 1) if n > 1 else k
    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    # Eta for numeric (measurements) vs categorical (categories)
    try:
        cat = categories.astype(str)
        y = pd.to_numeric(measurements, errors="coerce")
        valid = (~cat.isna()) & (~y.isna())
        cat = cat[valid]
        y = y[valid]
        if y.empty or cat.nunique() < 2:
            return np.nan
        groups = [y[cat == g] for g in cat.unique()]
        ss_between = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
        ss_total = ((y - y.mean()) ** 2).sum()
        return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan
    except Exception:
        return np.nan


def association_matrix(df: pd.DataFrame, max_categories: int = 30) -> pd.DataFrame:
    cols = list(df.columns)
    n = len(cols)
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    dtypes = {c: ("num" if pd.api.types.is_numeric_dtype(df[c]) else "cat") for c in cols}
    for i, c1 in enumerate(cols):
        out.loc[c1, c1] = 1.0
        for j in range(i + 1, n):
            c2 = cols[j]
            t1, t2 = dtypes[c1], dtypes[c2]
            s1, s2 = df[c1], df[c2]
            val = np.nan
            if t1 == "num" and t2 == "num":
                try:
                    val = float(s1.corr(s2))
                except Exception:
                    val = np.nan
            elif t1 == "cat" and t2 == "cat":
                # Prevent explosion with high-cardinality
                if s1.nunique(dropna=True) <= max_categories and s2.nunique(dropna=True) <= max_categories:
                    val = _cramers_v(s1, s2)
            else:
                # numeric-categorical
                num = s1 if t1 == "num" else s2
                cat = s2 if t1 == "num" else s1
                if cat.nunique(dropna=True) <= max_categories:
                    val = _correlation_ratio(cat, num)
            out.loc[c1, c2] = val
            out.loc[c2, c1] = val
    return out


