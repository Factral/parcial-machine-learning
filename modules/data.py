from __future__ import annotations

import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, sep=";")
    return df


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if df.empty:
        return [], []
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio < 0.3:
            df[col] = df[col].astype("category")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def dataframe_info(df: pd.DataFrame) -> str:
    if df.empty:
        return "DataFrame vacío"
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({"faltantes": miss, "%": pct}).sort_values("faltantes", ascending=False)
    out.index.name = "columna"
    return out


def head_tail_describe(df: pd.DataFrame, head_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df, df
    # Pandas versions < 1.5 don't support datetime_is_numeric
    try:
        desc = df.describe(include="all", datetime_is_numeric=True).T
    except TypeError:
        desc = df.describe(include="all").T
    return df.head(head_n), df.tail(5), desc


