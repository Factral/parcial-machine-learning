from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.express as px


def build_preprocessor(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return X, y, preprocessor


def registry(task: str = "classification") -> Dict[str, Dict]:
    if task == "classification":
        return {
            "Decision Tree": {
                "estimator": DecisionTreeClassifier,
                "help": "Árbol de decisión; divide usando Gini o Entropía (Gain).",
                "params": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": (1, 30),
                    "min_samples_split": (2, 20),
                    "min_samples_leaf": (1, 20),
                },
            },
            "Bagging": {
                "estimator": BaggingClassifier,
                "help": "Ensamble de bootstrap sobre un estimador base (árbol por defecto).",
                "params": {
                    "n_estimators": (10, 300),
                    "max_samples": (0.1, 1.0),
                    "max_features": (0.1, 1.0),
                },
            },
            "AdaBoost": {
                "estimator": AdaBoostClassifier,
                "help": "Boosting secuencial que ajusta ejemplos mal clasificados.",
                "params": {
                    "n_estimators": (10, 300),
                    "learning_rate": (0.01, 2.0),
                },
            },
            "Gradient Boosting": {
                "estimator": GradientBoostingClassifier,
                "help": "Boosting por gradiente; agrega árboles débiles minimizando pérdida.",
                "params": {
                    "n_estimators": (10, 300),
                    "learning_rate": (0.01, 1.0),
                    "max_depth": (1, 10),
                },
            },
            "Random Forest": {
                "estimator": RandomForestClassifier,
                "help": "Bagging de árboles con selección aleatoria de características.",
                "params": {
                    "n_estimators": (50, 500),
                    "max_depth": (1, 30),
                    "max_features": ["sqrt", "log2", None],
                },
            },
        }
    # Regression
    return {
        "Decision Tree": {
            "estimator": DecisionTreeRegressor,
            "help": "Árbol de decisión para regresión (MSE/MAE).",
            "params": {
                "criterion": ["squared_error", "absolute_error"],
                "max_depth": (1, 30),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 20),
            },
        },
        "Bagging": {
            "estimator": BaggingRegressor,
            "help": "Ensamble de bootstrap sobre un estimador base.",
            "params": {
                "n_estimators": (10, 300),
                "max_samples": (0.1, 1.0),
                "max_features": (0.1, 1.0),
            },
        },
        "AdaBoost": {
            "estimator": AdaBoostRegressor,
            "help": "Boosting para regresión.",
            "params": {
                "n_estimators": (10, 300),
                "learning_rate": (0.01, 2.0),
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingRegressor,
            "help": "Boosting por gradiente para regresión.",
            "params": {
                "n_estimators": (10, 300),
                "learning_rate": (0.01, 1.0),
                "max_depth": (1, 10),
            },
        },
        "Random Forest": {
            "estimator": RandomForestRegressor,
            "help": "Bosque aleatorio para regresión.",
            "params": {
                "n_estimators": (50, 500),
                "max_depth": (1, 30),
                "max_features": ["sqrt", "log2", None],
            },
        },
    }


def make_estimator(task: str, name: str, params: Dict) -> object:
    reg = registry(task)[name]
    Est = reg["estimator"]
    common_kwargs = {"random_state": params.get("random_state", 42)}

    if name == "Decision Tree":
        # Map criterion per task
        crit = params.get("criterion")
        if task == "classification":
            crit_kw = {"criterion": crit or "gini"}
        else:
            # Map common UI names to regressor-accepted values
            mapping = {
                "squared_error": "squared_error",
                "absolute_error": "absolute_error",
                "friedman_mse": "friedman_mse",
                "poisson": "poisson",
            }
            crit_kw = {"criterion": mapping.get(crit or "squared_error", "squared_error")}
        return Est(
            **crit_kw,
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            **common_kwargs,
        )
    if name == "Bagging":
        return Est(
            n_estimators=params.get("n_estimators", 100),
            max_samples=params.get("max_samples", 1.0),
            max_features=params.get("max_features", 1.0),
            **common_kwargs,
        )
    if name == "AdaBoost":
        return Est(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 1.0),
            **common_kwargs,
        )
    if name == "Gradient Boosting":
        return Est(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            **common_kwargs,
        )
    if name == "Random Forest":
        extra = {"n_jobs": -1} if task == "classification" else {"n_jobs": -1}
        return Est(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            max_features=params.get("max_features", "sqrt"),
            **common_kwargs,
            **extra,
        )
    raise ValueError("Modelo no soportado")


def plot_confusion_matrix(y_true, y_pred, labels: List[str]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ztext = [[str(v) for v in row] for row in cm]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            colorscale="Blues",
            hovertemplate="Pred: %{x}<br>Real: %{y}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Matriz de confusión",
        xaxis_title="Predicción",
        yaxis_title="Real",
        yaxis=dict(autorange="reversed"),
        height=420,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(x=str(labels[j]), y=str(labels[i]), text=ztext[i][j], showarrow=False, font=dict(color="#000"))
    return fig


def plot_roc_auc(estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    if not hasattr(estimator, "predict_proba"):
        return None, None
    y_prob = estimator.predict_proba(X_test)
    classes = estimator.classes_ if hasattr(estimator, "classes_") else np.unique(y_test)
    n_classes = len(classes)
    fig = go.Figure()
    if n_classes == 2:
        fpr, tpr, _ = metrics.roc_curve(y_test, y_prob[:, 1])
        auc = metrics.auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#888"), name="Aleatorio"))
        fig.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR", height=420)
        return fig, auc
    y_bin = label_binarize(y_test, classes=classes)
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = metrics.roc_curve(y_bin[:, i], y_prob[:, i])
        auc_i = metrics.auc(fpr, tpr)
        aucs.append(auc_i)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Clase {classes[i]} (AUC {auc_i:.2f})"))
    macro_auc = float(np.mean(aucs))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#888"), name="Aleatorio"))
    fig.update_layout(title=f"Curvas ROC OvR (macro AUC {macro_auc:.3f})", xaxis_title="FPR", yaxis_title="TPR", height=480)
    return fig, macro_auc


def regression_diagnostics(y_true: pd.Series, y_pred: np.ndarray):
    df_plot = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred})
    # Pred vs true
    fig1 = px.scatter(df_plot, x="y_true", y="y_pred", trendline="ols")
    fig1.add_trace(go.Scatter(x=[df_plot["y_true"].min(), df_plot["y_true"].max()], y=[df_plot["y_true"].min(), df_plot["y_true"].max()], mode="lines", line=dict(dash="dash", color="#888"), name="y=x"))
    fig1.update_layout(title="Predicción vs Real", xaxis_title="Real", yaxis_title="Predicción", height=420)
    # Residuals vs prediction
    fig2 = px.scatter(df_plot, x="y_pred", y="residual")
    fig2.add_hline(y=0, line_dash="dash", line_color="#888")
    fig2.update_layout(title="Residuos vs Predicción", xaxis_title="Predicción", yaxis_title="Residuo", height=420)
    return fig1, fig2


def feature_importances(estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    try:
        pipe_model = estimator.named_steps.get("model")
        pre = estimator.named_steps.get("preprocess")
        if hasattr(pipe_model, "feature_importances_") and hasattr(pre, "get_feature_names_out"):
            names = pre.get_feature_names_out()
            imps = pipe_model.feature_importances_
            return (
                pd.DataFrame({"feature": names, "importance": imps})
                .sort_values("importance", ascending=False)
                .head(25)
            )
    except Exception:
        pass
    try:
        r = permutation_importance(estimator, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        names = getattr(estimator.named_steps.get("preprocess"), "get_feature_names_out", lambda: np.array(X_test.columns))( )
        return (
            pd.DataFrame({"feature": names, "importance": r.importances_mean})
            .sort_values("importance", ascending=False)
            .head(25)
        )
    except Exception:
        return pd.DataFrame()


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    params: Dict,
    test_size: float = 0.25,
    random_state: int = 42,
    cv_folds: int = 0,
    task: str = "classification",
):
    X, y, pre = build_preprocessor(df, target_col)
    est = make_estimator(task, model_name, {**params, "random_state": random_state})
    pipe = Pipeline([("preprocess", pre), ("model", est)])

    # Split (stratify only for classification)
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if task == "classification":
        metrics_out = {
            "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
            "precision": float(metrics.precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall": float(metrics.recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1": float(metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }
        clf_report = metrics.classification_report(y_test, y_pred, zero_division=0)
        cm_fig = plot_confusion_matrix(y_test, y_pred, labels=np.unique(y))
        roc_fig, auc_val = plot_roc_auc(pipe, X_test, y_test)
        if auc_val is not None:
            metrics_out["auc"] = float(auc_val)
        cv_summary = None
        if cv_folds and cv_folds > 1:
            # Detailed cross-validation with OOF predictions and plots
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            class_labels = np.unique(y)
            y_true_oof: list = []
            y_pred_oof: list = []
            y_prob_oof: list = []
            per_fold_rows: list = []

            supports_proba = hasattr(pipe, "predict_proba")
            for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
                # Fresh preprocessor and estimator per fold
                _, _, pre_f = build_preprocessor(df, target_col)
                est_f = make_estimator(task, model_name, {**params, "random_state": random_state})
                pipe_f = Pipeline([("preprocess", pre_f), ("model", est_f)])

                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                pipe_f.fit(X_tr, y_tr)

                y_pred_f = pipe_f.predict(X_va)
                y_true_oof.append(y_va)
                y_pred_oof.append(pd.Series(y_pred_f, index=y_va.index))

                fold_metrics = {
                    "fold": fold_idx,
                    "accuracy": float(metrics.accuracy_score(y_va, y_pred_f)),
                    "precision": float(metrics.precision_score(y_va, y_pred_f, average="macro", zero_division=0)),
                    "recall": float(metrics.recall_score(y_va, y_pred_f, average="macro", zero_division=0)),
                    "f1": float(metrics.f1_score(y_va, y_pred_f, average="macro", zero_division=0)),
                }

                if hasattr(pipe_f, "predict_proba"):
                    y_prob_f = pipe_f.predict_proba(X_va)
                    # y_prob_f shape: (n, n_classes)
                    y_prob_oof.append(pd.DataFrame(y_prob_f, index=y_va.index, columns=list(class_labels)))
                    try:
                        # Handle binary and multiclass AUC macro
                        if len(class_labels) == 2:
                            auc_f = float(metrics.roc_auc_score(y_va, y_prob_f[:, 1]))
                        else:
                            y_bin = label_binarize(y_va, classes=class_labels)
                            auc_f = float(metrics.roc_auc_score(y_bin, y_prob_f, average="macro", multi_class="ovr"))
                        fold_metrics["auc"] = auc_f
                    except Exception:
                        pass

                per_fold_rows.append(fold_metrics)

            # Concatenate OOF
            y_true_oof = pd.concat(y_true_oof).reindex(index=y.index)
            y_pred_oof = pd.concat(y_pred_oof).reindex(index=y.index)
            y_prob_oof_df = None
            if y_prob_oof:
                y_prob_oof_df = pd.concat(y_prob_oof).reindex(index=y.index)

            per_fold_df = pd.DataFrame(per_fold_rows)
            # Summary mean/std per metric
            metric_cols = [c for c in per_fold_df.columns if c != "fold"]
            summary = {
                m: {"mean": float(per_fold_df[m].mean()), "std": float(per_fold_df[m].std(ddof=1))}
                for m in metric_cols
            }

            # Figures
            # Metrics distribution across folds
            long_df = per_fold_df.melt(id_vars=["fold"], value_vars=metric_cols, var_name="metric", value_name="value")
            metrics_box = px.box(long_df, x="metric", y="value", points="all", color="metric")
            metrics_box.update_layout(title="Distribución de métricas por fold", showlegend=False, height=420)

            # OOF confusion matrix
            cm_oof = confusion_matrix(y_true_oof, y_pred_oof, labels=class_labels)
            ztext = [[str(v) for v in row] for row in cm_oof]
            oof_cm_fig = go.Figure(data=go.Heatmap(z=cm_oof, x=[str(l) for l in class_labels], y=[str(l) for l in class_labels], colorscale="Blues"))
            oof_cm_fig.update_layout(title="Matriz de confusión (OOF)", xaxis_title="Predicción", yaxis_title="Real", yaxis=dict(autorange="reversed"), height=420)
            for i in range(cm_oof.shape[0]):
                for j in range(cm_oof.shape[1]):
                    oof_cm_fig.add_annotation(x=str(class_labels[j]), y=str(class_labels[i]), text=ztext[i][j], showarrow=False, font=dict(color="#000"))

            # OOF ROC curves (if available)
            oof_roc_fig = None
            if y_prob_oof_df is not None:
                oof_roc_fig = go.Figure()
                if len(class_labels) == 2:
                    fpr, tpr, _ = metrics.roc_curve(y_true_oof, y_prob_oof_df.iloc[:, 1])
                    auc_oof = metrics.auc(fpr, tpr)
                    oof_roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc_oof:.3f})"))
                    oof_roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#888"), name="Aleatorio"))
                    oof_roc_fig.update_layout(title="Curva ROC (OOF)", xaxis_title="FPR", yaxis_title="TPR", height=420)
                else:
                    y_bin = label_binarize(y_true_oof, classes=class_labels)
                    aucs = []
                    for i, cls in enumerate(class_labels):
                        fpr, tpr, _ = metrics.roc_curve(y_bin[:, i], y_prob_oof_df.iloc[:, i])
                        auc_i = metrics.auc(fpr, tpr)
                        aucs.append(auc_i)
                        oof_roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Clase {cls} (AUC {auc_i:.2f})"))
                    macro_auc = float(np.mean(aucs))
                    oof_roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#888"), name="Aleatorio"))
                    oof_roc_fig.update_layout(title=f"Curvas ROC OvR (OOF, macro AUC {macro_auc:.3f})", xaxis_title="FPR", yaxis_title="TPR", height=480)

            cv_summary = {
                "method": "StratifiedKFold",
                "folds": int(cv_folds),
                "per_fold": per_fold_df,
                "summary": summary,
                "oof": pd.DataFrame({"y_true": y_true_oof, "y_pred": y_pred_oof}),
                "figs": {
                    "metrics_box": metrics_box,
                    "oof_confusion": oof_cm_fig,
                    "oof_roc": oof_roc_fig,
                },
            }
        imp_df = feature_importances(pipe, X_test, y_test)
        return {
            "pipeline": pipe,
            "metrics": metrics_out,
            "report": clf_report,
            "cm_fig": cm_fig,
            "roc_fig": roc_fig,
            "cv": cv_summary,
            "importances": imp_df,
            "task": task,
        }

    # Regression branch
    mae = float(metrics.mean_absolute_error(y_test, y_pred))
    mse = float(metrics.mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(metrics.r2_score(y_test, y_pred))
    metrics_out = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    report = pd.DataFrame({"metric": list(metrics_out.keys()), "value": list(metrics_out.values())}).to_string(index=False)
    pred_true_fig, residuals_fig = regression_diagnostics(y_test, y_pred)
    cv_summary = None
    if cv_folds and cv_folds > 1:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        y_true_oof_r: list = []
        y_pred_oof_r: list = []
        per_fold_rows_r: list = []

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
            _, _, pre_f = build_preprocessor(df, target_col)
            est_f = make_estimator(task, model_name, {**params, "random_state": random_state})
            pipe_f = Pipeline([("preprocess", pre_f), ("model", est_f)])
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            pipe_f.fit(X_tr, y_tr)
            y_pred_f = pipe_f.predict(X_va)
            y_true_oof_r.append(y_va)
            y_pred_oof_r.append(pd.Series(y_pred_f, index=y_va.index))

            mae_f = float(metrics.mean_absolute_error(y_va, y_pred_f))
            mse_f = float(metrics.mean_squared_error(y_va, y_pred_f))
            rmse_f = float(np.sqrt(mse_f))
            r2_f = float(metrics.r2_score(y_va, y_pred_f))
            per_fold_rows_r.append({"fold": fold_idx, "MAE": mae_f, "MSE": mse_f, "RMSE": rmse_f, "R2": r2_f})

        y_true_oof_r = pd.concat(y_true_oof_r).reindex(index=y.index)
        y_pred_oof_r = pd.concat(y_pred_oof_r).reindex(index=y.index)
        per_fold_df_r = pd.DataFrame(per_fold_rows_r)
        metric_cols_r = [c for c in per_fold_df_r.columns if c != "fold"]
        summary_r = {m: {"mean": float(per_fold_df_r[m].mean()), "std": float(per_fold_df_r[m].std(ddof=1))} for m in metric_cols_r}

        # Figures
        long_df_r = per_fold_df_r.melt(id_vars=["fold"], value_vars=metric_cols_r, var_name="metric", value_name="value")
        metrics_box_r = px.box(long_df_r, x="metric", y="value", points="all", color="metric")
        metrics_box_r.update_layout(title="Distribución de métricas por fold", showlegend=False, height=420)

        # OOF Pred vs True and Residuals
        oof_df = pd.DataFrame({"y_true": y_true_oof_r, "y_pred": y_pred_oof_r})
        oof_pred_true_fig = px.scatter(oof_df, x="y_true", y="y_pred", trendline="ols")
        oof_pred_true_fig.add_trace(go.Scatter(x=[oof_df["y_true"].min(), oof_df["y_true"].max()], y=[oof_df["y_true"].min(), oof_df["y_true"].max()], mode="lines", line=dict(dash="dash", color="#888"), name="y=x"))
        oof_pred_true_fig.update_layout(title="Predicción vs Real (OOF)", xaxis_title="Real", yaxis_title="Predicción", height=420)

        oof_df["residual"] = oof_df["y_true"] - oof_df["y_pred"]
        oof_residuals_fig = px.scatter(oof_df, x="y_pred", y="residual")
        oof_residuals_fig.add_hline(y=0, line_dash="dash", line_color="#888")
        oof_residuals_fig.update_layout(title="Residuos vs Predicción (OOF)", xaxis_title="Predicción", yaxis_title="Residuo", height=420)

        cv_summary = {
            "method": "KFold",
            "folds": int(cv_folds),
            "per_fold": per_fold_df_r,
            "summary": summary_r,
            "oof": oof_df,
            "figs": {
                "metrics_box": metrics_box_r,
                "oof_pred_true": oof_pred_true_fig,
                "oof_residuals": oof_residuals_fig,
            },
        }
    imp_df = feature_importances(pipe, X_test, y_test)
    return {
        "pipeline": pipe,
        "metrics": metrics_out,
        "report": report,
        "pred_true_fig": pred_true_fig,
        "residuals_fig": residuals_fig,
        "cv": cv_summary,
        "importances": imp_df,
        "task": task,
    }


