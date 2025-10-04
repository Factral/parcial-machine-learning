from __future__ import annotations

import textwrap
from typing import Any, Dict
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .data import load_csv, infer_column_types, dataframe_info, missing_summary, head_tail_describe
from .plots import plot_numeric_histograms, plot_categorical_bars, plot_correlation_heatmap, association_matrix
from .models import registry, train_and_evaluate

# FUNCIÓN PARA ESTABLECER QUE COLUMNAS SON VÁLIDAS COMO VARIABLES OBJETIVO
def _valid_targets_for_task(df: pd.DataFrame, task_key: str) -> list[str]:
    cols: list[str] = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        if s.dropna().empty:
            continue
        if task_key == "classification":
            nun = int(s.nunique(dropna=True))
            if nun < 2:
                continue                     #FILA CON VALORES UNICOS MENOORES A 2 SE DESCARTAN
            vc = s.astype(str).value_counts(dropna=True)
            if vc.min() < 2:
                continue                    #CATEGORIAS CON VARIABLES UNICAS CON UNA FRECUENCIA MENOR A 2 SE DESCARTAN
            if (not pd.api.types.is_numeric_dtype(s)) or nun <= 20 or nun <= max(5, int(0.05 * n)):
                cols.append(c)               # COLUMNAS ACEPTADAS COMO CATEGORICAS
        else:
            if pd.api.types.is_numeric_dtype(s) and int(s.nunique(dropna=True)) >= 5:
                cols.append(c)                # COLUMNAS ACEPTADAS COMO NUMERICAS
    return cols


def _max_cv_folds_for_target(df: pd.DataFrame, target_col: str, task_key: str) -> int:
    if task_key != "classification":
        return 10                            
    vc = df[target_col].astype(str).value_counts(dropna=True)         #CONTEO DE VARIABLES CATEGORICAS OBJETIVO
    return int(vc.min()) if not vc.empty else 0                        #FOLDS DEL CV NO PUEDE SER MAYOR A LA CANTIDAD DE VARIABLES DE UNA CATEGORÍA (DEBE EXISTIR AL MENOS UNA VARIBALE EN CADA FOLD)


def _load_demo_dataset() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df.rename(columns={"target": "target"}, inplace=True)
    return df

#SI NO HAY DATA SET MOSTRAR MENSAJE DE CARGAR DATA SET
def _get_df() -> pd.DataFrame:
    df = st.session_state.get("df", pd.DataFrame())
    if df.empty:
        st.info("Carga un CSV o usa el dataset de demostración en la barra lateral.")
    return df

#MANTENER EL DATASET CARGADO DURANTE LA EJECUCIÓN DE LA APLICACIÓN PARA NO TENER QUE ESTAR CARGANDO ESTE EN CADA MOMENTO
def _handle_file_upload():
    """Callback for file upload to trigger immediate update"""
    if st.session_state.uploaded_file is not None:
        st.session_state.df = load_csv(st.session_state.uploaded_file)
        st.session_state.dataset_name = getattr(st.session_state.uploaded_file, "name", "dataset.csv")
        st.session_state.file_processed = True

#MISMO ANTERIOR CON LOS DATA DEMO
def _handle_demo_load():
    """Callback for demo dataset load"""
    st.session_state.df = _load_demo_dataset()
    st.session_state.dataset_name = "breast_cancer_demo.csv"
    st.session_state.file_processed = True

#BLOQUES DE CARGA DEL DATA Y DATA DEMO (3 COLUMNAS)
def page_data_eda() -> None:
    # Initialize session state
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False

    # Upload area with clean layout
    upload_container = st.container()
    with upload_container:
        col1, col2, col3 = st.columns([3, 0.5, 2])
        
        with col1:
            # File uploader with callback
            uploaded = st.file_uploader(
                "Selecciona tu archivo CSV",
                type=["csv"],
                key="uploaded_file",
                on_change=_handle_file_upload,
                label_visibility="collapsed"
            )
        
        with col2:
            # Spacer with "or" divider
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 60px; color: #9CA3AF; font-weight: 500;">
                    o
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            # Demo button with better positioning
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)  # Small spacer
            demo_btn = st.button(
                "🧬 Usar dataset demo",
                on_click=_handle_demo_load,
                help="Cargar dataset de cáncer de mama para pruebas"
            )

    df = _get_df()
    if df.empty:
        return  

    # add a line of separation
    st.markdown("---")

    num_cols, cat_cols = infer_column_types(df)
    h, t, desc = head_tail_describe(df)    #PRIMERAS (t) Y ULTIMOS 10 FILAS (h) E INFORMACION DESCRIPTIVA (DESC)

    # Dashboard header
    ds_name = st.session_state.get("dataset_name", "dataset.csv")
    st.markdown(f"### 📁 {ds_name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")
    c3.metric("Numéricas", f"{len(num_cols)}")
    c4.metric("Categóricas", f"{len(cat_cols)}")

    # Processed DataFrame info table
    info_df = pd.DataFrame({
        "columna": df.columns,
        "tipo": [str(dt) for dt in df.dtypes],
        "faltantes": df.isna().sum().values,
        "% faltantes": (df.isna().mean().values * 100).round(2),
        "únicos": [df[c].nunique(dropna=True) for c in df.columns],
    })
    info_df = info_df.sort_values("% faltantes", ascending=False).reset_index(drop=True)
    
    st.markdown("#### Vista rápida y esquema")
    st.dataframe(h, width='stretch')
    with st.expander("Esquema de columnas"):
        st.dataframe(info_df, width='stretch')   #INFORMACIÓN DE CADA CATEGORÍA DEL DATA SET 
    with st.expander("Estadísticos descriptivos"):
        st.dataframe(desc, width='stretch')
    with st.expander("Valores faltantes (detalles)"):
        st.dataframe(missing_summary(df), width='stretch')

    st.markdown("#### Visualizaciones")
    tabs = st.tabs(["Histogramas", "Barras", "Correlación"]) 
    with tabs[0]:
        if num_cols:
            col_sel = st.multiselect("Variables numéricas", num_cols, default=num_cols[: min(5, len(num_cols))])   #BARRA DE SELECCIÓN DE VARIABLES A GRAFICAR
            for col in col_sel:
                fig = px.histogram(df, x=col, nbins=30, opacity=0.9, color_discrete_sequence=["#4F46E5"])
                fig.update_layout(height=320, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay variables numéricas para graficar.")
    with tabs[1]:
        if cat_cols:
            col_sel = st.multiselect("Variables categóricas", cat_cols, default=cat_cols[: min(5, len(cat_cols))])    #BARRA DE SELECCIÓN DE VARIABLES A GRAFICAR
            # Determine max categories dynamically based on current selection
            if col_sel:
                max_cats = max(int(df[c].astype(str).nunique(dropna=True)) for c in col_sel)
            else:
                max_cats = max(int(df[c].astype(str).nunique(dropna=True)) for c in cat_cols) if cat_cols else 3
            max_cats = max(1, max_cats)
            default_top = min(12, max_cats)
            top_n = st.slider("Top categorías", 1, int(max_cats), int(default_top))     #CATEGORIAS A MOSTRAR, 12 O MAX DE CATEGORIAS DE LOS DATA SET
            for col in col_sel:
                vc = df[col].astype(str).value_counts().nlargest(top_n).reset_index()
                vc.columns = [col, "Frecuencia"]
                fig = px.bar(vc, x=col, y="Frecuencia", color="Frecuencia", color_continuous_scale="Viridis")
                fig.update_layout(height=320, template="plotly_white", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay variables categóricas para graficar.")
    with tabs[2]:
        st.caption("Asociaciones entre todas las columnas: Pearson (num-num), Cramér's V (cat-cat), Ratio de correlación (num-cat).")
        assoc = association_matrix(df)
        fig = px.imshow(assoc, zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto")
        fig.update_layout(height=650, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

#DEL MODELO SELECCIONADO Y DE LOS PARAMETROS DEFINIDOS DE ESTE (REGISTRY) SE CREAN LAS OPCIONES INTERACTIVAS PARA MODIFICARLOS
def _render_params_controls(name: str, task: str) -> Dict[str, Any]:
    cfg = registry(task)[name]
    st.caption(cfg["help"])
    params = {}
    with st.container():
        cols = st.columns(3)
        for i, (p, rng) in enumerate(cfg["params"].items()):
            with cols[i % 3]:
                if isinstance(rng, list):
                    params[p] = st.selectbox(p, rng)
                elif isinstance(rng, tuple) and all(isinstance(v, int) for v in rng):
                    params[p] = st.slider(p, int(rng[0]), int(rng[1]))
                elif isinstance(rng, tuple):
                    params[p] = st.slider(p, float(rng[0]), float(rng[1]))
                else:
                    st.write(p, ":", rng)
    return params


def page_modeling() -> None:
    df = _get_df()
    if df.empty:
        return
                                                        
    st.markdown("### Configuración de modelo")                    #RANGOS Y VALORES DE LAS MULTIPLES OPCIONES LISTADAS
    cols = st.columns([1, 1, 1, 1, 1])
    with cols[0]:
        task = st.selectbox("Tarea", ["Clasificación", "Regresión"], index=0)
        task_key = "classification" if task == "Clasificación" else "regression"
    with cols[1]:
        target_options = _valid_targets_for_task(df, task_key)
        if not target_options:
            st.error("No hay columnas válidas para la tarea seleccionada.")
            return
        target_col = st.selectbox("Columna objetivo", target_options)
    with cols[2]:
        test_size = st.slider("Proporción de test", 0.1, 0.5, 0.25)
    with cols[3]:
        max_cv = _max_cv_folds_for_target(df, target_col, task_key)
        cv_folds = st.slider("CV (0 sin CV)", 0, max(0, max_cv), 0)
    with cols[4]:
        rnd = st.number_input("Random state", 0, 10_000, 42)
    st.markdown("---")

    # Algorithms registry depends on task
    algos = list(registry(task_key).keys())
    model_name = st.selectbox("Algoritmo", algos)
    params = _render_params_controls(model_name, task_key)
    run = st.button("Entrenar modelo")

    if not run:
        out = st.session_state.get("training_results")
        if out is None:
            st.info("Configura los hiperparámetros y presiona 'Entrenar modelo'.")
            return
    else:
        out = train_and_evaluate(
            df,
            target_col,
            model_name,
            params,
            test_size=test_size,
            random_state=int(rnd),
            cv_folds=cv_folds,
            task=task_key,
        )
    
    # Persist trained pipeline and feature metadata for inference page
    X_cols = [c for c in df.columns if c != target_col]
    X_only = df[X_cols]
    num_feats = X_only.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in X_cols if c not in num_feats]
    cat_values = {c: sorted(pd.Series(X_only[c].astype(str)).dropna().unique().tolist()) for c in cat_feats}
    defaults = {**{c: float(X_only[c].median()) if pd.api.types.is_numeric_dtype(X_only[c]) else None for c in X_cols}}
    for c in cat_feats:
        if defaults.get(c) in (None, float("nan")):
            defaults[c] = (cat_values[c][0] if cat_values.get(c) else "")

    st.session_state.trained_pipeline = out.get("pipeline")
    st.session_state.trained_task = task_key
    st.session_state.trained_target = target_col
    st.session_state.trained_features = X_cols
    st.session_state.trained_num_features = num_feats
    st.session_state.trained_cat_features = cat_feats
    st.session_state.trained_cat_values = cat_values
    st.session_state.trained_feature_defaults = defaults
    st.session_state.training_results = out

    if run:
        st.rerun()
        return

    # Download trained model button (persists across reruns)
    try:
        model_bytes = pickle.dumps(out.get("pipeline"))
        file_name = f"model_{model_name.replace(' ', '_').lower()}_{task_key}.pkl"
        dl_cols = st.columns([1, 0.3])
        with dl_cols[1]:
            st.download_button(
                "⬇️ Descargar modelo (.pkl)",
                data=model_bytes,
                file_name=file_name,
                mime="application/octet-stream",
                key="download_model_btn",
            )
    except Exception:
        pass

    # Show metrics and plots depending on task
    if out.get("task") == "classification":
        m = out["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("Precision (macro)", f"{m['precision']:.3f}")
        c3.metric("Recall (macro)", f"{m['recall']:.3f}")
        c4.metric("F1 (macro)", f"{m['f1']:.3f}")
        c5.metric("AUC" if 'auc' in m else "AUC", f"{m.get('auc', float('nan')):.3f}")

        st.markdown("#### Reporte de clasificación")
        st.code(out["report"])

        cols = st.columns(2)
        with cols[0]:
            st.markdown("##### Matriz de confusión")
            st.plotly_chart(out["cm_fig"], use_container_width=True)
        with cols[1]:
            st.markdown("##### Curva ROC")
            if out["roc_fig"] is not None:
                st.plotly_chart(out["roc_fig"], use_container_width=True)
            else:
                st.caption("El estimador no soporta probabilidades.")

        if out["cv"] is not None:
            st.markdown("#### Validación Cruzada")
            cv = out["cv"]
            st.caption(f"Método: {cv.get('method', '')} · folds: {cv.get('folds', '')}")
            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                figs = cv.get("figs", {})
                if figs.get("metrics_box") is not None:
                    st.plotly_chart(figs["metrics_box"], use_container_width=True, key="cv_clf_metrics_box")
                if figs.get("oof_confusion") is not None:
                    st.plotly_chart(figs["oof_confusion"], use_container_width=True, key="cv_clf_oof_cm")
                if figs.get("oof_roc") is not None:
                    st.plotly_chart(figs["oof_roc"], use_container_width=True, key="cv_clf_oof_roc")
                if figs.get("oof_pred_true") is not None:
                    st.plotly_chart(figs["oof_pred_true"], use_container_width=True, key="cv_reg_oof_pred_true")
                if figs.get("oof_residuals") is not None:
                    st.plotly_chart(figs["oof_residuals"], use_container_width=True, key="cv_reg_oof_residuals")
            with c_right:
                # Summary table
                summary = cv.get("summary", {})
                if isinstance(summary, dict) and summary:
                    summary_df = (
                        pd.DataFrame(summary).T
                        .rename(columns={"mean": "media", "std": "desv"})
                        .reset_index().rename(columns={"index": "métrica"})
                    )
                    st.dataframe(summary_df, use_container_width=True)
                # Per-fold details
                if isinstance(cv.get("per_fold"), pd.DataFrame):
                    with st.expander("Detalles por fold"):
                        st.dataframe(cv["per_fold"], use_container_width=True)
    else:
        m = out["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{m['MAE']:.3f}")
        c2.metric("MSE", f"{m['MSE']:.3f}")
        c3.metric("RMSE", f"{m['RMSE']:.3f}")
        c4.metric("R2", f"{m['R2']:.3f}")

        st.markdown("#### Diagnósticos")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("##### Predicción vs Real")
            st.plotly_chart(out["pred_true_fig"], use_container_width=True, key="reg_pred_true")
        with cols[1]:
            st.markdown("##### Residuos")
            st.plotly_chart(out["residuals_fig"], use_container_width=True, key="reg_residuals")

    imp = out["importances"]
    if not imp.empty:
        st.markdown("#### Importancia de características")
        top_imp = imp.iloc[:20][::-1].copy()
        fig_imp = px.bar(top_imp, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Viridis")
        fig_imp.update_layout(height=max(360, 22 * len(top_imp)), template="plotly_white", coloraxis_showscale=False, xaxis_title="Importancia", yaxis_title="")
        st.plotly_chart(fig_imp, use_container_width=True, key="feature_importances")


#PESTAÑA INFERENCIA
def page_inference() -> None:
    st.markdown("### 🧪 Inferencia")
    pipe = st.session_state.get("trained_pipeline")
    if pipe is None:
        st.info("Entrena un modelo primero en la pestaña de Modelado.")
        return

    features = st.session_state.get("trained_features", [])
    if not features:
        st.warning("No hay metadatos de características disponibles.")
        return

    num_feats = set(st.session_state.get("trained_num_features", []))
    cat_feats = set(st.session_state.get("trained_cat_features", []))
    cat_values = st.session_state.get("trained_cat_values", {})
    defaults = st.session_state.get("trained_feature_defaults", {})

    st.caption("Ingresa los valores para un nuevo punto de datos:")
    with st.form("inference_form"):
        cols = st.columns(3)
        inputs: Dict[str, Any] = {}
        for i, c in enumerate(features):
            with cols[i % 3]:
                if c in num_feats:
                    default_val = defaults.get(c)
                    try:
                        default_val = float(default_val) if default_val is not None else 0.0
                    except Exception:
                        default_val = 0.0
                    inputs[c] = st.number_input(c, value=default_val)
                else:
                    opts = cat_values.get(c, [])
                    if opts:
                        opts_with_other = list(opts) + ["Otro…"]
                        sel = st.selectbox(c, opts_with_other, index=0, key=f"inf_sel_{c}")
                        if sel == "Otro…":
                            custom = st.text_input(f"{c} (otro)", value="", key=f"inf_txt_{c}")
                            inputs[c] = custom.strip()
                        else:
                            inputs[c] = sel
                    else:
                        st.caption(f"Sin valores observados para '{c}'. Ingresa un valor:")
                        custom = st.text_input(f"{c}", value="", key=f"inf_txt_{c}")
                        inputs[c] = custom.strip()
        submitted = st.form_submit_button("Predecir")

    if not submitted:
        return

    x_new = pd.DataFrame([inputs])
    try:
        y_pred = pipe.predict(x_new)[0]
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        return

    task = st.session_state.get("trained_task", "classification")
    if task == "classification":
        st.success(f"Predicción: {y_pred}")
        prob_fig = None
        try:          
            if hasattr(pipe, "predict_proba"):                   # MOSTRAR GRAFICOS DE PROBABILIDADES OBTENIDAS EN CADA CLASE O VARIABLE
                prob = pipe.predict_proba(x_new)[0]         
                classes = list(getattr(pipe, "classes_", [str(i) for i in range(len(prob))]))
                prob_df = pd.DataFrame({"clase": classes, "prob": prob})
                prob_fig = px.bar(prob_df, x="clase", y="prob", color="prob", color_continuous_scale="Viridis")
                prob_fig.update_layout(template="plotly_white", height=320, coloraxis_showscale=False)
        except Exception:
            pass
        if prob_fig is not None:
            st.plotly_chart(prob_fig, use_container_width=True, key="infer_proba")
    else:
        try:
            y_pred_f = float(y_pred)
            st.success(f"Predicción: {y_pred_f:.4f}")
        except Exception:
            st.success(f"Predicción: {y_pred}")

