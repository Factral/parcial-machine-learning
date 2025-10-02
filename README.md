## ML Studio · Árboles y Ensambles

Aplicación Streamlit modular para cargar datos, explorar EDA, entrenar clasificadores (Árbol de Decisión, Bagging, AdaBoost, Gradient Boosting, Random Forest) y evaluar métricas (Matriz de Confusión, Reporte de Clasificación, ROC/AUC). Incluye demostración de criterios de división: Gini, Information Gain y Gain Ratio.

### Estructura

```
.
├── app.py                 # Entrada principal de Streamlit
├── modules/
│   ├── ui.py              # Estilos y encabezados
│   ├── data.py            # Carga de CSV e info de DataFrame
│   ├── plots.py           # Gráficos EDA
│   ├── models.py          # Preprocesamiento, entrenamiento y métricas
│   └── pages.py           # Páginas de Datos/EDA y Modelado
├── data/
│   └── (coloca aquí tus CSV si deseas)
├── requirements.txt
└── README.md
```

### Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

En la app:
- Carga un CSV desde la barra lateral o usa el dataset de demostración.
- Explora EDA (head/describe/info, faltantes, histogramas, barras, correlación).
- Configura el modelo, hiperparámetros y entrena. Visualiza métricas, CM, ROC, CV e importancias.
- Explora los criterios de división (Gini, Gain y Gain Ratio) sobre cualquier variable.

UIS - 2025

