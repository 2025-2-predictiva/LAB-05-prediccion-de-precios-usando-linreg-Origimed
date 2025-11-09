#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


import os
import gzip
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

MODELS_DIR = "files/models"
OUTPUT_DIR = "files/output"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl.gz")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    grading = {
        "x_train": "files/grading/x_train.pkl",
        "y_train": "files/grading/y_train.pkl",
        "x_test":  "files/grading/x_test.pkl",
        "y_test":  "files/grading/y_test.pkl",
    }
    if all(os.path.exists(p) for p in grading.values()):
        with open(grading["x_train"], "rb") as f: x_train = pickle.load(f)
        with open(grading["y_train"], "rb") as f: y_train = pickle.load(f)
        with open(grading["x_test"],  "rb") as f: x_test  = pickle.load(f)
        with open(grading["y_test"],  "rb") as f: y_test  = pickle.load(f)
        return x_train, y_train, x_test, y_test

    train_csv = "files/input/train.csv"
    test_csv = "files/input/test.csv"
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv)
        return train.drop(columns=["Selling_Price"]), train["Selling_Price"].values, \
               test.drop(columns=["Selling_Price"]),  test["Selling_Price"].values

    raise FileNotFoundError(
        "No se encontraron pkl en files/grading/ ni train/test.csv en files/input/."
    )

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year" in df.columns and "Age" not in df.columns:
        df["Age"] = 2021 - df["Year"]
    for col in ["Year", "Car_Name"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def build_gscv(x_train: pd.DataFrame) -> GridSearchCV:
    # ColumnTransformer
    cat_sel = selector(dtype_include=object)
    num_sel = selector(dtype_include=np.number)

    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_sel),
            ("num", MinMaxScaler(), num_sel),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    pipe = Pipeline(steps=[
        ("ct", ct),
        ("select", SelectKBest(score_func=f_regression, k=10)),  # placeholder
        ("reg", LinearRegression()),
    ])

    ct_tmp = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_sel),
            ("num", MinMaxScaler(), num_sel),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    X_ct = ct_tmp.fit_transform(x_train)
    n_features = X_ct.shape[1]

    candidate_k = [5, 10, 15, 20, 30, 50]
    valid_k = sorted({k for k in candidate_k if isinstance(k, int) and k <= n_features})
    if not valid_k:
        valid_k = [min(5, n_features)] if n_features > 0 else [1]

    param_grid = {"select__k": valid_k + ["all"]}

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    return gscv


def main():
    x_train, y_train, x_test, y_test = load_data()

    # Preprocesamiento requerido (fuera del pipeline)
    x_train = preprocess_df(pd.DataFrame(x_train))
    x_test  = preprocess_df(pd.DataFrame(x_test))

    model = build_gscv(x_train)
    model.fit(x_train, y_train)

    with gzip.open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Métricas
    def compute_metrics(name, X, y):
        y_pred = model.predict(X)
        return {
            "type": "metrics",
            "dataset": name,
            "r2": float(r2_score(y, y_pred)),
            "mse": float(mean_squared_error(y, y_pred)),
            "mad": float(mean_absolute_error(y, y_pred)),
        }

    m_train = compute_metrics("train", x_train, y_train)
    m_test  = compute_metrics("test",  x_test,  y_test)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(m_train) + "\n")
        f.write(json.dumps(m_test) + "\n")

    print("Modelo guardado en:", MODEL_PATH)
    print("Métricas guardadas en:", metrics_path)
    print("Mejor params:", model.best_params_)

if __name__ == "__main__":
    main()
