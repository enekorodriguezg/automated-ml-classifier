import argparse
import sys
import json
import pandas as pd
import pickle
import subprocess
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos ML')
    parser.add_argument('archivo_datos', type=str, help='Ruta al dataset (ej. iris.csv)')
    parser.add_argument('--algo', type=str, choices=['knn', 'tree', 'all'], default='all',
                        help='Algoritmo a ejecutar: knn, tree o all (por defecto)')
    args = parser.parse_args()

    archivo_datos = args.archivo_datos
    archivo_config = "configuration.json"

    print(f"--- 1. Dividiendo datos: 80% Entrenamiento / 20% Test Ciego ---")
    df = pd.read_csv(archivo_datos)

    # Separar características (X) y objetivo (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    # División estratificada para mantener la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Guardar el test ciego para test.py (sin la columna solución)
    archivo_test = "datos_test_20_ciego.csv"
    X_test.to_csv(archivo_test, index=False)

    print("--- 2. Construyendo Pipelines de preprocesamiento ---")
    with open(archivo_config, 'r') as f:
        config = json.load(f)["preprocessing"]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.get('impute_strategy', 'mean'))),
        ('scaler', StandardScaler() if config.get('scaling') == 'standard' else 'passthrough')
    ])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)],
                                     remainder='passthrough')

    print("--- 3. Definiendo algoritmos y espacios de búsqueda ---")
    pipe_knn = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())])
    pipe_tree = Pipeline(
        steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(random_state=42))])

    # En lugar de bucles for, definimos mallas de parámetros
    parametros_knn = {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]
    }

    parametros_tree = {
        'classifier__max_depth': [3, 5, 7, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__criterion': ['gini', 'entropy']
    }

    print("--- 4. Ejecutando entrenamiento intensivo ---")
    mejor_knn_score = -1
    mejor_tree_score = -1
    ganador = None
    nombre_archivo = "mejor_modelo_ganador.sav"

    if args.algo in ['knn', 'all']:
        print("Buscando el mejor kNN...")
        grid_knn = GridSearchCV(pipe_knn, parametros_knn, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_knn.fit(X_train, y_train)
        mejor_knn_score = grid_knn.best_score_
        print(f"Mejor kNN: F-score {mejor_knn_score:.4f}")
        ganador = grid_knn.best_estimator_

    if args.algo in ['tree', 'all']:
        print("Buscando el mejor Árbol de Decisión...")
        grid_tree = GridSearchCV(pipe_tree, parametros_tree, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_tree.fit(X_train, y_train)
        mejor_tree_score = grid_tree.best_score_
        print(f"Mejor Árbol: F-score {mejor_tree_score:.4f}")
        # Si se ejecutan ambos ('all'), gana el mejor. Si solo se ejecuta 'tree', este pisa a ganador.
        if args.algo == 'tree' or mejor_tree_score > mejor_knn_score:
            ganador = grid_tree.best_estimator_

    print("--- 5. Seleccionando el Ganador Absoluto y Guardando ---")
    pickle.dump(ganador, open(nombre_archivo, 'wb'))
    print(f"Modelo físico generado: {nombre_archivo}")

    print("--- 6. Ejecutando Test Ciego ---")
    # Llama a tu test.py inmaculado, pasándole el modelo ganador
    subprocess.run(["python", "test.py", archivo_test, nombre_archivo])


if __name__ == "__main__":
    main()