# Clasificación Automatizada (Sistemas de Apoyo a la Decisión)

Este repositorio contiene un sistema completo y defensivo de Machine Learning (Entrenamiento e Inferencia) capaz de procesar datos crudos, realizar limpieza dinámica, balancear clases y ejecutar una búsqueda de hiperparámetros (GridSearchCV) sobre múltiples algoritmos de clasificación.

## 🛠️ Requisitos e Instalación

* **Python:** Versión 3.12.3
* **Dependencias:** No se requiere ningún archivo extra. Para instalar todas las librerías necesarias, abre la terminal en la raíz del proyecto y ejecuta este único comando:

    pip install -r requirements.txt

## 📁 Estructura del Proyecto

* **`train.py`**: Script principal de la fase de entrenamiento. Ingiere los datos crudos, aísla dinámicamente la variable objetivo, preprocesa (imputación, escalado, One-Hot Encoding), balancea las clases de forma segura para evitar filtraciones y entrena los modelos especificados. Exporta el modelo ganador en formato `.sav` y genera el diccionario de clases (`label_encoder.sav`).
* **`test.py`**: Script de la fase de inferencia. Carga un modelo `.sav` pre-entrenado, el diccionario de clases y un dataset ciego para generar predicciones finales. Incluye mecanismos de defensa contra formatos de entrada incorrectos, fugas de dimensionalidad y restos de la variable objetivo.
* **`config.json`**: Archivo de control. Permite modificar las estrategias de imputación, escalado y balanceo sin necesidad de alterar el código fuente.

## 🚀 Instrucciones de Ejecución

El flujo de trabajo se divide en dos fases obligatorias y secuenciales:

### Fase 1: Entrenamiento (`train.py`)

Para entrenar un modelo, se debe ejecutar el script indicando el dataset de entrenamiento, la columna a predecir, el archivo de configuración y, opcionalmente, el algoritmo deseado.

**Sintaxis básica:**

    python train.py <archivo_datos.csv> -p <columna_objetivo> -c <archivo_config.json> [--algo <algoritmo>]

**Opciones principales:**
* `-p` / `--pred`: (Obligatorio) Nombre exacto de la columna que el sistema debe aprender a predecir.
* `-c` / `--config`: (Obligatorio) Ruta al archivo de configuración JSON.
* `--algo`: (Opcional) Permite aislar el entrenamiento (`knn`, `tree`, `nb`, `rf`). Por defecto, ejecuta todos (`all`) compitiendo entre sí.

### Fase 2: Clasificación de Nuevos Ítems (`test.py`)

Una vez generados el modelo físico (`.sav`) y el `label_encoder.sav` en la Fase 1, se utiliza este script unificado para predecir sobre un nuevo conjunto de datos.

**Sintaxis básica:**

    python test.py <datos_nuevos_ciegos.csv> -m <modelo_guardado.sav> [-p <columna_objetivo>]

**Opciones principales:**
* `-p` / `--pred`: (Opcional) Si tu dataset de prueba incluye la columna objetivo pegada, indícala aquí para que el sistema la ignore de forma segura antes de predecir.

## 🛡️ Arquitectura Defensiva y Notas para la Evaluación

El sistema ha sido reestructurado asumiendo que los datos de evaluación pueden presentar inconsistencias. Se han implementado las siguientes capas de seguridad de grado producción:

1. **Aislamiento Dinámico del Objetivo (Target):** El sistema ya no asume que la variable a predecir está en la última columna. Requiere su declaración explícita por terminal (`-p`), lo que previene la amputación accidental de variables predictoras legítimas.
2. **Extracción Dinámica de Identificadores:** El sistema escanea el dataset buscando variaciones de variables identificadoras. Si las detecta, las aísla temporalmente para evitar que el algoritmo las memorice, restaurándolas al final.
3. **Persistencia Categórica (LabelEncoder):** La variable objetivo no se traduce destructivamente. El mapeo de clases se serializa en `label_encoder.sav` durante el entrenamiento.
4. **Prevención de Feature Mismatch (Data Leakage):** Durante la inferencia, el script escanea el dataset de entrada contra la firma matemática del modelo entrenado extirpando columnas no deseadas.
5. **Aislamiento del Balanceo (ImbPipeline):** Se ha utilizado la tubería de la librería `imbalanced-learn` para asegurar que el balanceo de clases sintético se aplique **exclusivamente** sobre los pliegues de entrenamiento durante la validación cruzada.

## ⚖️ Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
