[README.md](https://github.com/user-attachments/files/33004833/README.md)
# Predicción de Precios de Autos Usados con K-Nearest Neighbors (KNN)

Proyecto de portafolio desarrollado como parte de un plan de aprendizaje modular de Machine Learning, basado en el material de la certificación **CAIPC® (Certiprof)**. Aplica el flujo completo de un problema de regresión supervisada: limpieza de datos, ingeniería de características, modelado con KNN, evaluación, optimización de hiperparámetros y validación cruzada.

## Objetivo

Predecir el precio de venta (`selling_price`) de un auto usado a partir de sus características (año, kilometraje, marca, combustible, tipo de vendedor, transmisión y número de dueños anteriores).

## Dataset

**[Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)** (Kaggle, autor: nehalbirla) — archivo `CAR DETAILS FROM CAR DEKHO.csv`.

- 4,340 filas originales, 8 columnas: `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`.
- Sin valores nulos en ninguna columna.
- Licencia del dataset: ver página de Kaggle.

## Metodología

### 1. Limpieza de datos
- **Se eliminaron 763 filas duplicadas (17.6% del dataset)** — un hallazgo de calidad de datos no trivial, detectado al investigar un valor atípico de kilometraje que resultó estar repetido en el dataset.
- Se revisaron los valores atípicos de precio (dos autos de lujo, Mercedes-Benz y Audi, sobre los $8M) y se confirmaron como datos legítimos, no errores — se conservaron en el dataset.

### 2. Ingeniería de características
- `name` (1,491 valores únicos) se redujo a `brand` (marca, primera palabra del nombre) para evitar la maldición de la dimensionalidad. Las marcas con menos de 10 apariciones se agruparon en `"Other"`, quedando 15 categorías de marca.
- `owner` se codificó como variable **ordinal** (`0`–`4`, de "Test Drive Car" a "Fourth & Above Owner") en vez de one-hot, para conservar su orden natural.
- `brand`, `fuel`, `seller_type` y `transmission` se codificaron con one-hot encoding (`drop_first=True`).
- Dataset final: 3,577 filas × 27 features (tras eliminar duplicados).

### 3. Modelado
- Split holdout 80/20 (`random_state=42`).
- Estandarización (`StandardScaler`) ajustada **solo** sobre el set de entrenamiento, para evitar fuga de datos hacia el test.
- Modelo: `KNeighborsRegressor` (scikit-learn).
- Optimización de `k` vía `GridSearchCV` (rango 1–20) con validación cruzada K-Fold (5 folds).
- Exploración adicional: transformación logarítmica (`log1p`) de la variable objetivo, dado su fuerte sesgo a la derecha.

## Resultados

| Métrica | Baseline (k=5) | Optimizado (k=4) | Log-transform (k=6) |
|---|---|---|---|
| MAE | 171,232 | 169,516 | **166,211** |
| RMSE | 422,842 | 423,831 | **417,416** |
| R² | 0.445 | 0.442 | **0.459** |

El modelo final (log-transform, k=6) explica cerca del 46% de la variación en el precio de venta, con un error absoluto promedio de ~166,000 (≈ 35% del precio promedio del dataset, $473,913).

## Hallazgos clave

- **La optimización de `k` por sí sola tuvo un efecto casi nulo** sobre el set de prueba (RMSE incluso empeoró levemente de k=5 a k=4). La curva de validación cruzada era prácticamente plana entre k=4 y k=10, así que la diferencia entre esos valores es ruido estadístico más que una mejora real.
- **La transformación logarítmica sí produjo una mejora consistente** en las tres métricas, al comprimir la cola de precios altos.
- **El sesgo sistemático en autos de lujo no se resolvió** con ninguna de las dos técnicas: el modelo subestima consistentemente los autos más caros del dataset. En un caso extremo, dos autos del set de prueba con precios reales de $3.8M y $8.15M recibieron **exactamente la misma predicción** — evidencia directa de que las columnas disponibles no alcanzan para distinguir estos casos.

## Limitaciones

- **Representación insuficiente de autos de lujo:** solo un puñado de autos en el dataset superan los $3M, por lo que KNN (que predice promediando vecinos) no tiene suficiente "contexto" cercano para predecir bien en ese rango.
- **Features limitadas:** el dataset no incluye marca/modelo detallado más allá de la marca general, ni condición, color, o versión del vehículo — variables que probablemente explican buena parte del precio de los autos premium.
- **KNN es sensible a la dimensionalidad:** con 27 features tras el one-hot encoding, el modelo ya empieza a acercarse al límite donde las distancias pierden poder discriminativo.

## Próximos pasos

- Probar un modelo menos sensible a la dispersión en zonas de baja densidad de datos (ej. Random Forest, Gradient Boosting).
- Incorporar una columna de modelo/versión del auto como feature adicional (más allá de la marca).
- Evaluar si excluir o tratar por separado los autos de lujo (modelado por segmentos) mejora el resultado general.

## Estructura del repositorio

```
knn-car-price-prediction/
├── README.md
├── data/
│   └── raw/
│       └── CAR DETAILS FROM CAR DEKHO.csv
├── notebooks/
│   └── 01_exploracion_y_limpieza.ipynb
└── requirements.txt
```

## Cómo ejecutar el proyecto

1. Clona el repositorio o ábrelo directo en [Google Colab](https://colab.research.google.com/) desde GitHub.
2. Instala las dependencias: `pip install -r requirements.txt`
3. Corre el notebook `notebooks/01_exploracion_y_limpieza.ipynb` de principio a fin.

## Tecnologías

Python · pandas · scikit-learn · matplotlib · seaborn · Google Colab
