# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configurar la URL de los datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Definir los nombres de las columnas
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Leer los datos desde la URL
data = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Poner etiquetas a la columna objetivo (income)
data["income"] = data["income"].apply(lambda x: 0 if x == "<=50K" else 1)

# Separar los datos en características (X) y etiquetas (y)
X = data.drop("income", axis=1)
y = data["income"]

# Convertir variables categóricas a numéricas
le = LabelEncoder()
categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
for column in categorical_columns:
    data[column] = le.fit_transform(data[column])

# Separar los datos en características (X) y etiquetas (y)
X = data.drop("income", axis=1)
y = data["income"]

# Dividir los datos en conjuntos de entrenamiento y prueba sin sobre-muestreo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ajustar parámetros de XGBoost utilizando GridSearchCV
param_grid = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros:", grid_search.best_params_)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros:", grid_search.best_params_)

# Crear y entrenar el modelo con los mejores parámetros
best_model = XGBClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Calcular la precisión del modelo y otras métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))

# Verificar proporción de clases en conjunto de entrenamiento y prueba
print("Proporción de clases en conjunto de entrenamiento:")
print(y_train.value_counts(normalize=True))

print("\nProporción de clases en conjunto de prueba:")
print(y_test.value_counts(normalize=True))

# Analizar el conjunto de datos
print("Información del conjunto de datos:")
print(data.info())

print("\nResumen estadístico del conjunto de datos:")
print(data.describe())

# Revisar las predicciones del modelo en el conjunto de prueba
print("Predicciones del modelo en el conjunto de prueba:")
print(pd.DataFrame({"Real": y_test, "Predicción": y_pred}))
