# Análisis de Ingresos con XGBoost
Este proyecto utiliza el conjunto de datos "Adult" de UCI Machine Learning Repository para predecir si un individuo tiene ingresos superiores a $50,000 al año. Se implementa un modelo XGBoost para este propósito.

# Configuración y Preparación de Datos

Importar bibliotecas necesarias: pandas, sklearn, xgboost.

* Configurar la URL de los datos.

* Definir los nombres de las columnas.

* Leer los datos desde la URL.

* Etiquetar la columna objetivo ("income").

* Convertir variables categóricas a numéricas.

# Análisis y Modelado con XGBoost

Este repositorio contiene un código en Python que realiza un análisis de datos y utiliza el algoritmo XGBoost para la clasificación de ingresos entre personas que ganan mas de 50k o menos o igual a 50k al año. A continuación, se proporciona una descripción detallada del código sin incluir el código fuente.

## Descripción del Código

### Bibliotecas Utilizadas
- **pandas**: Para manipulación y análisis de datos.
- **scikit-learn**: Para herramientas de aprendizaje automático y preprocesamiento de datos.
- **XGBoost**: Implementación de Gradient Boosting para mejorar la eficiencia y el rendimiento.

### Configuración de Datos
Se accede a datos sobre ingresos desde la [URL proporcionada](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data). Los datos se leen en un DataFrame de pandas, utilizando nombres predefinidos para las columnas.

### Preprocesamiento de Datos
- La columna objetivo "income" se etiqueta como 0 si es "<=50K" y 1 en caso contrario.
- Se convierten las variables categóricas en variables numéricas mediante la codificación de etiquetas.

### División de Datos
Los datos se dividen en conjuntos de entrenamiento y prueba en una proporción del 80-20, sin sobre-muestreo.

### Ajuste de Parámetros con GridSearchCV
Se realiza una búsqueda de cuadrícula para ajustar los parámetros del clasificador XGBoost utilizando GridSearchCV. Se exploran diferentes combinaciones de tasas de aprendizaje, número de estimadores, profundidad máxima y peso mínimo por hoja.

### Entrenamiento y Evaluación del Modelo
- El modelo se entrena con los mejores parámetros encontrados.
- Se realizan predicciones en el conjunto de prueba y se evalúa la precisión del modelo junto con otras métricas de clasificación.

### Análisis Adicional
- Se muestra la proporción de clases en los conjuntos de entrenamiento y prueba.
- Se proporciona información y un resumen estadístico del conjunto de datos.
- Se muestran las predicciones del modelo en el conjunto de prueba.

## Ejecución del Código

Para ejecutar este código, se recomienda seguir los siguientes pasos:

# 1. Clonar este repositorio:

    git clone [https://github.com/tu_usuario/tu_repositorio.git](https://github.com/Cha0smagick/Modelo_predictivo_ganar_mas_de50k.git)
    cd Modelo_predictivo_ganar_mas_de50k

# 2. Crear un ambiente virtual (se asume que tienes Python instalado):

    python -m venv venv

# 3. Activar el ambiente virtual:

En Windows:

    .\venv\Scripts\activate

En Linux/Mac:

    source venv/bin/activate
    
# 4. Instalar las dependencias:

    pip install -r requirements.txt

# 5. Ejecutar el script Python:

    python app.py
