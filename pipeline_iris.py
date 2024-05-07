import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# Cargar el dataset
iris = load_iris()
X, y = iris.data , iris.target

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo
accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Guardar el modelo
joblib.dump(pipeline, 'pipeline_iris.pkl')

# Cargar el modelo
modelo_cargado = joblib.load('pipeline_iris.pkl')

# Hacer una predicción
ejemplo = X_test[0].reshape(1, -1)
print(f'Dato de ejemplo usado para predecir: {ejemplo}')

prediccion = modelo_cargado.predict(ejemplo)
print(f'Predicción: {prediccion}')
print(f'Predicción (nombre): {iris.target_names[prediccion]}')
