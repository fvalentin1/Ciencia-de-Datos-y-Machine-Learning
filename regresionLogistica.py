import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Carga de datos
datos = load_iris()
X = datos.data
y = (datos.target != 0) * 1 # 1 si es clase 1, 0 si es clase 0
#y = datos.target
print(X)
print(y)

# Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creacion del modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Prediccion y evaluacion del modelo
y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#print(y_pred)
