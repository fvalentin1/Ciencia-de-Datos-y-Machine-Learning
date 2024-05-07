import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Cargamos el dataset
datos = load_breast_cancer()
X = pd.DataFrame(datos.data, columns=datos.feature_names)
y = pd.Series(datos.target)

# Dividimos el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizamos los datos
scaler = StandardScaler()
X_train_escalado = scaler.fit_transform(X_train)
X_test_escalado = scaler.transform(X_test)
# print(X_train_escalado)
# print(X_test_escalado)

''' MODELO K NEAREST NEIGHBORS '''

# Creamos el modelo K-NN
k = 5
knn_modelo = KNeighborsClassifier(n_neighbors = k)
knn_modelo.fit(X_train_escalado, y_train)
# print(knn_modelo)

# Realizamos predicciones y evaluamos el modelo
knn_pred = knn_modelo.predict(X_test_escalado) # y_pred = knn_pred
knn_accuracy = accuracy_score(y_test, knn_pred)
print("Accuracy del modelo KNN: ", knn_accuracy)
print("Reporte de clasificación del modelo KNN: ")
print(classification_report(y_test, knn_pred))
print("Matriz de confusión del modelo KNN: ")
print(confusion_matrix(y_test, knn_pred))

print("*"*80)

''' MODELO RANDOM FOREST '''
# Creamos el modelo Random Forest y lo entrenamos
rf_modelo = RandomForestClassifier(n_estimators=100, random_state=42)
rf_modelo.fit(X_train_escalado, y_train)

# Evaluamos el modelo
rf_pred = rf_modelo.predict(X_test_escalado)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Accuracy del modelo Random Forest: ", rf_accuracy)
print("Reporte de clasificación del modelo Random Forest: ")
print(classification_report(y_test, rf_pred))
print("Matriz de confusión del modelo Random Forest: ")
print(confusion_matrix(y_test, rf_pred))

print("*"*80)

''' SELECCIÓN MEJOR MODELO '''
# Condición para seleccionar el mejor modelo
mejor_modelo = knn_modelo if knn_accuracy > rf_accuracy else rf_modelo
print("El mejor modelo es: ", mejor_modelo)

print("*"*80)
    
''' GUARDAR Y CARGAR MODELO '''
# Guardamos el mejor modelo
joblib.dump(mejor_modelo, "mejor_modelo.pkl")
print("Modelo guardado correctamente!")

print("*"*80)

# Cargamos el mejor modelo
modelo_cargado = joblib.load("mejor_modelo.pkl")
print("Modelo cargado correctamente!")

print("*"*80)

# Realizamos predicciones con el modelo cargado
pred = modelo_cargado.predict(X_test_escalado)

# Comparamos las predicciones
comparacion = pd.DataFrame({"Real": y_test, "Predicción": pred})
print("Comparación valores reales y predichos: ")
print(comparacion)
