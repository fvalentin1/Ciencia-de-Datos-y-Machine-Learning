import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Cargamos el dataset
iris = load_iris()
X = iris.data
y = iris.target

# Dividimos el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el modelo K-NN
k = 3
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, y_train)

# Realizamos predicciones
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Ajustar el valor de K para obtener el mejor rendimiento
# for k in range(1, 20):
#     knn = KNeighborsClassifier(n_neighbors = k)
#     knn.fit(X_train, y_train)
#     print(f"K = {k}, Accuracy = {knn.score(X_test, y_test)}")

# Visualizamos los resultados
for i, class_name in enumerate(iris.target_names):
    plt.scatter(X_test[y_pred == i, 0], X_test[y_pred == i, 1], label=class_name)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Clasificación K-NN')
plt.legend()  # Añadir leyenda
plt.show()