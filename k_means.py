import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar el dataset de iris
datos = load_iris()
X = pd.DataFrame(datos.data, columns=datos.feature_names)
print(X.info())
print("*"*50)
print(X.describe())
print("*"*50)
print(X.head())

# Escalar los datos
scaler = StandardScaler()
x_escalado = scaler.fit_transform(X)

# Crear y entrenar el modelo
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x_escalado)

# Obtener las etiquetas de los clusters y asignarlas a los datos
etiquetas = kmeans.labels_
X['cluster'] = etiquetas
print(X.head(25))

# Visualizar los clusters
plt.scatter(X['sepal length (cm)'], X['sepal width (cm)'], c=X['cluster'], cmap='viridis')
plt.xlabel('Longitud sepalo (cm)')
plt.ylabel('Ancho sepalo (cm)')
plt.title('Clasificación por Clusters de iris con k-Means')
plt.show()
