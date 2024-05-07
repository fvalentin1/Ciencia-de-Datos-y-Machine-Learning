import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Crear un dataset sintético
X, _ = make_moons(n_samples=1000, noise=0.05)

# Instanciar el modelo
dbscan = DBSCAN(eps=0.2, min_samples=5)

# Entrenar el modelo
dbscan.fit(X)

# Obtener las etiquetas de los clusters
labels = dbscan.labels_

# Graficar los clusters
# plt.scatter(X[labels == 0][:, 0], X[labels == 0][:, 1], color='red', label='Cluster 1')
# plt.scatter(X[labels == 1][:, 0], X[labels == 1][:, 1], color='blue', label='Cluster 2')
# plt.scatter(X[labels == -1][:, 0], X[labels == -1][:, 1], color='green', label='Outliers')
# plt.legend()
# plt.show()

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()