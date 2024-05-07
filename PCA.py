import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Cargar el dataset
datos = load_iris()
X = datos.data
y = datos.target

# Instanciar el modelo PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X) # Ajustar y transformar los datos
print(X)
print("*"*50)
print(X_pca)

# Graficar los datos
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', cmap='viridis')
# plt.xlabel('Componente 1')
# plt.ylabel('Componente 2')
# plt.title('PCA')
# plt.colorbar()
# plt.show()
