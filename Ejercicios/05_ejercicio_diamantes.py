# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

# Cargar el dataset de diamantes
datos = sns.load_dataset("diamonds")
df = pd.DataFrame(datos)
print(df.head())

# Mostrar información del dataset
print('*'*80)
print(df.info())

# Mostrar estadísticas del dataset
print('*'*80)
print(df.describe())

# Mostrar la cantidad de nulos
print('*'*80)
print(df.isnull().sum())

# Borrar los nulos
#df_limpio = df.dropna()

# Graficar
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['carat'], kde=True, color='blue')
plt.title('Distribución de los quilates')
plt.xlabel('Quilates')
plt.ylabel('Frecuencia')
plt.subplot(1, 2, 2)
sns.histplot(df['price'], kde=True, color='green')
plt.title('Distribución de los precios')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# Gráfico de relación entre quilates y precio
plt.figure(figsize=(10, 5))
sns.scatterplot(x='carat', y='price', data=df, color='red', alpha=0.5)
plt.title('Relación entre quilates y precio')
plt.xlabel('Quilates [ct]')
plt.ylabel('Precio [USD]')
plt.show()

# Boxplot de relación precio por claridad
plt.figure(figsize=(10, 5))
sns.boxplot(x='clarity', y='price', data=df, palette='viridis', order=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
plt.title('Relación precio por claridad')
plt.xlabel('Claridad')
plt.ylabel('Precio [USD]')
plt.show()

