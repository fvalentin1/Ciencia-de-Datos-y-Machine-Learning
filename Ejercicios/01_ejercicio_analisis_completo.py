import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datos = {
    'Fecha': pd.date_range(start='2023-01-01', end='2023-01-10'),
    'Producto' : ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    'Ventas': [100, 46, 87, 86, 98, 79, 110, 211, 312, 415]
}

df = pd.DataFrame(datos)

# Guardamos datos en un archivo csv
df.to_csv('ventas.csv', index=False)

# Cargamos los datos del archivo csv
df = pd.read_csv('ventas.csv')

# Mostramos los datos
print(df.head(10))

# Mostramos la información de los datos
print(df.info())

# Mostramos la descripción de los datos
print(df.describe())

# Número de valores únicos de la columna Producto
print(df['Producto'].value_counts())

# Gráfico de barras de las ventas por producto
df.groupby('Producto')['Ventas'].sum().plot(kind='bar')
#plt.bar(df['Producto'], df['Ventas'])
plt.xlabel('Producto')
plt.ylabel('Ventas')
plt.title('Ventas por Producto')
plt.show()

# Gráfico de líneas, tendencia de ventas por tiempo
#df.groupby('Fecha')['Ventas'].sum().plot(kind='line', marker='o')
plt.plot(df['Fecha'], df['Ventas'], marker='o')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.title('Tendencia de Ventas')
plt.xticks(rotation=45)
plt.show()
