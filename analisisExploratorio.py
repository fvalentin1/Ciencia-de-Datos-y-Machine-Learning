import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Carga de datos
datos = pd.read_csv('precios_coches02.csv', sep=';')
datos.set_index('Unnamed: 0', inplace=True)
datos.index.name = "Indice"

print(datos.head())

print("*"*80)
# Traducir las columnas al español
print(datos.columns)
titulos_cabecera = ["Nombre", "Localizacion", "Anho", "Kilometros recorridos", "Combustible", "Transmision", "Tipo de Propietario", "Autonomia", "Motor", "Potencia", "Asientos", "Nuevo precio", "Precio"]
datos.columns = titulos_cabecera
print(datos.head())

print("*"*80)
# Mostrar las estadísticas básicas de las variables numéricas
print(datos.describe())
print("- "*30)
print(datos.describe(include='all'))

print("*"*80)
# Reemplazar nulos por la media
media = datos['Asientos'].mean()
datos['Asientos'].replace(np.nan, media, inplace=True)
print(datos.describe())

print("*"*80)
# Estadisticas de Motor y Potencia
print(datos['Motor'])
print("- "*30)
print(datos['Potencia'])

print("*"*80)
# Cambiar el tipo de datos de Potencia
datos['Potencia'] = pd.to_numeric(datos['Potencia'], errors='coerce')
print(datos['Potencia'])
datos['Potencia'].replace(np.nan, datos['Potencia'].mean(), inplace=True)


print("*"*80)
# Otras estadisicas
print(datos.info())
print(datos["Nombre"].value_counts())

print("*"*80)
## Visualización de datos
# Gráfico de caja para precio
plt.boxplot(datos['Precio'])
plt.title("Precio de los coches")
plt.show()

# Grafico de dispersión de precio vs potencia
y = datos['Precio']
x = datos['Potencia']
plt.scatter(x, y)
plt.title("Precio vs Potencia")
plt.xlabel("Potencia")
plt.ylabel("Precio")
plt.show()

# Grafico de dispersión de precio vs motor
y = datos['Precio']
x = datos['Motor']
plt.scatter(x, y)
plt.title("Precio vs Motor")
plt.xlabel("Motor")
plt.ylabel("Precio")
plt.show()


# Coeficiente de correlación de Pearson y p-valor
"""
    Coeficiente de correlación de Pearson
        Si p(x,y) prox a 1, correlación positiva
        Si p(x,y) prox a -1, correlación negativa
        Si p(x,y) prox a 0, no hay correlación

    P valor
        Si p valor < 0.001, fuerte certeza en el resultado
        Si p valor < 0.05, moderada certeza en el resultado
        Si p valor < 0.1, débil certeza en el resultado
        Si p valor > 0.1, no hay certeza de correlación
"""

# Borrar NaN de los datos de Motor y Precio
datos['Motor'].replace(np.nan, datos['Motor'].mean(), inplace=True)
datos['Precio'].replace(np.nan, datos['Precio'].mean(), inplace=True)

pearson_coef, p_value = stats.pearsonr(datos['Motor'], datos['Precio'])
print(f"El coeficiente de correlación de Pearson es: {pearson_coef}")
print(f"El p-valor es: {p_value}")

"""
    El coeficiente de correlación de Pearson es: 0.6571175434760662
    El p-valor es: 0.0

    Por lo tanto, hay una correlación positiva entre el motor y el precio
    y una fuerte certeza en el resultado.
"""
# Estadisticas por año
columnas = ['Anho', 'Kilometros recorridos', 'Motor', 'Potencia', 'Precio']
datos_nuevos = datos[columnas]
print(datos_nuevos.head())

datos_agrupados = datos_nuevos.groupby(['Anho'], as_index=False).mean()
print(datos_agrupados.head())

# Guardar los datos
datos.to_csv('datosModificados.csv')