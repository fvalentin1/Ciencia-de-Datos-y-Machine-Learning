import pandas as pd

# Carga de datos
datos = pd.read_csv('precios_coches.csv')

print(datos.head())

print("*"*50)

# Convertir la variable Fuel_Type a una variable categórica
#columna_dummies = pd.get_dummies(datos['Fuel_Type'])
#print(columna_dummies)

datos_dummies = pd.get_dummies(datos, columns=['Fuel_Type'])

print(datos_dummies.head())