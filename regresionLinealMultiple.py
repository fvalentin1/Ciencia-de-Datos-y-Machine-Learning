import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import seaborn as sns

# Carga de datos
datos = pd.read_csv('datosModificados.csv')
datos.set_index('Indice', inplace=True)
print(datos.head())

# Columnas con nulos
columnas_nulos = [col for col in datos.columns if datos[col].isnull().any()]
print("Columnas con nulos: ", columnas_nulos)

# Todas las columnas
print("Todas las columnas", datos.columns)

# Hot encoding
datos_dummies = pd.get_dummies(datos['Combustible'])
datos = pd.concat([datos, datos_dummies], axis=1)
print(datos.head())

# Normalización de datos
datos['Motor'] = datos['Motor'] / datos['Motor'].max()
datos['Potencia'] = datos['Potencia'] / datos['Potencia'].max()

# Creamos el modelo de regresión lineal
lm = LinearRegression()

# Separamos las variables independientes de la variable dependiente
X = datos[['Motor', 'Potencia', 'Asientos', 'CNG', 'Diesel', 'Electric', 'LPG', 'Petrol']]
y = datos['Precio']

#print(X.head())

# Entrenamiento
lm.fit(X, y)

# Predicciones
prediccion = lm.predict(X)
a = lm.intercept_
b1 = lm.coef_[0]
b2 = lm.coef_[1]
b3 = lm.coef_[2]
b4 = lm.coef_[3]
b5 = lm.coef_[4]
b6 = lm.coef_[5]
b7 = lm.coef_[6]
b8 = lm.coef_[7]

# Mostramos la ecuación
#print(f"y = {a} + {b1}x1 + {lm.coef_[1]}x2 + {lm.coef_[2]}x3 + {lm.coef_[3]}x4 + {lm.coef_[4]}x5 + {lm.coef_[5]}x6 + {lm.coef_[6]}x7 + {lm.coef_[7]}x8")

print('*'*50)
print("Predicción")
print(prediccion)
print(y)
print(lm.score(X, y))

# Comparativa
resultado = {'Real': datos['Precio'], 'Predicción': prediccion}
R = pd.DataFrame(data=resultado)
print(R.head())

ECM = mean_squared_error(y, prediccion)
R2 = r2_score(y, prediccion)

print("Error cuadrático medio: ", ECM)
print("Coeficiente de determinación: ", R2)

# Visualizamos el modelo de regresión lineal
#plt.plot(datos['Motor'], prediccion, color='red',label='Predicción')
plt.scatter(datos['Motor'], y, label='Datos reales')
plt.scatter(datos['Motor'], prediccion, color='red',label='Predicción')
plt.title("Regresión lineal múltiple")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()