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

# Creamos el modelo de regresión lineal
lm = LinearRegression()

# Separamos las variables independientes de la variable dependiente
X = datos[['Motor']]
y = datos['Precio']

# Entrenamiento o ajuste del modelo
lm.fit(X, y)

# Hacemos predicciones
prediccion = lm.predict(X)

a = lm.intercept_
b = lm.coef_[0]

# Visualizamos el modelo de regresión lineal
print(f"y = {a} + {b}x")

# Valores de la predicción
print("Predicción ", prediccion)

# Coeficiente de determinación
print("Coef. Determinación del modelo: ", lm.score(X, y))

# Visualizamos el modelo de regresión lineal
sns.regplot(x='Motor', y='Precio', data=datos)
# Visualizar residuales, prediccion - error
sns.residplot(x=datos['Motor'], y=datos['Precio'])
plt.title("Residuales")
plt.show()

## Separamos en entrenamiento y prueba

print("*"*50)
print("")
print("*"*50)


# Separamos las variables independientes de la variable dependiente
X = datos[['Motor']]
y = datos['Precio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Entrenamiento o ajuste del modelo
lm.fit(X_train, y_train)

# Hacemos predicciones
prediccion = lm.predict(X_test)

a = lm.intercept_
b = lm.coef_[0]

# Visualizamos el modelo de regresión lineal
print(f"y = {a} + {b}x")

# Valores de la predicción
print("Predicción ", prediccion)

#print(y_test)

# Coeficiente de determinación
print("Coef. Determinación del modelo: ", lm.score(X_train, y_train))

# Visualizamos el modelo de regresión lineal
sns.regplot(x='Motor', y='Precio', data=datos)
# Visualizar residuales, prediccion - error
sns.residplot(x=datos['Motor'], y=datos['Precio'])
plt.title("Residuales")
plt.show()



