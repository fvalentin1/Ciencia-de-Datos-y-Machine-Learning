import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Creamos conjunto de datos con gastos en publicidad y ventas
datos = {
    'Gasto_publicidad': [100, 200, 300, 400, 500, 200, 300, 400, 500],
    'Ventas': [800, 900, 700, 600, 500, 900, 700, 600, 500]
}

# Convertimos el diccionario en un DataFrame
df = pd.DataFrame(data=datos)

# Separamos las variables independientes de la variable dependiente
X = df[['Gasto_publicidad']]
y = df['Ventas']

# Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el modelo de regresión lineal
modelo = LinearRegression()

# Entrenamos el modelo
modelo.fit(X_train, y_train)

# Hacemos predicciones
y_pred = modelo.predict(X_test)

# Coeficientes, pendiente e intersección
print("Coeficiente: ", modelo.coef_)
print("Pendiente: ", modelo.coef_[0])
print("Intersección: ", modelo.intercept_)

# Evaluamos el modelo
print("Error cuadrático medio (MSE): ", mean_squared_error(y_test, y_pred))
print("Coeficiente de determinación (R^2): ", r2_score(y_test, y_pred))

# Visualizamos el modelo de regresión lineal
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title("Regresión lineal: Gasto en publicidad vs Ventas")
plt.xlabel("Gasto en publicidad")
plt.ylabel("Ventas")
plt.show()
