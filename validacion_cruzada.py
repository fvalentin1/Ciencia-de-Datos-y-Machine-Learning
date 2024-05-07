import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

# Cargamos el dataset de propinas
propinas = sns.load_dataset('tips')
print("Estadísticas del dataset de propinas: ")
print(propinas.describe())
print("*"*50)
print("Información del dataset de propinas: ")
print(propinas.info())

# Limpiamos los datos
propinas_limpias = propinas.dropna().drop_duplicates()
print("Estadísticas del dataset de propinas limpio: ")
print(propinas_limpias.describe())
print("*"*50)
print("Información del dataset de propinas limpio: ")
print(propinas_limpias.info())

# Excluir columnas no numéricas
propinas_limpias_numericas = propinas_limpias.select_dtypes(include=[np.number]) # Incluimos solo las columnas numéricas

# Graficamos los datos
sns.heatmap(propinas_limpias_numericas.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación entre variables')
plt.show()

# Definimos las variables predictoras y la variable objetivo
X = propinas_limpias.drop('tip', axis=1)
X = X.drop('time', axis=1)
y = propinas_limpias['tip']

# Creamos el modelo
modelo = Pipeline([
    ('preprocesador', ColumnTransformer(
        [
            ('scaler', StandardScaler(), ['total_bill', 'size']),
            ('onehot', OneHotEncoder(), ['day', 'smoker', 'sex'])
        ],remainder='passthrough'
    )),
    ('regressor', RandomForestRegressor())
])

# Evaluamos el modelo
scores = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print("\nResultados de la validación cruzada: ")
print("RMSE medio: ", rmse_scores.mean())
print("Desviación estándar: ", rmse_scores.std())

# Entrenamos el modelo
modelo.fit(X, y)
fichero = 'modelo_random_forest.pkl'

# Guardamos el modelo
joblib.dump(modelo, fichero)
print("Modelo guardado en fichero: ", fichero)

# Cargamos el modelo
modelo_cargado = joblib.load(fichero)

# Realizamos predicciones
X_nuevos = X.head(5)
predicciones = modelo_cargado.predict(X_nuevos)

# Mostramos las predicciones
print("\nPredicciones p: ")
print(X_nuevos)
print(predicciones)
