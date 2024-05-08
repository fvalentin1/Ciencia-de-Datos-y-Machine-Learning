from joblib import dump, load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm
import time

# Cargar el CSV
def carga_datos(nombre):
    '''Carga un CSV en un DataFrame de Pandas. Si no existe el directorio 'datos', lo crea.'''
    directorio = 'datos'
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    fichero = os.path.join(directorio, nombre)
    datos = pd.read_csv(fichero)
    return datos

# Guardar el CSV
'''Guarda un DataFrame de Pandas en un CSV. Si no existe el directorio 'datos', lo crea.'''
def guarda_datos(df, nombre='ventas.csv'):
    directorio = 'datos'
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    fichero = os.path.join(directorio, nombre)
    df.to_csv(fichero, index=False)
    print('Datos guardados en', fichero)

# Verificar si el día es festivo
def es_festivo(fecha):
    ''''Devuelve 1 si la fecha es festiva, 0 si no lo es.'''
    festivos = ['01-01', '06-01', '28-02', '01-05', '15-08', '12-10', '01-11', '06-12', '08-12', '25-12']
    fecha_sin_ano = fecha.strftime('%d-%m') # Se ignora el año
    dia_semana = fecha.weekday() # 0: lunes, 6: domingo
    if fecha_sin_ano in festivos or dia_semana == 6:
        return 1 # Festivo
    else:
        return 0 # No festivo
    
# Obtener la estación del año    
def obtener_estacion(mes):
    '''Devuelve la estación del año a partir del mes.'''
    if mes in (1, 2, 12):
        return 'Invierno'
    elif mes in (3, 4, 5):
        return 'Primavera'
    elif mes in (6, 7, 8):
        return 'Verano'
    else:
        return 'Otoño'
    
# Preparar las fechas en el dataframe    
def prepara_fechas(df):
    '''Prepara las fechas para el modelo en el dataframe.'''
    df['AÑO'] = (df['FECHA']).str[:4]
    df['MES'] = (df['FECHA']).str[5:7]
    df['DIA'] = (df['FECHA']).str[8:10]
    # Lo anterior es porque la fecha está en tipo string
    # Se elimina la columna de fecha
    df.drop('FECHA', axis=1, inplace=True)
    # Se reconstruye la fecha con formato fecha
    df['FECHA'] = df['AÑO']+'-'+df['MES']+'-'+df['DIA']
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['DIA_SEMANA'] = df['FECHA'].dt.weekday
    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    df['NOMBRE_DIA_SEMANA'] = df['FECHA'].dt.dayofweek.apply(lambda x: dias_semana[x])
    df['FESTIVO'] = df['FECHA'].apply(es_festivo)
    df['ESTACION'] = df['MES'].apply(obtener_estacion)
    return df

# Preparar los datos para el modelo, para que las variables categóricas sean numéricas
def prepara_datos_modelo(df):
    '''Prepara los datos para el modelo, para que las variables categóricas sean numéricas.'''
    df_modelo = df[['TIENDA', 'TOTAL', 'AÑO', 'MES', 'DIA', 'DIA_SEMANA', 'FESTIVO', 'ESTACION']].copy()
    mapeo_estaciones = {'Invierno': 0, 'Primavera': 1, 'Verano': 2, 'Otoño': 3}
    df_modelo['ESTACION'] = df_modelo['ESTACION'].map(mapeo_estaciones)
    mapeo_tiendas = { 'Tienda_00': 0, 'Tienda_01': 1, 'Tienda_02': 2, 'Tienda_03': 3, 'Tienda_04': 4 , 'Tienda_05': 5}
    df_modelo['TIENDA'] = df_modelo['TIENDA'].map(mapeo_tiendas)
    guarda_datos(df_modelo, 'ventas_modelo.csv')

    # Se separan las variables predictoras de la variable objetivo
    X = df_modelo.drop('TOTAL', axis=1)
    y = df_modelo['TOTAL']
    return X, y

# Entrenar el modelo
def entrenar_modelo(X,y):
    '''Entrena un modelo de Random Forest Regressor y Regresión Lineal. Devuelve el modelo entrenado.'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    # Random Forest Regressor
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=40)
    modelo_rf.fit(X_train, y_train)
    # Regresión Lineal
    modelo_rl = LinearRegression()
    modelo_rl.fit(X_train, y_train)
    dump(modelo_rf, './modelos/modelo_rf.joblib')
    dump(modelo_rl, './modelos/modelo_rl.joblib')
    return modelo_rf, modelo_rl, X_test, y_test

# Evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test, nombre_modelo):
    '''Evalúa un modelo con el conjunto de datos de prueba y devuelve el error cuadrático medio.'''
    print(f'Evaluando modelo {nombre_modelo}...')
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
    plt.xlabel('Valor real')
    plt.ylabel('Predicción')
    plt.title('Predicciones vs Valores reales con '+nombre_modelo)
    plt.grid(True)
    plt.show()

    print(f'Metricas del modelo {nombre_modelo}')
    print(f'Error cuadrático medio: {mse:.2f}')
    print(f'Coeficiente de determinación R2: {r2:.2f}')
    return mse, r2

# Main
if __name__=='__main__':
    datos = carga_datos('ventas.csv')
    print(datos.head())
    print("*"*80)
    datos = prepara_fechas(datos)
    print(datos.head())
    print("*"*80)
    X, y = prepara_datos_modelo(datos)
    print(X.head())
    print("*"*80)
    print("Entrenando modelos...")
    with tqdm(total=100) as pbar:
        for i in range(2):
            modelo_rf, modelo_rl, X_test, y_test = entrenar_modelo(X, y)
            time.sleep(0.1)  # Simula el tiempo de entrenamiento
            pbar.update(50)  # Actualiza la barra de progreso
    print("Modelos entrenados")
    print("*"*80)
    mseRF, r2RF = evaluar_modelo(modelo_rf, X_test, y_test, 'Random Forest Regressor')
    print("*"*80)
    mseLR, r2LR = evaluar_modelo(modelo_rl, X_test, y_test, 'Regresión Lineal')