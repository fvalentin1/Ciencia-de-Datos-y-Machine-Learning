from datetime import datetime
from arboles_regresion import obtener_estacion, guarda_datos
import pandas as pd
import joblib

# Predecir
def predecir(modelo):
    tienda = input("Introduce el nombre de la tienda: ")
    ano = int(input("Introduce el año: "))
    mes = int(input("Introduce el mes: "))
    dia = int(input("Introduce el día: "))
    es_festivo = input("¿Es festivo? (Si / No): ")
    fecha = f'{ano}/{mes}/{dia}'
    fecha = datetime.strptime(fecha, '%Y/%m/%d')
    dia_semana = fecha.weekday()

    tiendas = { 'Tienda_00': 0, 'Tienda_01': 1, 'Tienda_02': 2, 'Tienda_03': 3, 'Tienda_04': 4 , 'Tienda_05': 5}
    tienda_num = tiendas[tienda]
    festivos = {'No': 0, 'Si': 1}
    festivo_num = festivos[es_festivo]
    estacion = obtener_estacion(mes)
    estaciones = {'Invierno': 0, 'Primavera': 1, 'Verano': 2, 'Otoño': 3}
    estacion_num = estaciones[estacion]
    datos = {
        'TIENDA': [tienda_num,],
        'AÑO': [ano,],
        'MES': [mes,],
        'DIA': [dia,],
        'DIA_SEMANA': [dia_semana,],
        'FESTIVO': [festivo_num,],
        'ESTACION': [estacion_num,]
    }

    df_datos = pd.DataFrame(datos)
    prediccion = modelo.predict(df_datos)
    return df_datos, prediccion

# Cargar modelo
def cargar_modelo(nombre='modelo_rf.joblib'):
    archivo = f"./modelos/{nombre}"
    modelo = joblib.load(archivo)
    return modelo

# Main
if __name__ == '__main__':
    modelo = cargar_modelo()
    datos, prediccion = predecir(modelo)
    print(f'Los datos son: {datos}')
    print(f'La predicción es: {prediccion}')

    datos['PREDICCIÓN'] = prediccion

    guarda_datos(datos, 'prediccion.csv')
