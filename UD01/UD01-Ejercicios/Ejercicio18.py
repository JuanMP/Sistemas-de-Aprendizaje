import pandas as pd

#Cargar archivo csv en DataFrame
datos = pd.read_csv('mpg.csv')

#Primeras filas
print(datos.head())

#Eliminación de la columna nombre
datos = datos.drop(columns=['nombre'])

#Mostrar las primeras filas después de la eliminación
print(datos.head())

#Versión escalada en 0,1 de los datos con código propio
#Crear una copia del DataFrame para los datos escalados
datos_scaled = datos.copy()

#Escalar los datos numéricos
numerical_columns = datos_scaled.select_dtypes(include=['float64', 'int64']).columns
datos_scaled[numerical_columns] = datos_scaled[numerical_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#Mostrar las primeras filas de los datos escalados
print(datos_scaled.head())

#Usar MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

#Inicializar el MinMaxScaler
scaler = MinMaxScaler()

#Aplicar el escalado
datos_scaled_sklearn = datos.copy()
datos_scaled_sklearn[numerical_columns] = scaler.fit_transform(datos[numerical_columns])

#Verificar la transformación
datos_scaled_sklearn.head()

