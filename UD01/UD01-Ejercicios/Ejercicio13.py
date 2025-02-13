import numpy as np
import pandas as pd

#PARTE A
#Creación de DataFrame
df = pd.DataFrame([['verde', 'M', 10.1, 'clase2'],
                   ['rojo', 'L', 13.5, 'clase1'],
                   ['azul', 'XL', 15.3, 'clase2']])
df.columns = ['color', 'talla', 'precio', 'class']

#Mostrar el Dataframe original
print("Dataframe original:")
print(df)


#PARTE B
#Crear el mapeo de valores
mapeo = {label: idx for idx, label in enumerate(np.unique(df['class']))}
mapeo_inverso = {v: k for k, v in mapeo.items()}

#Aplicar el mapeo
df['class'] = df['class'].map(mapeo)
print("\nDataFrame con 'class' transformada:")
print(df)

#Revertir la transformación
df['class'] = df['class'].map(mapeo_inverso)
print("\nDataFrame con 'class' revertida:")
print(df)

#PARTE C
from sklearn.preprocessing import LabelEncoder

#Crear el objeto LabelEncoder
le = LabelEncoder()

#Aplicar el LabelEncoder y mostrar la columna transformada
df['class'] = le.fit_transform(df['class'])
print("\nDataFrame con 'class' transformada usando LabelEncoder:")
print(df)

#Revertir la transformación
df['class'] = le.inverse_transform(df['class'])
print("\nDataFrame con 'class' revertida usando LabelEncoder:")
print(df)

