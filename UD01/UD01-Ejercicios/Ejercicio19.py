import pandas as pd
import numpy as np

#a) Cargar archivo csv en DataFrame
datos = pd.read_csv('mpg.csv')

#Eliminar la columna 'nombre'
datos_sin_nombre = datos.drop(columns=['nombre'])

#Primeras filas
print(datos_sin_nombre.head())

#b) Versión escalada normalizando los datos de las columnas
#Crear una copia del DataFrame para los datos escalados
datos_scaled = datos_sin_nombre.copy()

#Escalado de los datos numéricos
numerical_columns = datos_scaled.select_dtypes(include=['float64', 'int64']).columns
datos_scaled[numerical_columns] = datos_scaled[numerical_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#Primeras filas
print(datos_scaled.head())


#c) Lo mismo con StandarScaler
from sklearn.preprocessing import StandardScaler

#Crear una copia del DataFrame para los datos escalados con StandardScaler
scaler = StandardScaler()
datos_scaled_sklearn = datos_sin_nombre.copy()

#Aplicar el escalado a las columnas numéricas
datos_scaled_sklearn[numerical_columns] = scaler.fit_transform(datos_scaled_sklearn[numerical_columns])

#Mostrar las primeras filas de los datos escalados con StandardScaler
print(datos_scaled_sklearn.head())

#d) Encontrar outliers en las columnas con el método de la mediana
#Función para detectar outliers usando la mediana y el IQR
def detectar_outliers_mediana(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    return outliers

#Detectar outliers en las columnas numéricas
outliers_mediana = detectar_outliers_mediana(datos_sin_nombre[numerical_columns])

#Mostrar los outliers detectados
print(outliers_mediana)

#e) Encontrar outliers en las columnas con los z-scores
def detectar_outliers_zscore(df):
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = z_scores > 3
    return outliers

#Detectar outliers con z-scores
outliers_zscore = detectar_outliers_zscore(datos_sin_nombre[numerical_columns])

#Mostrar los outliers detectados
print(outliers_zscore)


#f) Detectar outliers dibujando boxplots
import matplotlib.pyplot as plt
import seaborn as sns

#Dibujar boxplots de las columnas numéricas
plt.figure(figsize=(12, 6))
sns.boxplot(data=datos_sin_nombre[numerical_columns])
plt.title("Boxplots de las columnas numéricas")
plt.xticks(rotation=90)
plt.show()
