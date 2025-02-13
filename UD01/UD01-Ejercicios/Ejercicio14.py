import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Parte a)

#Ejemplo de Dataframe
data = {
    'color': ['rojo', 'verde', 'azul', 'rojo', 'verde'],
    'talla': ['S', 'M', 'L', 'XL', 'S'],
    'precio': [10, 15, 20, 25, 30]
}
df = pd.DataFrame(data)

#Seleccionamos las columnas relevantes
X = df[['color', 'talla', 'precio']].values

#Instancia del objeto LabelEncoder
le = LabelEncoder()

#Aplicamos el LabelEncoder a la columna de color
X[:, 0] = le.fit_transform(X[:, 0])

#Datos transformados
print("Datos originales\n", df)
print("X transformado\n", X)


#Parte b)
from sklearn.preprocessing import OneHotEncoder

#Instanciamos el objeto OneHotEncoder
ohe = OneHotEncoder()

# Aplica el OneHotEncoder a la columna de color (index 0)
X_color_encoded = ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

#Concatenamos la transformación con las demás columnas
X_transformed = pd.concat([
    pd.DataFrame(X_color_encoded, columns=ohe.get_feature_names_out(['color'])),
    pd.DataFrame(X[:, 1:], columns=['talla', 'precio'])
], axis=1)

#Imprimir los datos transformados
print("Datos transformados con OneHotEncoder\n", X_transformed)



#Parte c)
#Con get_dummies aplicado a las columnas color, talla y precio
df_dummies = pd.get_dummies(df, columns=['color', 'talla'])

print("DataFrame con dummies\n", df_dummies)


#Parte d)
#Con get_dummies con drop_first=True (eliminando la primera columna)
df_dummies_reduced = pd.get_dummies(df, columns=['color', 'talla'], drop_first=True)

print("DataFrame con dummies drop_first=True\n", df_dummies_reduced)
