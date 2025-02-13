import pandas as pd

#EJEMPLO DE DATAFRAME

col = ['jugados', 'ganados', 'perdidos']
fil = ['VCF', 'Betis', 'ATM', 'FCB']

datos = [{'jugados':3, 'ganados':3, 'perdidos':0},
         {'jugados':3, 'ganados':2, 'perdidos':0},
         {'jugados':3, 'ganados':2, 'perdidos':1},
         {'jugados':3, 'ganados':0, 'perdidos':3}]

df1 = pd.DataFrame(data=datos, index=fil, columns=col)  #Pandas

#Todo el dataframe
#print(df1)

#Acceder por nombres (Solo datos del Valencia)
#print(df1.loc['VCF'])

#Acceder por posiciones (Solo datos del Valencia)
#print(df1.iloc[0])

#Añadir una serie al dataframe (Añadir partidos empatados)
df1['empatados'] = df1['jugados'] - df1['ganados'] - df1['perdidos']
print(df1)


#17/10/2024
#Creado excel en un Calc y exportado como csv y leído aquí
ds2 = pd.read_csv("ds_2.csv", delimiter=";" )

#Imprimir desde arriba o abajo
#print(ds2.head(2))
#print(ds2.tail(2))

#Imprime datos como medias y sumas
#print(ds2.describe())
#Imprime los tipos de datos
#print(ds2.info())
#print("Columnas", ds2.values)
#print(ds2.iloc[0:2,1:3])

#Imprime a lo largo
#print(ds2.T)

#Imprime los que tienen más peso de 72
#print(ds2 [ ds2['Peso'] >72])

#Borra el campo edad
ds2 = ds2.drop( ['Edad'], axis=1)
print(ds2)



