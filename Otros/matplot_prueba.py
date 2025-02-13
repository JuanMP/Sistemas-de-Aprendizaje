import matplotlib.pyplot as plt
import numpy as np

#PRIMERAS GRÁFICAS (PÁG 46 PDF)



x = np.linspace(0, 10, 50)
senos = np.sin(x)
plt.plot(x, senos, "o")
plt.show()


#x = np.linspace(0, 10, 50)
#senos = np.sin(x)
#cosenos = np.cos(x)
#plt.plot(x, senos, "-b", x, senos, "ob", x, cosenos, "-r", x, cosenos, "or")
#plt.xlabel("Este es el eje x!") # Etiqueta del eje X
#plt.ylabel("Este es el eje y!") # Etiqueta del eje Y
#plt.title("Mis primeras gráficas") # Título
#plt.show()


# Parecido al anterior pero indicando el nombre de cada cosa de manera individual
#x = np.linspace(0, 10, 50)
#senos = np.sin(x)
#cosenos = np.cos(x)
#plt.plot(x, senos, label = "seno", color="blue", linestyle = "--", linewidth = "2")
#plt.plot(x, cosenos, label = "coseno", color="red", linestyle = "-", linewidth = "1")
#plt.legend() # leyenda
#plt.show()



# importamos un dataset con pandas (si no tenemos el dataset lo descarga con una URL)
#import pandas as pd
#try:
# salarios = pd.read_csv("salary_table.csv")
#except:
# url = "https://media.githubusercontent.com/media/neurospin/pystatsml/master/datasets/salary_table.csv"
# salarios = pd.read_csv(url)
#df = salarios

# Gráfico nube puntos con colores asociados a valores del dataset con un diccionario
#colores = colores_edu = {'Bachelor':'r', 'Master':'g', 'Ph.D':'blue'}
#plt.scatter(df['experience'], df['salary'],
# c=df['education'].apply(lambda x: colores[x]), s=100)
#plt.show()


