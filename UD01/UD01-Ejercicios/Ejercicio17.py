import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#a) Datos
datos =  [[3,0.4], [8, 0.6], [18, 1.2], [27, 1.4]]
df = pd.DataFrame(datos, columns=['Peso', 'Altura'])

#b) Dibujar un scatterplot()
plt.figure(figsize=(10,8))
plt.scatter(df['Peso'], df['Altura'], color='blue', label='Datos')
plt.ylabel('Altura')
plt.xlabel('Peso')
plt.title('Scatterplot de Peso y Altura')
plt.legend()
plt.show()


#c) Punto de corte y pendiente de una recta
w0 = 2
w1 = 5

#d) Dibujo de la recta en el gráfico con color azul
x_vals = np.linspace(min(df['Peso']), max(df['Peso']), 100)  #Peso
y_vals = w0 + w1 * x_vals  #Recta
plt.plot(x_vals, y_vals, color='blue', label=f"Recta: y = {w0} + {w1}x")
plt.legend()
plt.show()

#e) Calculo del cuadrado de la distancia de cada punto de datos (cuando x=peso)
distanciapuntos = (df['Altura'] - (w0 + w1 * df['Peso']))**2
print(distanciapuntos)

#f) Cálculo de la media de la suma de los cuadrados de esas diferencias
media_suma = np.mean(distanciapuntos)
print(f"\nMedia_suma: {media_suma}") #acaba en error

#g) Creación de varias rectas para bajar el error de 0.5
while media_suma > 0.5:
    print("El error cuadrado medio es mayor a 0.5, vuelve a intentarlo")
    w0 = float(input("Dime un nuevo punto de corte: "))
    w1 = float(input("Dime una nueva pendiente: "))
    
    #Recalcular las distancias
    distanciapuntos = (df['Altura'] - (w0 + w1 * df['Peso']))**2
    media_suma = np.mean(distanciapuntos)
    print(f"\nNueva Media_suma: {media_suma}") #del error de f
    
print("\n¡Error cuadrado medio reducido por debajo de 0.5!")


#h) Fórmula con vectores y matrices
#Crear la matriz X y el vector y
X = np.hstack([np.ones((len(df), 1)), df['Peso'].values.reshape(-1, 1)])  # Matriz X con columna de unos y pesos
y = df['Altura'].values.reshape(-1, 1)  # Vector y con alturas

#Calcular los parámetros W = (X^T X)^-1 X^T y
W = np.linalg.inv(X.T @ X) @ X.T @ y
w0_opt, w1_opt = W.flatten()

print("\nParámetros óptimos calculados:")
print(f"w0: {w0_opt:.4f}, w1: {w1_opt:.4f}")

#Recta en scatterplot
plt.figure(figsize=(10,8))
plt.scatter(df['Peso'], df['Altura'], color='red', label='Datos (Peso vs Altura)')
plt.plot(x_vals, w0_opt + w1_opt * x_vals, color='green', label=f"Recta óptima: y = {w0_opt:.2f} + {w1_opt:.2f}x")
plt.ylabel("Altura")
plt.xlabel("Peso")
plt.title("Recta")
plt.legend()
plt.show()