import matplotlib.pyplot as plt
import numpy as np


#Preguntar al usuario
w0 = 4
w1 = 7

#Crear el intervalo [-5,5]
x = np.linspace(-5, 5, 100)

#Calcular valores de y con la ecuación y = w0 + w1 x
y = w0 + w1 * x

#Gráfico
plt.figure(figsize=(10,8))
plt.plot(x, y, label=f"y = {w0} + {w1}x", color='blue')
plt.title("Gráfica de los puntos w0 y w1 con el intervalo [-5,5]")
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()


