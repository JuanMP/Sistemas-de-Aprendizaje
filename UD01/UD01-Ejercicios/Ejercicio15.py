import numpy as np
import matplotlib.pyplot as plt

#Pedir al usuario los valores
a = int(input("Dime el valor de a: "))
b = int(input("Dime el valor de b: "))

#Crear intervalo (array creo)
x = np.linspace(-5, 5, 100)

#Calcular valores de f(x) = ax^2+b (función)
fx = a * x**2 + b

#Gráfico
plt.figure(figsize=(10,8))
plt.plot(x, fx, label=f"f(x) = {a}x² + {b}", color='blue')
plt.title("Gráfica de valores a y b con función f(x) en el intervalo [-5,5] ")
plt.ylabel("f(x)")
plt.xlabel("x")
plt.legend()
plt.show()

