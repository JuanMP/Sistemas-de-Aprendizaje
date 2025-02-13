# coding: utf8
# **********************
# ** EJEMPLO DE Q-LEARNING **
# **********************
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Define el grafo del laberinto mediante una matriz de conexión
# -1 Significa que no hay enlace entre un lugar y otro por ejemplo
# El estado a no puede ir al b directamente
R = np.array([[-1, -1, -1, -1, 0, -1],
                [-1, -1, -1, 0, -1, 100],
                [-1, -1, -1, 0, -1, -1],
                [-1, 0, 0, -1, 0, -1],
                [0, -1, -1, 0, -1, 100],
                [-1, 0, -1, -1, 0, 100]]).astype("float32")

Q = np.zeros_like(R) #El conocimiento
gamma = 0.8 #Ratio de aprendizaje
estado_inicial = randint(0,4) #Inicializar aleatoriamente el estado

def acciones_posibles(estado):    #Acciones posibles en este estado
    fila_estado_actual = R[estado,]
    posibilidades = np.where(fila_estado_actual >= 0)[0]
    return posibilidades

def escoge_sig_paso(movimientos_disponibles):   #Escoge aleatoriamente una de las disponibles
    siguiente = np.random.choice(movimientos_disponibles, 1)[0]
    return siguiente

def actualiza(estado_actual, accion, gamma):    # actualiza la Q-matriz según la ruta seleccionada
    max_idx = np.where(Q[accion,] == np.max(Q[accion,]))[0]
    if max_idx.shape[0] > 1:
        max_idx = np.random.choice(max_idx, size = 1)[0]
    else:
        max_idx = max_idx[0]
    max_valor = Q[accion, max_idx]
    Q[estado_actual, accion] = R[estado_actual, accion] + gamma * max_valor #formula del Q learning
    
    
pasos_disponibles = acciones_posibles(estado_inicial) #Ver acciones posibles
accion = escoge_sig_paso(pasos_disponibles)
for i in range(100):    #Entrenar iteraciones
        estado_actual = np.random.randint(0, int(Q.shape[0]))
        pasos_disponibles = acciones_posibles(estado_actual)
        accion = escoge_sig_paso(pasos_disponibles)
        actualiza(estado_actual, accion, gamma)
print("Q matriz entrenada: \n", Q / np.max(Q) * 100)    #Normalizar la Q matriz entrenada
# Testing
estado_actual = 2
pasos = [estado_actual]
while estado_actual != 5:
        idx_sig_paso = np.where(Q[estado_actual,] == np.max(Q[estado_actual]))[0]
        if idx_sig_paso.shape[0] > 1:
            idx_sig_paso = np.random.choice(idx_sig_paso, size =1)[0]
        else:
            idx_sig_paso = idx_sig_paso[0]
        pasos.append(idx_sig_paso)
        estado_actual = idx_sig_paso
print("Mejor secuencia de ruta: ", [int(x) for x in pasos])
    
        