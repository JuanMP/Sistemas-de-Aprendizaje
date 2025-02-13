import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, semilla=1):
        self.eta = eta
        self.n_iter = n_iter
        self.semilla = semilla
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.semilla)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errores_ = []
        for _ in range(self.n_iter):
            errores = 0
            for xi, target in zip(X, y):
                incremento = self.eta * (target - self.predict(xi))
                self.w_[1:] += incremento * xi
                self.w_[0] += incremento
                errores += int(incremento != 0.0)
            self.errores_.append(errores)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
v1 = np.array([1,2,3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


#PASO 2: PREPARAR EL DATASET

#Leer el Dataset
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
print('Leyendo dataset de iris')
print('---- Ultimos 5 ejemplos:\n', df.tail())


y = df.iloc[0:100, 4].values        #seleccionar ejemplos de setosa y versicolor
y = np.where(y == 'Iris-setosa', -1,1) #codificamos en +1 (Iris versicolor) y -1 (Iris-setosa)
X = df.iloc[0:100, [0, 2]].values    #extraer longitud de sépalos y pétalos
#Dibujar los datos
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.ylabel('Longitud sépalo [cm]')
plt.xlabel('Longitud pétalo  [cm]')
plt.legend(loc='upper left')
plt.title(loc='center', label='Dataset Iris de Juan')
plt.show() #plt.savefig('images/02/06.png', dpi=300)


#PASO 3: CREAR Y ENTRENAR UN PERCEPTRÓN
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errores_) + 1), ppn.errores_, marker='o')
plt.xlabel('Epocas')
plt.ylabel('Nº de Errores')
plt.title('Entrenamiento de Juan Martínez')
plt.show() #plt.savefig('images/02/06.png', dpi=300)


#PASO 4: Función para dibujar regiones de decisión
def plot_regiones(X, y, clasificador, resolucion=0.02):
    marcadores = ('s', 'o', '^', 'v')
    colores = ('red', 'blue', 'lightgreen', 'gray')
    cmap = ListedColormap(colores[:len(np.unique(y))])
    #Dibujar la superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolucion),
                           np.arange(x2_min, x2_max, resolucion))
    Z = clasificador.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #Dibujar la clase de los ejemplos
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=colores[idx], marker=marcadores[idx],
                    label=cl, edgecolor='black')
        
plot_regiones(X, y, clasificador=ppn)
plt.xlabel('longitud sépalos [cm]')
plt.ylabel('longitud pétalos [cm]')
plt.legend(loc='upper left')        #plt.savefig('images/U02_P01_3.png', dpi=300)
plt.title('Regiones de decisión de Juan Martínez')
plt.show()
