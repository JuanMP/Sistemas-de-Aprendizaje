import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGDB(object):
    """Clasificador ADAptive LInear NEuron de Juan
    Parámetros:
    ------------
    eta: float Learning rate (entre 0.0 y 1.0)
    n_iter: int max. repasos que da (iteraciones) al dataset train
    semilla: int; semilla para el GMA
    Atributos:
    ------------
    w_: Id-array pesos que aprende después de entrenar
    coste_: lista valores de la función de coste SSE en cada época (repaso)
    """
    def __init__(self, eta=0.01, n_iter=50, semilla=1):
        self.eta = eta
        self.n_iter = n_iter
        self.semilla = semilla
        
    def fit(self, X, y):
        """ Entrena (Aprende)
        Parámetros:
        --------------
        X: {array}, estructura = [n_ejemplos, n_características]
        y: array, estructura = [n_ejemplos] valores target/label
        Devuelve:
        ----------
        self:object
        """
        rgen = np.random.RandomState(self.semilla)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.coste_ = []
        for _ in range(self.n_iter):
            entradas = self.entradas(X)
            salidas = self.activacion(entradas)
            errores = (y - salidas)
            self.w_[1:] += self.eta * X.T.dot(errores)
            self.w_[0] += self.eta * errores.sum()
            coste = (errores**2).sum() / 2.0
            self.coste_.append(coste)
        return self
    
    #Añadido Adaline
    def entradas(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    #def net_input(self, X):
    #    return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.entradas(X) >= 0.0, 1, -1)
    
    def activacion(self, X):
        """Calcula la activación lineal"""
        return X
    
    
    
v1 = np.array([1,2,3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))




#PASO 2: PREPARAR EL DATASET
#Leer el Dataset pidiéndolo
#lugar = input('Teclea el fichero con dataset Iris:')
#df = pd.read_csv(lugar, header=None, encoding='utf-8')
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
print('Leyendo dataset de iris')

#preparar los datos de entrenamiento de Iris
y = df.iloc[0:100, 4].values        #seleccionar ejemplos de setosa y versicolor
y = np.where(y == 'Iris-setosa', -1,1) #codificamos en +1 (Iris versicolor) y -1 (Iris-setosa)
X = df.iloc[0:100, [0, 2]].values    #extraer longitud de sépalos y pétalos

"""
#Dibujar los datos
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.ylabel('Longitud sépalo [cm]')
plt.xlabel('Longitud pétalo  [cm]')
plt.legend(loc='upper left')
plt.title(loc='center', label='Dataset Iris de Juan')
plt.show() #plt.savefig('images/02/06.png', dpi=300)
"""

#PASO 3: CREAR Y ENTRENAR UN PERCEPTRÓN
#PASO 4: ENCONTRAR UN VALOR PARA EL LEARNING RATE (0.0002 FINALMENTE)
learning_rate = 0.0002
ada1 = AdalineGDB(n_iter=20, eta=learning_rate).fit(X,y)
fig, ax = plt.subplots()
ax.plot(range(1, len(ada1.coste_)+1), np.log10(ada1.coste_), marker='o')
ax.set_yticks(np.log10(ada1.coste_))
plt.xlabel('Epocas')
plt.ylabel('Coste SSE')
plt.title('Learning rate {learning_rate}')
print("Último coste:", ada1.coste_[-1])
plt.show() #plt.savefig('images/02/06.png', dpi=300)

#PASO 5: PREPROCESAR LOS DATOS ANTES DE ENTRENAR
media_X = np.mean(X, axis=0)    #media de las columnas
desviacion_X = np.std(X, axis=0)
print("medias:", media_X)
print("desviaciones:", desviacion_X)
X_normal = (X - media_X) / desviacion_X
#--- Volver a entrenar AdalineGDB(), ajustar learning rate y graficar

learning_rate = 0.0005
ada1 = AdalineGDB(n_iter=20, eta=learning_rate)
ada1.fit(X_normal, y)  #Entrenamos con los datos normalizados

#Graficar el coste
import matplotlib.pyplot as plt
plt.plot(range(1, len(ada1.coste_) + 1), ada1.coste_, marker='o')
plt.xlabel('Época')
plt.ylabel('Coste')
plt.title('Coste vs Épocas')
plt.show()
print("Coste actualizado con normalización:", ada1.coste_[-1])

#PASO 6: Función para dibujar regiones de decisión
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
        
plot_regiones(X_normal, y, clasificador=ada1)
plt.xlabel('longitud sépalos [cm] normalizado')
plt.ylabel('longitud pétalos [cm] normalizado')
plt.legend(loc='upper left')        #plt.savefig('images/U02_P01_3.png', dpi=300)
plt.title('Regiones de decisión de Juan Martínez normalizadas')
plt.show()
