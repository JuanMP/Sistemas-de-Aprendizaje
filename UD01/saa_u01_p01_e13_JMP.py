import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc



#Dataset (Hay que decirle que empieza en la columna 1 y que usa delimitador ;)
datos = pd.read_csv("alumnos.csv", encoding='ISO-8859-1', header=0, sep=';')

#Convertir las columnas que pueden tener comas como separador decimal
datos['MIOPIA'] = datos['MIOPIA'].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)

#Convertir a tipo float las columnas relevantes
datos['MIOPIA'] = datos['MIOPIA'].astype(float)


#Quitar columna Nombre del dataset
datos = datos.drop(columns=['NOMBRE'])

datos.columns = datos.columns.str.strip()  # Elimina espacios al principio y al final

#Columna para clasificar target (Cafés)
datos['CAFES_CLASE'] = datos['CAFES_DIA'].apply(lambda x: 1 if x >= 3 else 0)

#Separar variables x y
x = datos.drop(columns=['CAFES_DIA', 'CAFES_CLASE'])
y = datos['CAFES_CLASE']

#Dividir en entrenamiento y prueba (20 % para prueba)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


#Modelo de regresión logística
param_grid_lr = {'C': [0.1, 1, 10]}
lr = LogisticRegression(max_iter=1000, random_state=42)
grid_lr = GridSearchCV(lr, param_grid_lr, cv=2, scoring='roc_auc')
grid_lr.fit(X_train, y_train)


    
#Modelo Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='roc_auc')
grid_rf.fit(X_train, y_train)


for model, name in zip([grid_lr, grid_rf], ['Logística de Regresión', 'Random Forest']):
    y_pred = model.best_estimator_.predict(X_test)
    y_proba = model.best_estimator_.predict_proba(X_test)[:, 1]
    print(f"Resultados para {name}")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    
    
     #Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

#Graficar todas las curvas ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC')
plt.legend()
plt.show()




#En definitiva podría añadir más datos para que los modelos puedan trabajar mejor o también aplicar diferentes
#técnicas, balanceo, probar Xgboost, Gradient y m