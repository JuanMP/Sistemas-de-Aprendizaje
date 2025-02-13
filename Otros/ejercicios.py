import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Crear un DataFrame de ejemplo con valores ausentes
data = {
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [10, 11, 12, np.nan]
}
df = pd.DataFrame(data)

si = SimpleImputer(missing_values=np.nan, strategy='mean')
si = si.fit(df.values)
x = si.transform(df.values)
print("Datos transformados\n", x)
print("Datos originales\n", df.values)
