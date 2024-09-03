import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Cargar el conjunto de datos IRIS
iris = load_iris()
X = iris.data
species = iris.target

# Número de componentes
NCOMP = 2

# Análisis de componentes principales
pca = PCA(n_components=NCOMP)
X_pca = pca.fit_transform(X)

# Crear un DataFrame con los componentes principales y las especies
Xproy = pd.DataFrame(data=X_pca, columns=[f'Comp.{i+1}' for i in range(NCOMP)])
Xproy['Species'] = species

# Imprimir resultado
print("Reducción de dimensiones")
print(Xproy.head())

# Graficación
plt.figure(figsize=(8, 6))
plt.scatter(Xproy['Comp.1'], Xproy['Comp.2'], cmap='viridis', edgecolor='k', s=100)
plt.title('PCA of IRIS dataset')
plt.xlabel('Comp.1')
plt.ylabel('Comp.2')
plt.grid(True)
plt.show()

