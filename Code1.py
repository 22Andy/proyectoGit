import numpy as np
import matplotlib.pyplot as plt

# Definir la función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generar un rango de valores para x
x = np.linspace(-10, 10, 100)

# Calcular los valores de y usando la función sigmoide
y = sigmoid(x)

# Crear la gráfica
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoide')
plt.show()