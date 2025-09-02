import math
import numpy as np
import matplotlib.pyplot as plt

pdf = lambda x: 4 * (x ** 2) * np.exp(-2 * x)
g_func = lambda x: 0.5 * np.exp(-x / 2)
C = 2
loops = 10000
number_array = []
y_pdf = [pdf(i) for i in np.linspace(0, 6, 100)]
y_g_func = [g_func(i) for i in np.linspace(0, 6, 100)]
uniform = []
i = 0
for i in range(loops):
    u = np.random.uniform(0, 1)
    y = - 2 * math.log(1 - np.random.uniform(0, 1), math.e)
    uniform.append(y)
    if C * g_func(y) * u < pdf(y):
        number_array.append(y)

plt.hist(uniform, bins=100, density=True)
plt.hist(number_array, bins=100, density=True, alpha=0.7)
plt.xlim([0, 6])
plt.plot(np.linspace(0, 6, 100), y_pdf, label='f(x)')
plt.plot(np.linspace(0, 6, 100), y_g_func, label='g(x)')
plt.legend(loc='upper right', fontsize='large')
plt.show()
