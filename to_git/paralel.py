import numpy as np
import pandas as pd
import os

def parallel(x, y): return (x*y)/(x + y)  # f6

ranges = {
    'parallel_simple': (1e-3, 10),
}

n_samples = 1000

# Функция для генерации данных для 2D функций
def sample_2d(func, name, x_range, y_range=None):
    if y_range is None:
        x = np.linspace(*x_range, int(np.sqrt(n_samples)))
        y = np.linspace(*x_range, int(np.sqrt(n_samples)))
    else:
        x = np.linspace(*x_range, int(np.sqrt(n_samples)))
        y = np.linspace(*y_range, int(np.sqrt(n_samples)))
    xx, yy = np.meshgrid(x, y)
    zz = func(xx, yy)
    # Формат: x, y, output
    return pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'output': zz.ravel()})

df = sample_2d(parallel, 'parallel_simple', ranges['parallel_simple'])
df.to_csv('datasets_bc/parallel_simple.csv', index=False)
print(f"Сохранено: datasets_bc/parallel_simple.csv")

