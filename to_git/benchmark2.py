import numpy as np
import pandas as pd
import os

# Определим функции ℝ→ℝ с более описательными именами
def polynomial(x): return 3.2 * x**2 - 2.7 * x + 5.3  # f1
def sin_plus_x(x): return 2.1*np.sin(1.2*x) + 0.5 * x   # f2
def rational_func(x): return (3.2*x**2 + 4.4) / (1.5*x**2 + 8.1)  # f3
# def atan_plus_x(x): return np.arctan(2.3 * x) + 1.3*x  # f4
def atan_plus_x(x): return np.arctan(2.3 * x) + 0.7*x  # f4

# ℝ²→ℝ с более описательными именами
def quad_surface(x, y): return 0.9*x**2 + 1.3*y**2 + 0.5*x*y - 1.5  # f5
def parallel(x, y): return (3.1*x*y)/(1.2*x + 0.8*y)  # f6
def power_diff(x, y): return 0.7*x**3 - 0.1*y**3  # f7
def atan_div(x, y): return 1.7*np.arctan(3.3*x / (2.5*y + 1.7))  # f8

# exponents
def sin_times_exp_2d(x, y): return np.sin(x) * np.exp(-y)  # f9
def gaussian_2d(x, y): return np.exp(-(x**2 + y**2))  # f10
def exp_times_sin_1d(x): return np.exp(-0.5*x) * np.sin(3.3*x)  # f11
# def exp_times_sin_1d(x): return np.exp(-0.5*x) * np.sin(5*x)  # f11

# Интервалы для генерации данных (оставлены без изменений)
ranges_1d = {
    'polynomial': (-5, 6),
    'sin_plus_x': (-10, 10),
    'rational_func': (-6, 6),
    'atan_plus_x': (-5, 5),
}

ranges_2d = {
    'quad_surface': (-5, 5),
    'parallel': (1e-3, 10),
    'power_diff': (-3, 3),
    'atan_div': (-6, 6),
}

ranges_exp = {
    'sin_times_exp_2d': ((-6, 6), (-3, 3)),
    'gaussian_2d': (-3, 3),
    'exp_times_sin_1d': (-5, 5),
}

# Количество сэмплов
n_samples = 1000

# Функция для генерации данных для 1D функций
def sample_1d(func, name, x_range):
    x = np.linspace(*x_range, n_samples)
    y = func(x)
    # Формат: x, output
    return pd.DataFrame({'x': x, 'output': y})

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

# Функция для сохранения датафреймов в CSV
def save_dataframes_to_csv(dataframes_dict, output_dir='datasets_bc_new'):
    # Создаем директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Сохраняем каждый датафрейм в CSV
    for name, df in dataframes_dict.items():
        file_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Сохранено: {file_path}")

# Генерация датафреймов для 1D функций
dfs_1d = {
    'polynomial': sample_1d(polynomial, 'polynomial', ranges_1d['polynomial']),
    'sin_plus_x': sample_1d(sin_plus_x, 'sin_plus_x', ranges_1d['sin_plus_x']),
    'rational_func': sample_1d(rational_func, 'rational_func', ranges_1d['rational_func']),
    'atan_plus_x':  sample_1d(atan_plus_x, 'atan_plus_x', ranges_1d['atan_plus_x']),
}

# Генерация датафреймов для 2D функций
dfs_2d = {
    'quad_surface': sample_2d(quad_surface, 'quad_surface', *([ranges_2d['quad_surface']]*2)),
    'parallel': sample_2d(parallel, 'parallel', *([ranges_2d['parallel']]*2)),
    'power_diff': sample_2d(power_diff, 'power_diff', *([ranges_2d['power_diff']]*2)),
    'atan_div': sample_2d(atan_div, 'atan_div', *([ranges_2d['atan_div']]*2)),
}

dfs_exp = {
    'sin_times_exp_2d': sample_2d(sin_times_exp_2d, 'sin_times_exp_2d', *([ranges_exp['sin_times_exp_2d'][0]]*2, [ranges_exp['sin_times_exp_2d'][1]])),
    'gaussian_2d': sample_2d(gaussian_2d, 'gaussian_2d', *([ranges_exp['gaussian_2d']]*2)),
    'exp_times_sin_1d': sample_1d(exp_times_sin_1d, 'exp_times_sin_1d', ranges_exp['exp_times_sin_1d']),
}

# Объединение всех датафреймов (только для отображения)
all_dfs = {}
all_dfs.update(dfs_1d)
all_dfs.update(dfs_2d)
all_dfs.update(dfs_exp)

# Вывод примеров данных для проверки
for name, df in all_dfs.items():
    print(f"\n{name}:")
    print(df.head())

# Сохранение датафреймов в CSV файлы
save_dataframes_to_csv(all_dfs, output_dir='datasets_bc_new')

print("\nВсе датасеты успешно созданы и сохранены в директорию datasets_bc_new/")
