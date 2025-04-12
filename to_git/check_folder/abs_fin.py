import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List, Union, Optional

# Установка случайного seed для воспроизводимости
np.random.seed(42)

# =============== Модельные функции ===============

# 1D модель: синус
def model_1d(x: np.ndarray) -> np.ndarray:
    """Одномерная модельная функция: синус"""
    return np.sin(x)

def dmodel_1d(x: np.ndarray) -> np.ndarray:
    """Производная одномерной модельной функции"""
    return np.cos(x)

# 2D модель: параболоид
def model_2d(x: np.ndarray) -> np.ndarray:
    """Двумерная модельная функция: параболоид"""
    return x[0]**2 + x[1]**2

# =============== Создание датасета ===============

def create_1d_dataset(n_points: int = 100, x_min: float = -5, x_max: float = 5, 
                     noise_scale: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создаёт одномерный датасет с шумом для модели model_1d.
    
    Аргументы:
        n_points: Количество точек
        x_min, x_max: Границы для x
        noise_scale: Масштаб шума
    
    Возвращает:
        X_data: Входные значения x
        y_data: Чистые значения y
        y_noise_data: Значения y с шумом
    """
    X_data = np.linspace(x_min, x_max, n_points)
    y_data = model_1d(X_data)
    y_noise = np.random.normal(0, noise_scale, n_points)
    y_noise_data = y_data + y_noise
    return X_data, y_data, y_noise_data

def create_2d_dataset(n_points: int = 100, x_min: float = -2, x_max: float = 2, 
                      noise_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создаёт двумерный датасет с шумом для модели model_2d.
    
    Аргументы:
        n_points: Количество точек
        x_min, x_max: Границы для x1 и x2
        noise_scale: Масштаб шума
    
    Возвращает:
        U: Массив входных векторов [x1, x2]
        Y: Чистые значения y
        Y_noise: Значения y с шумом
    """
    # Генерируем случайные точки в указанном диапазоне
    U = np.random.uniform(x_min, x_max, (n_points, 2))
    
    # Вычисляем значения модели
    Y = np.zeros(n_points)
    for i in range(n_points):
        Y[i] = model_2d(U[i])
    
    # Добавляем шум
    noise = np.random.normal(0, noise_scale, n_points)
    Y_noise = Y + noise
    
    return U, Y, Y_noise

# =============== Оптимизационные функции ===============

def objective_1d(x_pred: np.ndarray, x0: float, y0: float, model_func: Callable) -> float:
    """
    Целевая функция для одномерного случая: квадрат расстояния от точки до кривой.
    
    Аргументы:
        x_pred: Предполагаемое значение x (оптимизируемая переменная)
        x0: Исходное значение x
        y0: Исходное значение y (возможно с шумом)
        model_func: Модельная функция
    
    Возвращает:
        Квадрат расстояния
    """
    # x_pred — массив длины 1, так как minimize передаёт его как вектор
    x = x_pred[0]
    if x0 - x == 0:
        x0 = x0 + 0.0001
    return np.sum((x - x0)**2) + (model_func(x) - y0)**2

def objective_nd(u_pred: np.ndarray, point: np.ndarray, model_func: Callable) -> float:
    """
    Целевая функция для многомерного случая: квадрат расстояния от точки до поверхности.
    
    Аргументы:
        u_pred: Предполагаемый вектор входных значений (оптимизируемые переменные)
        point: Исходная точка [x1, x2, ..., xn, y]
        model_func: Модельная функция
    
    Возвращает:
        Квадрат расстояния
    """
    n = len(point) - 1  # Размерность входа (без учёта y)
    u0 = point[:n]  # Исходные координаты входа
    y0 = point[n]  # Исходное значение выхода
    
    # Вычисляем квадрат расстояния: сумма квадратов разностей координат + квадрат разности выходов
    return np.sum((u0 - u_pred)**2) + (model_func(u_pred) - y0)**2

# =============== Основная функция оптимизации ===============

def find_closest_points(X_data: np.ndarray, Y_data: np.ndarray, 
                       model_func: Callable, method: str = 'Nelder-Mead', 
                       u_bounds: Optional[List] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Находит точки на поверхности/кривой модели, ближайшие к заданным точкам.
    Работает как с одномерными, так и с многомерными данными.
    
    Аргументы:
        X_data: Входные значения x (для 1D) или массив входных векторов [x1, x2, ...] (для n-D)
        Y_data: Значения y (возможно с шумом)
        model_func: Модельная функция
        method: Метод оптимизации ('Nelder-Mead' для 1D, 'L-BFGS-B' для n-D рекомендуется)
        u_bounds: Ограничения на оптимизируемые переменные (опционально)
    
    Возвращает:
        Для 1D: (X_pred, None) - только оптимизированные значения x
        Для n-D: (U_pred, optimal_points) - оптимизированные векторы и полные точки на поверхности
    """

    total_difference = 0

    # Определяем размерность входных данных
    if X_data.ndim == 1:  # Одномерный случай
        n_samples = len(X_data)
        n_dims = 1
        # Преобразуем X_data в формат [[x1], [x2], ...] для единообразия
        X_reshaped = X_data.reshape(-1, 1)
    else:  # Многомерный случай
        n_samples = X_data.shape[0]
        n_dims = X_data.shape[1]
        X_reshaped = X_data
    
    # Инициализируем массивы для результатов
    X_pred = np.zeros((n_samples, n_dims))
    optimal_points = [] if n_dims > 1 else None
    
    # Создаём полные точки [x1, x2, ..., y]
    if n_dims == 1:
        full_points = np.column_stack((X_reshaped, Y_data))
    else:
        full_points = np.column_stack((X_data, Y_data))
    
    # Автоматический подбор границ для BFGS, если они не заданы
    if u_bounds is None and method == 'L-BFGS-B' and n_dims > 1:
        u_bounds = []
        for j in range(n_dims):
            col = X_reshaped[:, j]
            delta = 0.1 * (col.max() - col.min()) if col.max() != col.min() else 1.0
            u_bounds.append((col.min() - delta, col.max() + delta))
    
    # Адаптируем функцию model_func для работы с разными размерностями
    def wrapped_model_func(u):
        if n_dims == 1:
            # Если u приходит как массив длины 1, извлекаем скаляр
            u_val = u[0] if hasattr(u, '__getitem__') else u
            return model_func(u_val)
        else:
            return model_func(u)
    
    # Для каждой точки решаем задачу оптимизации
    for i in range(n_samples):
        point = full_points[i]
        u0 = point[:n_dims]  # Исходные координаты входа
        
        # Выбираем целевую функцию в зависимости от размерности
        if n_dims == 1:
            # Для 1D случая используем специализированную целевую функцию
            objective = lambda x_pred, x0=u0[0], y0=point[n_dims]: (
                np.sum((x_pred[0] - x0)**2) + (wrapped_model_func(x_pred) - y0)**2
            )
            x0_param = [u0[0]]  # Передаем как список для 1D случая
        else:
            # Для n-D случая используем общую целевую функцию
            objective = lambda u_pred, p=point: (
                np.sum((u_pred - p[:n_dims])**2) + (wrapped_model_func(u_pred) - p[n_dims])**2
            )
            x0_param = u0  # Передаем как массив для n-D случая
        
        # Проводим оптимизацию
        res = minimize(
            objective,
            x0=x0_param,
            method=method,
            bounds=u_bounds if method == 'L-BFGS-B' and n_dims > 1 else None,
            options={'maxiter': 100}
        )
        
        if res.success:
            if n_dims == 1:
                X_pred[i, 0] = res.x[0]
            else:
                X_pred[i] = res.x
            total_difference += res.fun
        else:
            # В случае неудачи используем исходные координаты
            X_pred[i] = u0
            total_difference += 100
        
        # Для многомерного случая сохраняем полные точки на поверхности
        if n_dims > 1:
            y_pred = wrapped_model_func(X_pred[i])
            full_point = np.append(X_pred[i], y_pred)
            optimal_points.append(full_point)
    
    # Возвращаем результаты
    if n_dims == 1:
        # Для 1D возвращаем только вектор X (а не матрицу)
        return X_pred.flatten(), None, total_difference
    else:
        return X_pred, np.array(optimal_points), total_difference

# =============== Функции визуализации ===============

def visualization_plot_1d(X_data: np.ndarray, y_noise_data: np.ndarray, 
                         X_pred: np.ndarray, model_func: Callable):
    """
    Визуализирует результаты для одномерного случая.
    
    Аргументы:
        X_data: Исходные значения x
        y_noise_data: Значения y с шумом
        X_pred: Оптимизированные значения x
        model_func: Модельная функция
    """
    plt.figure(figsize=(12, 8))
    
    # Гладкая кривая модели
    x_curve = np.linspace(min(X_data), max(X_data), 1000)
    y_curve = model_func(x_curve)
    plt.plot(x_curve, y_curve, 'b-', linewidth=2, label='Model function')

    # Предсказанные значения
    y_pred = model_func(X_pred)
    
    # Шумные наблюдения
    plt.scatter(X_data, y_noise_data, color='green', s=40, alpha=0.7, label='Noisy data points')
    
    # Предсказания (точки на кривой)
    plt.scatter(X_pred, y_pred, color='red', s=40, alpha=0.7, label='Predicted points')
    
    # Соединяем отрезками оригинальные точки и их проекции на кривую
    for i in range(len(X_data)):
        plt.plot([X_data[i], X_pred[i]], [y_noise_data[i], y_pred[i]], 'r-', alpha=0.5)
    
    plt.axis('equal')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.title('Closest Points on the Curve to Noisy Observations', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualization_plot_2d(U: np.ndarray, Y_noise: np.ndarray, 
                        optimal_points: np.ndarray, model_func: Callable):
    """
    Визуализирует результаты для двумерного случая.
    
    Аргументы:
        U: Массив входных векторов [x1, x2]
        Y_noise: Значения y с шумом
        optimal_points: Точки на поверхности [x1, x2, y]
        model_func: Модельная функция
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Формируем полные исходные точки [x1, x2, y]
    data = np.column_stack((U, Y_noise))
    
    # Точки датасета с шумом
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='g', s=50, 
              label='Noisy data points', alpha=0.7)
    
    # Оптимальные точки на поверхности
    ax.scatter(optimal_points[:, 0], optimal_points[:, 1], optimal_points[:, 2], 
              color='r', s=50, label='Points on surface', alpha=0.7)
    
    # Соединяем линиями точки данных с их проекциями на поверхность
    for i in range(len(data)):
        ax.plot([data[i, 0], optimal_points[i, 0]], 
                [data[i, 1], optimal_points[i, 1]], 
                [data[i, 2], optimal_points[i, 2]], 
                'r-', alpha=0.4)
    
    # Строим поверхность функции
    u0_lin = np.linspace(data[:, 0].min()-0.5, data[:, 0].max()+0.5, 30)
    u1_lin = np.linspace(data[:, 1].min()-0.5, data[:, 1].max()+0.5, 30)
    U0, U1 = np.meshgrid(u0_lin, u1_lin)
    
    # Вычисляем значения функции для сетки
    Z = np.zeros(U0.shape)
    for i in range(U0.shape[0]):
        for j in range(U0.shape[1]):
            Z[i, j] = model_func(np.array([U0[i, j], U1[i, j]]))
    
    # Отображаем поверхность
    surf = ax.plot_surface(U0, U1, Z, alpha=0.3, color='b', edgecolor='none')
    
    # Установка равных масштабов для всех осей (аналог plt.axis('equal') для 3D)
    # Получаем границы данных
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    z_min, z_max = min(data[:, 2].min(), Z.min()) - 0.5, max(data[:, 2].max(), Z.max()) + 0.5
    
    # Вычисляем радиус сферы, охватывающей все данные
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    
    # Вычисляем центры для каждой оси
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    
    # Устанавливаем границы по каждой оси
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    # Установка равного соотношения сторон (для сохранения равных масштабов осей)
    ax.set_box_aspect([1, 1, 1])
    
    # Улучшаем внешний вид графика
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('X₂', fontsize=14)
    ax.set_zlabel('Y', fontsize=14)
    ax.set_title("Visualization of Data Points and Their Projections on Surface", fontsize=16)
    ax.legend(fontsize=12)
    
    # Добавляем направляющие сетки для лучшего восприятия 3D
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============== Основной код ===============

if __name__ == "__main__":
    # 1D пример
    print("=== 1D Example ===")
    X_data, _, y_noise_data = create_1d_dataset(n_points=50, noise_scale=0.5)
    X_pred, _, total_difference = find_closest_points(X_data, y_noise_data, model_1d, method='Nelder-Mead')
    visualization_plot_1d(X_data, y_noise_data, X_pred, model_1d)
    print(f"Total difference: {total_difference}")
    
    # 2D пример
    print("\n=== 2D Example ===")
    U, _, Y_noise = create_2d_dataset(n_points=50, noise_scale=1.5)
    U_pred, optimal_points, total_difference = find_closest_points(U, Y_noise, model_2d, method='L-BFGS-B')
    visualization_plot_2d(U, Y_noise, optimal_points, model_2d)
    print(f"Total difference: {total_difference}")
    
    print("Optimization complete. Visualizations displayed.")
