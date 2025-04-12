import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def orthogonal_distance_error(f, data, u_bounds=None):
    """
    Вычисляет суммарную ошибку (сумму квадратов ортогональных расстояний)
    между поверхностью, заданной скалярной функцией f, и точками датасета.
    
    Аргументы:
      - f: функция f: R^n -> R. Если n == 1, может ожидаться скалярный аргумент.
      - data: массив точек формы [(x1, x2, ..., xn, y), ...],
              где первые n координат соответствуют независимым переменным,
              а последний — зависимой переменной.
      - u_bounds: список или кортеж из n кортежей (low, high) для ограничения
                  области поиска переменной u. Если None, границы вычисляются
                  автоматически по каждому измерению.
    
    Возвращает:
      - total_error: сумма квадратов минимальных ортогональных расстояний.
      - errors: список найденных минимальных расстояний для каждой точки.
      - optimal_points: список точек на поверхности функции, ближайших к исходным точкам.
    """
    data = np.asarray(data)
    n = data.shape[1] - 1  # число независимых переменных
    total_error = 0.0
    errors = []
    optimal_points = []
    
    # Автоматический подбор границ, если они не заданы:
    if u_bounds is None:
        u_bounds = []
        for j in range(n):
            col = data[:, j]
            delta = 0.1 * (col.max() - col.min()) if col.max() != col.min() else 1.0
            u_bounds.append((col.min() - delta, col.max() + delta))
    
    # Если функция одномерная, создаём обёртку, чтобы передавать скаляр
    if n == 1:
        def f_wrapped(u):
            return f(u[0])
    else:
        f_wrapped = f
    
    # Для каждой точки решаем задачу оптимизации
    for point in data:
        u0 = point[:n]  # исходное приближение: проекция точки на R^n
        y = point[n]
        
        def objective(u):
            u = np.array(u)
            # Квадрат расстояния между (p1, ..., pn) и u плюс квадрат разницы f(u) и y
            return np.sum((point[:n] - u)**2) + (y - f_wrapped(u))**2
        
        res = minimize(objective, x0=u0, bounds=u_bounds, method='L-BFGS-B')
        
        # Можно добавить проверку res.success, если требуется
        total_error += res.fun
        errors.append(np.sqrt(res.fun))
        
        # Сохраняем оптимальную точку на поверхности
        u_optimal = res.x
        y_optimal = f_wrapped(u_optimal)
        optimal_points.append(np.append(u_optimal, y_optimal))
    
    return total_error, errors, optimal_points

def visualization_plot_3d(data, optimal_points, f_example):
    """
    Создает 3D визуализацию с соединением точек данных и их проекций на поверхность.
    
    Аргументы:
      - data: массив точек формы [(x1, x2, ..., xn, y), ...] - точки с шумом
      - optimal_points: массив точек на поверхности функции
      - f_example: функция поверхности для построения
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Точки датасета
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='g', s=50, label='Данные с шумом', alpha=0.7)
    
    # Оптимальные точки на поверхности
    optimal_points = np.array(optimal_points)
    ax.scatter(optimal_points[:, 0], optimal_points[:, 1], optimal_points[:, 2], 
               color='r', s=50, label='Точки на поверхности', alpha=0.7)
    
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
    Z = U0**2 + U1**2  # f_example(u)
    surf = ax.plot_surface(U0, U1, Z, alpha=0.3, color='b', edgecolor='none')
    
    # Улучшаем внешний вид графика
    ax.set_xlabel('u[0]', fontsize=14)
    ax.set_ylabel('u[1]', fontsize=14)
    ax.set_zlabel('f(u)', fontsize=14)
    ax.set_title("Визуализация данных и проекций на поверхность функции", fontsize=16)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Пример использования для функции двух переменных:
if __name__ == "__main__":
    # Пример функции: параболоид f(u) = u[0]**2 + u[1]**2
    def f_example(u):
        return u[0]**2 + u[1]**2

    # Генерация синтетических данных
    np.random.seed(0)
    N = 100
    # Выбираем u[0] и u[1] случайно из интервала [-2, 2]
    U = np.random.uniform(-2, 2, (N, 2))
    # Вычисляем истинное значение f(u) и добавляем шум к зависимой переменной
    # Увеличиваем шум в 3 раза по сравнению с исходным
    noise = np.random.normal(0, 1.5, N)  # Было 0.5, увеличено до 1.5
    Y = np.array([f_example(u) for u in U]) + noise
    # Формируем датасет: [u[0], u[1], y]
    data = np.hstack((U, Y.reshape(-1, 1)))
    
    total_error, point_errors, optimal_points = orthogonal_distance_error(f_example, data)
    print("Общая ошибка (сумма квадратов ортогональных расстояний):", total_error)
    
    # Первый способ визуализации: стандартный 3D график
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Точки датасета
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='g', label='Данные с шумом', alpha=0.6)
    
    # Строим поверхность функции
    u0_lin = np.linspace(data[:, 0].min()-0.5, data[:, 0].max()+0.5, 30)
    u1_lin = np.linspace(data[:, 1].min()-0.5, data[:, 1].max()+0.5, 30)
    U0, U1 = np.meshgrid(u0_lin, u1_lin)
    Z = U0**2 + U1**2  # f_example(u)
    ax.plot_surface(U0, U1, Z, alpha=0.5, color='b')
    
    ax.set_xlabel('u[0]')
    ax.set_ylabel('u[1]')
    ax.set_zlabel('f(u)')
    ax.set_title("Визуализация данных и функции (стандартный способ)")
    plt.legend()
    plt.show()
    
    # Второй способ визуализации: с соединением точек и их проекций на поверхность
    visualization_plot_3d(data, optimal_points, f_example)
