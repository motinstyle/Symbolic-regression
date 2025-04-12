import numpy as np
import time
import torch 
import matplotlib.pyplot as plt
from typing import Union
from scipy import optimize as opt

MU = 0.0
SIGMA = 0.5

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def check_speed(func, *args, **kwargs):
    # print(f"Args: {args}")
    # print(f"Kwargs: {kwargs}")
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time

def create_random_points(num_of_sets, num_of_points, num_of_features):
    return SIGMA*np.random.normal(MU, SIGMA, (num_of_sets, num_of_points, num_of_features))

def create_model_dataset(num_of_sets, num_of_points, num_of_features):
    """
    Creates a dataset with noisy model outputs.
    
    Args:
        num_of_sets: Number of sets in the dataset
        num_of_points: Number of points in each set
        num_of_features: Total features (input_dim + 1 for output)
    
    Returns:
        np.ndarray with shape (num_of_sets, num_of_points, num_of_features)
        containing [X, noisy_model(X)] pairs
    """
    # Generate random input points
    input_dim = num_of_features - 1
    X_STD = 5
    X = np.random.normal(MU, X_STD, (num_of_sets, num_of_points, input_dim))
    # print(X)
    
    # Apply model function to get clean outputs
    model = lambda x: 3*np.exp(-np.sum(np.square(x+5), axis=-1))
    Y = model(X)
    
    # Add noise to outputs and reshape to match input format
    Y_noise = Y[..., np.newaxis] + np.random.normal(MU, SIGMA, (num_of_sets, num_of_points, 1))
    
    # Combine inputs and noisy outputs
    data = np.concatenate([X, Y_noise], axis=-1)
    
    return data

def visualisation_data(data: np.ndarray):
    plt.figure(figsize=(12, 8))
    plt.scatter(data[:, 0], data[:, 1], color='green', s=40, alpha=0.7, label='Noisy data points')
    plt.axis('equal')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.title('dataset', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualization_plot_1d(X_data: np.ndarray, y_noise_data: np.ndarray, 
                         X_pred: np.ndarray, model_func: callable, inv_flag: bool = False):
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
    print(min(y_pred), max(y_pred))
    
    # Шумные наблюдения
    plt.scatter(X_data, y_noise_data, color='green', s=40, alpha=0.7, label='Noisy data points')
    
    # Предсказания (точки на кривой)
    plt.scatter(X_pred, y_pred, color='red', s=40, alpha=0.7, label='Predicted points')
    
    if inv_flag:
        indices = np.where(~((y_noise_data > max(y_pred)) | (y_noise_data < min(y_pred))))[0]
        print(indices)

        # Соединяем отрезками оригинальные точки и их проекции на кривую
        for i in indices:
            plt.plot([X_data[i], X_pred[i]], [y_noise_data[i], y_pred[i]], 'r-', alpha=0.5)
    else:
        indices = np.arange(len(X_data))
        print(indices)
        # Соединяем отрезками оригинальные точки и их проекции на кривую
        for i in indices:
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
                        optimal_points: np.ndarray, model_func: callable):
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

def model_1d(x):
    """
    Модельная функция для одномерных данных: sin(0.5*x) + cos(x)
    Корректно обрабатывает как torch.Tensor, так и np.ndarray разной размерности
    """
    if isinstance(x, torch.Tensor):
        # Обработка PyTorch тензора
        if x.dim() == 0:  # скаляр
            return torch.sin(0.5*x) + torch.cos(x)
        elif x.dim() == 1:  # вектор [x1, x2, ...]
            if x.size(0) == 1:  # один элемент
                return torch.sin(0.5*x[0]) + torch.cos(x[0])
            else:  # вектор элементов
                return torch.sin(0.5*x) + torch.cos(x)
        elif x.dim() == 2:  # батч [batch_size, features]
            if x.size(1) == 1:  # один признак
                return torch.sin(0.5*x[:, 0]) + torch.cos(x[:, 0])
            else:  # используем только первый признак
                return torch.sin(0.5*x[:, 0]) + torch.cos(x[:, 0])
        else:
            raise ValueError(f"Неподдерживаемая размерность тензора: {x.dim()}")
    else:
        # Обработка NumPy массива
        x = np.asarray(x)
        if x.ndim == 0:  # скаляр
            return np.sin(0.5*x) + np.cos(x)
        elif x.ndim == 1:  # вектор [x1, x2, ...]
            if x.size == 1:  # один элемент
                return np.sin(0.5*x[0]) + np.cos(x[0])
            else:  # вектор элементов
                return np.sin(0.5*x) + np.cos(x)
        elif x.ndim == 2:  # батч [batch_size, features]
            if x.shape[1] == 1:  # один признак
                return np.sin(0.5*x[:, 0]) + np.cos(x[:, 0])
            else:  # используем только первый признак
                return np.sin(0.5*x[:, 0]) + np.cos(x[:, 0])
        else:
            raise ValueError(f"Неподдерживаемая размерность массива: {x.ndim}")

def model_1d_2(x):
    """
    Модельная функция для одномерных данных: sin(0.5*x) + cos(x)
    Корректно обрабатывает как torch.Tensor, так и np.ndarray разной размерности
    """

    MAGNITUDE = 3.0

    if isinstance(x, torch.Tensor):
        # Обработка PyTorch тензора
        if x.dim() == 0:  # скаляр
            return MAGNITUDE*torch.exp(-1*torch.square(x))
        elif x.dim() == 1:  # вектор [x1, x2, ...]
            if x.size(0) == 1:  # один элемент
                return MAGNITUDE*torch.exp(-1*torch.square(x[0]))
            else:  # вектор элементов
                return MAGNITUDE*torch.exp(-1*torch.square(x))
        elif x.dim() == 2:  # батч [batch_size, features]
            if x.size(1) == 1:  # один признак
                return MAGNITUDE*torch.exp(-1*torch.square(x[:, 0]))
            else:  # используем только первый признак
                return MAGNITUDE*torch.exp(-1*torch.square(x[:, 0]))
        else:
            raise ValueError(f"Неподдерживаемая размерность тензора: {x.dim()}")
    else:
        # Обработка NumPy массива
        x = np.asarray(x)
        if x.ndim == 0:  # скаляр
            return MAGNITUDE*np.exp(-1*np.square(x))
        elif x.ndim == 1:  # вектор [x1, x2, ...]
            if x.size == 1:  # один элемент
                return MAGNITUDE*np.exp(-1*np.square(x[0]))
            else:  # вектор элементов
                return MAGNITUDE*np.exp(-1*np.square(x))
        elif x.ndim == 2:  # батч [batch_size, features]
            if x.shape[1] == 1:  # один признак
                return MAGNITUDE*np.exp(-1*np.square(x[:, 0]))
            else:  # используем только первый признак
                return MAGNITUDE*np.exp(-1*np.square(x[:, 0]))
        else:
            raise ValueError(f"Неподдерживаемая размерность массива: {x.ndim}")

def model_2d(x):
    """
    Модельная функция для двумерных данных: x[0]^2 + x[1]^2
    Корректно обрабатывает как torch.Tensor, так и np.ndarray разной размерности
    """
    if isinstance(x, torch.Tensor):
        # Обработка PyTorch тензора
        if x.dim() == 0:  # скаляр
            return x**2
        elif x.dim() == 1:  # вектор [x0, x1]
            if x.size(0) >= 2:
                return x[0]**2 + x[1]**2
            else:
                return x[0]**2
        elif x.dim() == 2:  # батч [batch_size, features]
            return x[:, 0]**2 + x[:, 1]**2
        else:
            raise ValueError(f"Неподдерживаемая размерность тензора: {x.dim()}")
    else:
        # Обработка NumPy массива
        x = np.asarray(x)
        if x.ndim == 0:  # скаляр
            return x**2
        elif x.ndim == 1:  # вектор [x0, x1]
            if x.size >= 2:
                return x[0]**2 + x[1]**2
            else:
                return x[0]**2
        elif x.ndim == 2:  # батч [batch_size, features]
            return x[:, 0]**2 + x[:, 1]**2
        else:
            raise ValueError(f"Неподдерживаемая размерность массива: {x.ndim}")

# print(check_speed(create_random_points, 10, 1000, 2))

def eval_inv_error_scipy(data: Union[np.ndarray, torch.Tensor], model: callable) -> float:
    """
    Evaluate inverse error using scipy.optimize.minimize.
    For each point in the dataset, tries to find x such that f(x) = y.
    """
    def wrapped_model(x, target_y):
        # Преобразуем входной вектор в нужный формат в зависимости от размерности
        if len(x.shape) == 0 or (isinstance(x, np.ndarray) and x.size == 1):
            # Для одномерной модели передаем скаляр
            model_input = x
        else:
            # Для многомерной модели преобразуем в подходящую форму
            if len(x.shape) == 1:
                model_input = x.reshape(1, -1)
            else:
                model_input = x
        
        # Вычисляем предсказание модели
        y_pred = model(model_input)
        
        # Возвращаем MSE
        return np.mean((y_pred - target_y) ** 2)

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # print("data shape in scipy:", data.shape)

    data_limits = data.max(axis=0) - data.min(axis=0)
    # print("data_limits:", data_limits)
    max_error = np.sqrt(np.sum(data_limits[:-1]**2)) # max error possible for this dataset on input space
    tolerance = 0.01*np.abs(data_limits[-1])

    # Convert data to float64
    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(np.float64)

    total_inv_error = 0.0
    
    for i in range(len(y)):
        # Use current X as initial guess
        x0 = X[i].copy()  # Make sure we have a contiguous array
        target_y = float(y[i])  # Convert to float64

        # Find x that minimizes (f(x) - y)^2
        try:
            result = opt.minimize(
                wrapped_model,
                x0=x0,  # Передаем x0 в фактическом формате без reshape
                args=(target_y,),
                method='Nelder-Mead',
                options={'maxiter': 50, 'disp': False}
            )

            if result.success:
                # Compute error between found x and original x
                x_found = result.x
                if x_found.shape != x0.shape:
                    x_found = x_found.reshape(x0.shape)
                
                if np.isclose(x_found, x0, atol=1e-6).all():
                    # Проверяем, что модель даёт правильный результат
                    # model_output = model(x_found.reshape(1, -1) if len(x_found.shape) == 1 else x_found)
                    # if np.isclose(model_output, target_y, atol=tolerance):
                    #     error = 0.0
                    # else:
                    #     error = max_error
                    #     # error = np.sqrt(np.sum((x_found - x0) ** 2) + (model_output - target_y) ** 2)
                    if result.fun < tolerance:
                        error = 0.0
                    else:
                        error = max_error
                else:
                    error = np.sqrt(np.sum((x_found - x0) ** 2))
                total_inv_error += error
            else:
                # Penalize failed inversions with a reasonable penalty
                total_inv_error += max_error
        except Exception as e:
            # Print error but continue with other samples
            print(f"Optimization failed: {str(e)}")
            total_inv_error += max_error  # Penalize errors
            continue

    # Return average inverse error
    return total_inv_error / len(y)

def eval_inv_error_torch(data: torch.Tensor, model: callable, steps=100, lr=1e-1) -> tuple:
    """
    Evaluate inverse error using PyTorch optimization (batch gradient descent).
    Penalizes samples where optimization can't find x such that f(x) ≈ y.
    
    Returns:
        tuple: (error, X_optim, y_final) - среднее значение ошибки, оптимизированные входы, выходы модели
    """

    # Определяем устройство - используем CUDA, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Подготавливаем данные - переносим на нужное устройство
    data = data.clone().detach().requires_grad_(True).to(device)

    # Разделяем входные и выходные данные
    X_orig = data[:, :-1].clone().detach()
    y_target = data[:, -1].clone().detach()

    # Определяем допустимую ошибку и максимальную ошибку
    data_limits = X_orig.max(dim=0).values - X_orig.min(dim=0).values
    tolerance = 0.01*data_limits[-1]
    max_error = torch.sqrt(torch.sum(data_limits[:-1]**2)).item() # max error possible for this dataset on input space

    y_pred = model(X_orig)

    # Создаем тензор для оптимизации с градиентами
    mask_in_range = ((y_target >= min(y_pred)) & (y_target <= max(y_pred)))
    X_optim = X_orig[mask_in_range]
    X_optim = X_optim.clone().detach().requires_grad_(True)
    # X_optim = X_orig[mask_in_range].clone().detach().requires_grad_(True)
    # X_optim = X_orig[mask_in_range].clone().detach().re
    y_target = y_target[mask_in_range]

    # Создаем оптимизатор
    # optimizer = torch.optim.SGD([X_optim], lr=lr)
    optimizer = torch.optim.Adam([X_optim], lr=lr)

    # Выполняем оптимизацию
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Вычисляем предсказания модели
        y_pred = model(X_optim)
        
        # Убеждаемся, что выходы на том же устройстве, что и целевые значения
        if y_pred.device != y_target.device:
            y_pred = y_pred.to(device)
            
        # Вычисляем функцию потерь

        loss = torch.mean((y_pred - y_target) ** 2)
        # loss = torch.mean(torch.abs(y_pred - y_target))
        # loss = torch.mean((y_pred - y_target) ** 2) - 0.1*torch.mean(torch.abs(X_optim - X_orig[mask_in_range]))
        
        # Вычисляем градиенты и делаем шаг оптимизации
        loss.backward()
        optimizer.step()
        # print(X_optim[0])

    # Считаем финальное f(x') после оптимизации
    with torch.no_grad():
        # Убедимся, что все тензоры на одном устройстве перед вычислениями
        y_final = model(X_optim)
        if y_final.device != device:
            y_final = y_final.to(device)
            
        diff = torch.abs(y_final - y_target)

        # # Значения, где не удалось приблизиться к y_target
        # if max_error is None:
        #     # Используем нормализованный penalty по размеру пространства
        #     data_range = torch.max(X_orig, dim=0).values - torch.min(X_orig, dim=0).values
        #     max_error = torch.sum(data_range[:-1]).item()

        # Определяем маски для успешных и неудачных оптимизаций
        failed_mask = diff > tolerance
        success_mask = ~failed_mask

        # Ошибка = либо расстояние в пространстве x, либо penalty
        spatial_error = torch.norm(X_optim - X_orig[mask_in_range], dim=1)
        error = torch.where(success_mask, spatial_error, torch.full_like(spatial_error, max_error, device=device))
        
        # # Выводим статистику об успешности оптимизации
        # total = len(y_target)
        # successful = success_mask.sum().item()
        # failed = failed_mask.sum().item()
        # print(f"Успешных оптимизаций: {successful}/{total} ({successful/total*100:.1f}%)")
        # print(f"Неудачных оптимизаций: {failed}/{total} ({failed/total*100:.1f}%)")
        # print(error)
        
        X_orig[mask_in_range] = X_optim

        # Возвращаем среднюю ошибку, оптимизированные входы и их выходы
        return error.mean().item(), X_orig.detach().cpu().numpy()


def eval_inv_error_first(data: torch.Tensor, model: callable, steps=100, lr=1e-1, max_error=None) -> float:
    """
    Вычисляет обратную ошибку (inverse error) по следующей схеме:
      1. Вычисляется y_pred = model(X_data)
      2. Определяются min_pred и max_pred.
      3. Для тех примеров, где y_data вне диапазона [min_pred, max_pred], ошибка задается как max_error.
      4. Для остальных примеров:
           - Берётся начальное приближение x_init = get_closest_x(y_data, unique_y, mean_x)
           - Решается обратная задача (solve) для получения x_pred, при котором model(x_pred) ≈ y_data.
           - Считается ошибка как норма разности между x_data и x_pred.
      5. Возвращается среднее значение ошибки по всем примерам.
    
    Аргументы:
        data: torch.Tensor с X_data и y_data
        model: вызываемая функция модели
        steps: число шагов оптимизации для функции solve
        lr: скорость обучения в solve
        max_error: штрафная ошибка для примеров вне диапазона; если None – вычисляется из диапазона X_data
    Возвращает:
        Среднее значение обратной ошибки.
    """

    def compressed_by_y(data: torch.Tensor):
        """
        Группирует данные по уникальным значениям y.
        Для каждого уникального y возвращает все x, соответствующие этому y.
        
        Возвращает:
            comp_dict: словарь, где ключ - значение y (float),
                       а значение - тензор всех x, для которых y равно этому ключу.
        """
        comp_dict = {}
        unique_y, inverse_indices = torch.unique(data[:, -1], sorted=True, return_inverse=True)
        # print("unique_y:", unique_y, "inverse_indices:", inverse_indices)
        for i, y_val in enumerate(unique_y):
            comp_dict[y_val.item()] = data[inverse_indices == i][:, :-1]
            # print(f"comp_dict[{y_val.item()}]:", comp_dict[y_val.item()])
        return comp_dict
    
    def get_closest_x(target_y: torch.Tensor, x_original: torch.Tensor, comp_dict: dict) -> torch.Tensor:
        """
        Для каждого значения из target_y ищет ближайший ключ (уникальное y) в comp_dict 
        и затем среди всех x, соответствующих этому ключу, выбирает тот, который минимизирует норму разности
        с соответствующим x_original.
        
        Аргументы:
            target_y: тензор формы (N,)
            x_original: тензор исходных x, соответствующих target_y, формы (N, d)
            comp_dict: словарь, где ключи — уникальные y (float), а значения — тензоры x (n_candidates, d)
            
        Возвращает:
            Тензор начальных приближений x формы (N, d)
        """
        # Преобразуем ключи словаря в тензор для векторизации
        unique_keys = torch.tensor(list(comp_dict.keys()), dtype=target_y.dtype, device=target_y.device)  # (K,)
        chosen_x = []
        for i in range(target_y.shape[0]):
            # Находим ключ, ближайший к target_y[i]
            diff_keys = torch.abs(target_y[i] - unique_keys)
            best_key_idx = torch.argmin(diff_keys).item()
            print(f"best_key_idx: {best_key_idx}, target_y[i]: {target_y[i]}, unique_keys.shape: {unique_keys.shape}")
            best_key = unique_keys[best_key_idx].item()
            print(f"best_key: {best_key}")
            # Из всех кандидатов для найденного ключа выбираем тот, что ближе всего к x_original[i]
            candidates = comp_dict[best_key]  # (n_candidates, d)
            print(f"candidates: {candidates}")
            differences = torch.norm(candidates - x_original[i], dim=1)
            best_candidate_idx = torch.argmin(differences).item()
            print(f"best_candidate_idx: {best_candidate_idx}, candidates[best_candidate_idx]: {candidates[best_candidate_idx]}")
            chosen_x.append(candidates[best_candidate_idx])
        return torch.stack(chosen_x, dim=0)
    
    def solve(model: callable, target_y: torch.Tensor, x_init: torch.Tensor, steps=100, lr=1e-1) -> torch.Tensor:
        """
        Оптимизирует входной тензор x_init так, чтобы model(x) ≈ target_y.
        Используется оптимизатор Adam.
        
        Аргументы:
            model: вызываемая функция, принимающая x и возвращающая y_pred
            target_y: целевые значения y, форма (N,)
            x_init: начальное приближение x, форма (N, d)
            steps: число шагов оптимизации
            lr: скорость обучения
        Возвращает:
            x_opt: оптимизированные входы, тензор формы (N, d)
        """
        x_opt = x_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_opt], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            y_pred = model(x_opt)
            loss = torch.mean((y_pred - target_y) ** 2)
            loss.backward()
            optimizer.step()
        return x_opt.detach()

    # Определяем устройство - используем CUDA, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X = data[:, :-1].to(device)
    y = data[:, -1].to(device)

    # Вычисляем предсказание модели для всех X
    y_pred = model(X)

    # print("X.shape:", X.shape, "y_pred.shape:", y_pred.shape)

    pred_data = torch.cat((X, y_pred.unsqueeze(-1)), dim=1)
    sorted_indices = torch.argsort(pred_data[:, -1], descending=True)
    pred_data_sorted = pred_data[sorted_indices]
    pred_data_sorted_compressed = compressed_by_y(pred_data_sorted)

    max_pred = pred_data_sorted[0, -1]
    min_pred = pred_data_sorted[-1, -1]
    
    # Если max_error не задан, вычисляем его из диапазона X
    if max_error is None:
        data_limits = X.max(dim=0).values - X.min(dim=0).values
        max_error = torch.sqrt(torch.sum(data_limits ** 2)).item()
    
    # Маска для значений y, попадающих в диапазон предсказаний
    mask_in_range = (y >= min_pred) & (y <= max_pred)
    mask_out_range = ~mask_in_range
    
    total_inv_error = 0.0
    
    # Для примеров вне диапазона сразу задаём штраф
    error_out = mask_out_range.sum().item() * max_error
    
    x_pred_init = torch.zeros_like(X)

    # Для примеров внутри диапазона решаем обратную задачу
    if mask_in_range.sum() > 0:
        y_in = y[mask_in_range]
        X_in = X[mask_in_range]
        
        # Получаем сжатое представление данных: для каждого уникального y – среднее x
        unique_y = compressed_by_y(pred_data_sorted)
        # Начальное приближение для каждого y_in
        x_init = get_closest_x(y_in, X_in, unique_y)
        # Оптимизируем начальное приближение так, чтобы model(x) ≈ y_in
        x_pred = solve(model, y_in, x_init, steps=steps, lr=lr)
        # Ошибка – евклидова норма разности между исходным x и оптимизированным x_pred
        errors = torch.norm(X_in - x_pred, dim=1)
        total_inv_error += errors.sum().item()
    
    total_samples = y.numel()
    avg_error = (total_inv_error + error_out) / total_samples

    x_pred_init[mask_in_range] = x_pred

    return avg_error, x_pred_init.detach().cpu().numpy()

def eval_abs_inv_error_scipy(data: Union[np.ndarray, torch.Tensor], model: callable) -> tuple:
        """
        Вычисляет абсолютную обратную ошибку (RMSE) между исходными и предсказанными входами.
        Использует ортогональные расстояния от точек до поверхности модели.
        
        Args:
            data: Входные данные формата [X, y], где X - входные переменные, y - выходное значение
            
        Returns:
            tuple: (rmse, optimal_points) - RMSE и найденные точки [X_pred, y_pred]
        """
        # Преобразуем в numpy массив, если передан torch.Tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
            
        # Разделяем входные и выходные значения
        X = data[:, :-1].astype(np.float64)
        y = data[:, -1].astype(np.float64)
        
        # Вычисляем предельные значения для ошибок
        data_limits = data.max(axis=0) - data.min(axis=0)
        max_error = np.sqrt(np.sum(data_limits[:-1]**2))  # Максимальная ошибка для случаев неудачной оптимизации
        
        total_sq_error = 0.0
        num_points = len(y)
        
        # Для отладки: количество успешных и неудачных оптимизаций
        successful_optimizations = 0
        failed_optimizations = 0
        
        # Массив для хранения найденных точек
        X_pred = np.zeros_like(X)
        
        for i in range(num_points):
            # Используем текущий X как начальное приближение
            x0 = X[i].copy()
            target_y = float(y[i])
            
            # Определяем функцию для оптимизации, которая учитывает оба компонента ошибки:
            # 1. Квадрат расстояния между входами
            # 2. Квадрат разности выходов
            def objective(u_pred):
                # Преобразуем входной вектор в правильный формат для forward
                if np.isscalar(u_pred) or (isinstance(u_pred, np.ndarray) and u_pred.size == 1):
                    u_input = np.array([u_pred]).reshape(1, -1)
                else:
                    u_input = u_pred.reshape(1, -1)
                
                # Вычисляем выход модели
                output = model(u_input)
                
                # Преобразуем выход к скаляру
                if isinstance(output, np.ndarray):
                    if output.size > 1:
                        output = output.flatten()[0]
                    else:
                        output = output.item() if hasattr(output, 'item') else output.flatten()[0]
                elif hasattr(output, 'detach'):  # torch tensor
                    output_np = output.detach().numpy()
                    if output_np.size > 1:
                        output = output_np.flatten()[0]
                    else:
                        output = output_np.item() if hasattr(output_np, 'item') else output_np.flatten()[0]
                
                # Вычисляем совокупную ошибку
                return np.sum((u_pred - x0)**2) + (output - target_y)**2
            
            try:
                # Оптимизация: ищем точку на поверхности, ближайшую к текущей точке
                result = opt.minimize(
                    objective,
                    x0=x0,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'disp': False}
                )
                
                if result.success:
                    successful_optimizations += 1
                    # Извлекаем найденную точку
                    x_found = result.x
                    X_pred[i] = x_found
                    # Вычисляем квадрат расстояния между исходной и найденной точками входа
                    total_sq_error += result.fun

                else:
                    failed_optimizations += 1
                    # Штрафуем неудачные оптимизации
                    total_sq_error += max_error**2
                    # Выводим информацию о неудачной оптимизации
                    if i < 5:
                        print(f"Точка {i}: Оптимизация не удалась. Статус: {result.message}")
            except Exception as e:
                failed_optimizations += 1
                # В случае ошибки продолжаем с другими точками
                print(f"Точка {i}: Оптимизация вызвала ошибку: {str(e)}")
                total_sq_error += max_error**2
                continue
        
        # Выводим статистику оптимизации
        # print(f"Успешных оптимизаций (abs scipy): {successful_optimizations}/{num_points} ({successful_optimizations/num_points*100:.1f}%)")
        # print(f"Неудачных оптимизаций (abs scipy): {failed_optimizations}/{num_points} ({failed_optimizations/num_points*100:.1f}%)")
                
        # Возвращаем среднеквадратичную ошибку (RMSE) и найденные точки
        rmse = np.sqrt(total_sq_error / num_points)
        # print(f"Абсолютная обратная ошибка (RMSE) (abs scipy): {rmse:.6f}")
        return rmse, X_pred

def eval_abs_inv_error_torch(data: Union[np.ndarray, torch.Tensor], model: callable, steps=100, lr=1e-1) -> float:
    """
    Векторизованная версия абсолютной обратной ошибки (RMSE).
    Использует PyTorch для оптимизации. Штрафует несошедшиеся точки.

    Args:
        data: torch.Tensor или np.ndarray формы [N, D+1] — входы и выход.
        steps: Количество итераций оптимизации.
        lr: Скорость обучения.
        tol: Допуск к f(x') ≈ y.

    Returns:
        float: RMSE между [x, f(x)] и восстановленным [x']
    """
    # Определяем устройство - используем CUDA, если доступно

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Подготавливаем данные - переносим на нужное устройство
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32).to(device)
    else:
        data = data.to(device=device, dtype=torch.float32)

    # Разделяем входные и выходные данные
    X_orig = data[:, :-1]
    y_target = data[:, -1]

    data_limits = X_orig.max(dim=0).values - X_orig.min(dim=0).values
    max_error = torch.sqrt(torch.sum(data_limits[:-1]**2)).item() # max error possible for this dataset on input space

    # Создаем тензор для оптимизации с градиентами
    X_optim = X_orig.clone().detach().requires_grad_(True)

    # Создаем оптимизатор
    # optimizer = torch.optim.SGD([X_optim], lr=lr)
    optimizer = torch.optim.Adam([X_optim], lr=lr)

    # Выполняем оптимизацию
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Вычисляем предсказания модели
        y_pred = model(X_optim.to("cpu"))
        
        # Убеждаемся, что выходы на том же устройстве, что и целевые значения
        if y_pred.device != y_target.device:
            y_pred = y_pred.to(device)
            
        # Вычисляем функцию потерь
        # Compute squared L2 norm of concatenated (x,y) vectors for each point in batch
        # alpha = 1e-5
        loss = torch.mean(torch.sum((X_optim - X_orig)**2, dim=1) + (y_pred - y_target)**2)

        # delta_x = 0.01*data_limits[:-1]
        # diff = torch.cat((X_orig - X_optim, y_target - y_pred.reshape(-1,1)), dim=1)
        
        # # Get gradient of y_pred with respect to X_optim
        # y_pred.backward(gradient=torch.ones_like(X_optim[:, 0]), create_graph=True)
        
        # # Calculate loss
        # norm_diff = torch.sum(diff**2)
        # grad_term = torch.cat((delta_x, torch.sum(grad_y * delta_x, dim=1).reshape(-1,1)), dim=1)
        # loss = norm_diff + alpha * torch.sum(diff * grad_term)
        # loss = torch.mean(torch.sum((X_optim - X_orig)**2, dim=1) + (y_pred - y_target)**2) + alpha*torch.sum(torch.abs(X_optim))
        
        # Вычисляем градиенты и делаем шаг оптимизации
        loss.backward()
        optimizer.step()

    # Вычисляем финальное f(x') после оптимизации
    with torch.no_grad():
        # Убедимся, что все тензоры на одном устройстве перед вычислениями
        y_final = model(X_optim.to("cpu"))
        if y_final.device != device:
            y_final = y_final.to(device)
            
        # Вычисляем абсолютную разницу в выходах
        # output_diff = torch.abs(y_final - y_target) # неправильно

        # Автоматический penalty, если не задан
        if max_error is None:
            data_range = torch.max(X_orig, dim=0).values - torch.min(X_orig, dim=0).values
            max_error = torch.sum(data_range).item()

        # Определяем маски для успешных и неудачных оптимизаций
        # tolerance = 0.01*torch.sum(data_range).item()
        # success_mask = output_diff <= tolerance
        # fail_mask = ~success_mask

        # Для успешных: sqrt(‖x' - x‖² + (f(x') - y)²)
        spatial_diff_sq = torch.sum((X_optim - X_orig) ** 2, dim=1)
        output_diff_sq = (y_final - y_target) ** 2
        rmse_per_point = torch.sqrt(spatial_diff_sq + output_diff_sq)
        rmse_per_point = torch.where(torch.isnan(rmse_per_point), torch.full_like(rmse_per_point, max_error, device=device), rmse_per_point)
        # rmse_per_point = torch.where(output_diff <= tolerance, rmse_per_point, torch.full_like(rmse_per_point, penalty, device=device))
        # print(rmse_per_point)

        # Назначаем ошибку-пенальти там, где не сошлось
        # rmse_per_point = torch.where(success_mask, rmse_per_point, torch.full_like(rmse_per_point, penalty, device=device))

        # # Вывод статистики
        # total = len(y_target)
        # successful = success_mask.sum().item()
        # failed = fail_mask.sum().item()
        # print(f"Успешных оптимизаций (abs torch): {successful}/{total} ({successful / total * 100:.1f}%)")
        # print(f"Неудачных оптимизаций (abs torch): {failed}/{total} ({failed / total * 100:.1f}%)")

        # Вычисляем общую RMSE и возвращаем
        rmse = torch.sqrt(torch.mean(rmse_per_point ** 2)).item()
        # print(f"Абсолютная обратная ошибка (RMSE) (abs torch): {rmse:.6f}")
        return rmse, X_optim.detach().cpu().numpy()





def main():

    num_of_sets = 100 # 100
    num_of_points = 100# 1000
    num_of_features_1d = 1+1 # +1 is for output
    num_of_features_2d = 2+1 # +1 is for output

    # print("Evaluating 1d model...")
    # data = create_random_points(num_of_sets=num_of_sets, 
    #                             num_of_points=num_of_points, 
    #                             num_of_features=num_of_features_1d)
    # data_torch = torch.tensor(data, dtype=torch.float32)
    # print("data created with shape:", data.shape)
    # print("data_torch created with shape:", data_torch.shape)
    # # scipy_time_inv = 0
    # torch_time_inv = 0
    # # torch_time_inv_first = 0
    # # scipy_time_abs = 0
    # # torch_time_abs = 0
    # for i in range(len(data)):
    #     # scipy_time_inv += check_speed(eval_inv_error_scipy, data[i], model_1d)
    #     torch_time_inv += check_speed(eval_inv_error_torch, data_torch[i], model_1d)
    #     # torch_time_inv_first += check_speed(eval_inv_error_first, data_torch[i], model_1d)
    #     # scipy_time_abs += check_speed(eval_abs_inv_error_scipy, data[i], model_1d)
    #     # torch_time_abs += check_speed(eval_abs_inv_error_torch, data_torch[i], model_1d)
    # print("-"*10, "RESULTS 1D", "-"*10)
    # # print("scipy time inv:", scipy_time_inv)
    # print("torch time inv:", torch_time_inv)
    # # print("torch time inv first:", torch_time_inv_first)
    # # print("scipy time abs:", scipy_time_abs)
    # # print("torch time abs:", torch_time_abs)

    # print("\nEvaluating 2d model...")
    # data = create_random_points(num_of_sets=num_of_sets, 
    #                             num_of_points=num_of_points, 
    #                             num_of_features=num_of_features_2d)
    # data_torch = torch.tensor(data, dtype=torch.float32)
    # print("data created with shape:", data.shape)
    # print("data_torch created with shape:", data_torch.shape)
    # # scipy_time_inv = 0  
    # torch_time_inv = 0
    # # torch_time_inv_first = 0
    # # scipy_time_abs = 0
    # # torch_time_abs = 0
    # for i in range(len(data)):
    #     # scipy_time_inv += check_speed(eval_inv_error_scipy, data[i], model_2d)
    #     torch_time_inv += check_speed(eval_inv_error_torch, data_torch[i], model_2d)
    #     # torch_time_inv_first += check_speed(eval_inv_error_first, data_torch[i], model_2d)
    #     # scipy_time_abs += check_speed(eval_abs_inv_error_scipy, data[i], model_2d)
    #     # torch_time_abs += check_speed(eval_abs_inv_error_torch, data_torch[i], model_2d)
    # print("-"*10, "RESULTS 2D", "-"*10)
    # # print("scipy time inv:", scipy_time_inv)
    # print("torch time inv:", torch_time_inv)
    # # print("torch time inv first:", torch_time_inv_first)
    # # print("scipy time abs:", scipy_time_abs)
    # # print("torch time abs:", torch_time_abs)


    # # data = create_random_points(num_of_sets=1, 
    # #                             num_of_points=100, 
    # #                             num_of_features=num_of_features_1d)
    # # data_torch = torch.tensor(data, dtype=torch.float32)
    # # print("data created with shape:", data.shape)
    # # print("data_torch created with shape:", data_torch.shape)

    # # # rmse_scipy, X_pred_scipy = eval_abs_inv_error_scipy(data[0], model_1d)
    # # # rmse_torch, X_pred_torch = eval_abs_inv_error_torch(data_torch[0], model_1d)
    # # rmse_torch, X_pred_torch = eval_inv_error_torch(data_torch[0], model_1d)
    # # rmse_torch_first, X_pred_torch_first = eval_inv_error_first(data_torch[0], model_1d)

    # # print(f"X_pred_torch.shape: {X_pred_torch.shape}")
    # # print(f"X_pred_torch_first.shape: {X_pred_torch_first.shape}")

    # # print(f"RMSE torch: {rmse_torch}")
    # # print(f"RMSE torch first: {rmse_torch_first}")
    # # # print(f"RMSE scipy: {rmse_scipy}")
    
    # # X_data = data[0][:, :-1]
    # # y_data = data[0][:, -1]
    
    # # visualization_plot_1d(X_data, y_data, X_pred_torch, model_1d)
    # # visualization_plot_1d(X_data, y_data, X_pred_torch_first, model_1d)
    # # visualization_plot_1d(X_data, y_data, X_pred_scipy, model_1d)

    data = create_model_dataset(num_of_sets=1, num_of_points=100, num_of_features=num_of_features_1d)
    # print(data.shape)
    # visualisation_data(data[0])
    data_torch = torch.tensor(data, dtype=torch.float32)
    rmse_torch, X_pred_torch = eval_inv_error_torch(data_torch[0], model_1d_2)
    visualization_plot_1d(data[0][:, :-1], data[0][:, -1], X_pred_torch, model_1d_2, inv_flag=True)
    # rmse_torch, X_pred_torch = eval_abs_inv_error_torch(data_torch[0], model_1d)
    # visualization_plot_1d(data[0][:, :-1], data[0][:, -1], X_pred_torch, model_1d, inv_flag=False)
    print(f"RMSE torch: {rmse_torch}")

    # write code here

if __name__ == "__main__":
    main()
