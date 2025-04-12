import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NUM_OF_EPOCHS = 50
NOISE_SCALE = 5

def generate_data(n: int, d: int) -> np.ndarray:
    X = np.random.randn(n, d)
    noise = np.random.normal(1, NOISE_SCALE, n)
    y = np.sum(X ** 2, axis=1) + noise
    return np.hstack((X, y.reshape(-1, 1)))

def model1(x: torch.Tensor) -> torch.Tensor:
    return 2*(x[:, 0]+500) ** 2 + 2*(x[:, 1]+500) ** 2 - 10 

def visualize_data(data: np.ndarray, X_pred: np.ndarray, model):
    """
    Визуализирует данные, предсказанные точки и график модели.
    Соединяет исходные и предсказанные точки отрезками.
    
    Args:
        data: Исходный датасет в формате [X, y]
        X_pred: Предсказанные значения X
        model: Функция модели, принимающая X и возвращающая y
    """
    # Преобразуем данные в numpy, если они в формате torch.Tensor
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(X_pred, torch.Tensor):
        X_pred = X_pred.detach().cpu().numpy()
        
    # Разделяем входные и выходные данные
    X_data = data[:, :-1]
    y_data = data[:, -1]
    
    # Вычисляем выходы модели для предсказанных входов
    if X_pred.ndim == 1:
        X_pred_reshaped = X_pred.reshape(-1, 1)
    else:
        X_pred_reshaped = X_pred
        
    # Преобразуем в тензор для использования с моделью
    X_pred_tensor = torch.tensor(X_pred_reshaped, dtype=torch.float32)
    y_pred = model(X_pred_tensor)
    
    # Преобразуем результат в numpy, если это тензор
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Создаем график в зависимости от размерности данных
    if X_data.shape[1] == 1:  # Одномерные данные
        plt.figure(figsize=(12, 8))
        
        # Рисуем исходные точки
        plt.scatter(X_data.flatten(), y_data, color='blue', s=50, alpha=0.7, label='Исходные точки')
        
        # Рисуем предсказанные точки
        plt.scatter(X_pred.flatten(), y_pred, color='red', s=50, alpha=0.7, label='Предсказанные точки')
        
        # Соединяем соответствующие точки отрезками
        for i in range(len(X_data)):
            plt.plot([X_data[i, 0], X_pred[i, 0]], 
                     [y_data[i], y_pred[i]], 
                     'k-', alpha=0.3)
        
        # Отображаем график функции
        x_range = np.linspace(min(X_data.min(), X_pred.min()) - 0.5, 
                             max(X_data.max(), X_pred.max()) + 0.5, 
                             1000).reshape(-1, 1)
        x_range_tensor = torch.tensor(x_range, dtype=torch.float32)
        y_range = model(x_range_tensor)
        if isinstance(y_range, torch.Tensor):
            y_range = y_range.detach().cpu().numpy()
        
        plt.plot(x_range.flatten(), y_range, 'g-', linewidth=2, label='Модель')
        
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.title('Визуализация данных и предсказаний', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
    else:  # Двумерные данные
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Рисуем исходные точки
        ax.scatter(X_data[:, 0], X_data[:, 1], y_data, 
                   color='blue', s=50, alpha=0.7, label='Исходные точки')
        
        # Рисуем предсказанные точки
        ax.scatter(X_pred[:, 0], X_pred[:, 1], y_pred, 
                   color='red', s=50, alpha=0.7, label='Предсказанные точки')
        
        # Соединяем соответствующие точки отрезками
        for i in range(len(X_data)):
            ax.plot([X_data[i, 0], X_pred[i, 0]], 
                    [X_data[i, 1], X_pred[i, 1]], 
                    [y_data[i], y_pred[i]], 
                    'k-', alpha=0.3)
        
        # Отображаем поверхность функции
        x_min, x_max = min(X_data[:, 0].min(), X_pred[:, 0].min()) - 0.5, max(X_data[:, 0].max(), X_pred[:, 0].max()) + 0.5
        y_min, y_max = min(X_data[:, 1].min(), X_pred[:, 1].min()) - 0.5, max(X_data[:, 1].max(), X_pred[:, 1].max()) + 0.5
        
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 30), 
                                    np.linspace(y_min, y_max, 30))
        
        # Подготавливаем данные для поверхности
        grid_points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        
        # Вычисляем значения функции
        z_grid = model(grid_tensor)
        if isinstance(z_grid, torch.Tensor):
            z_grid = z_grid.detach().cpu().numpy()
        
        # Преобразуем в формат, подходящий для отображения поверхности
        z_grid = z_grid.reshape(x_grid.shape)
        
        # Рисуем поверхность
        surf = ax.plot_surface(x_grid, y_grid, z_grid, 
                              cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
        
        # Добавляем цветовую шкалу для поверхности
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        ax.set_xlabel('x₁', fontsize=14)
        ax.set_ylabel('x₂', fontsize=14)
        ax.set_zlabel('y', fontsize=14)
        ax.set_title('Визуализация данных и предсказаний', fontsize=16)
        ax.legend(fontsize=12)
        
    plt.tight_layout()
    plt.show()

def eval_inv_error_torch(data: np.ndarray, steps=500, lr=1e-1, tol=1e-2, penalty=None) -> tuple:
    """
    Evaluate inverse error using PyTorch optimization (batch gradient descent).
    Penalizes samples where optimization can't find x such that f(x) ≈ y.
    
    Returns:
        tuple: (error, X_optim, y_final) - среднее значение ошибки, оптимизированные входы, выходы модели
    """

    # Определяем устройство - используем CUDA, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Подготавливаем данные - переносим на нужное устройство
    data = torch.tensor(data, dtype=torch.float32).to(device)

    # Разделяем входные и выходные данные
    X_orig = data[:, :-1]
    y_target = data[:, -1]

    # Создаем тензор для оптимизации с градиентами
    X_optim = X_orig.clone().detach().requires_grad_(True)

    # Создаем оптимизатор
    # optimizer = torch.optim.SGD([X_optim], lr=lr)
    optimizer = torch.optim.Adam([X_optim], lr=lr)

    # Выполняем оптимизацию
    for _ in range(steps):
        optimizer.zero_grad()
        
        # Вычисляем предсказания модели
        y_pred = model1(X_optim)
        
        # Убеждаемся, что выходы на том же устройстве, что и целевые значения
        if y_pred.device != y_target.device:
            y_pred = y_pred.to(device)
            
        # Вычисляем функцию потерь
        loss = torch.mean((y_pred - y_target) ** 2)
        
        # Вычисляем градиенты и делаем шаг оптимизации
        loss.backward()
        optimizer.step()
        # print(X_optim[0])

    # Считаем финальное f(x') после оптимизации
    with torch.no_grad():
        # Убедимся, что все тензоры на одном устройстве перед вычислениями
        y_final = model1(X_optim)
        if y_final.device != device:
            y_final = y_final.to(device)
            
        diff = torch.abs(y_final - y_target)

        # Значения, где не удалось приблизиться к y_target
        if penalty is None:
            # Используем нормализованный penalty по размеру пространства
            data_range = torch.max(X_orig, dim=0).values - torch.min(X_orig, dim=0).values
            penalty = torch.sum(data_range).item()

        # Определяем маски для успешных и неудачных оптимизаций
        failed_mask = diff > tol
        success_mask = ~failed_mask

        # Ошибка = либо расстояние в пространстве x, либо penalty
        print(X_optim)
        print(X_orig)
        spatial_error = torch.norm(X_optim - X_orig, dim=1)
        print(spatial_error)
        error = torch.where(success_mask, spatial_error, torch.full_like(spatial_error, penalty, device=device))
        
        # Выводим статистику об успешности оптимизации
        total = len(y_target)
        successful = success_mask.sum().item()
        failed = failed_mask.sum().item()
        print(f"Успешных оптимизаций: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"Неудачных оптимизаций: {failed}/{total} ({failed/total*100:.1f}%)")
        print(error)
        
        # Возвращаем среднюю ошибку, оптимизированные входы и их выходы
        return error.mean().item(), X_optim.detach().cpu().numpy(), y_final.detach().cpu().numpy()

def main():
    data = generate_data(100, 2)
    inv_error, X_pred, y_pred = eval_inv_error_torch(data)
    print(f"Средняя ошибка инверсии: {inv_error}")
    
    # Визуализируем результаты
    visualize_data(data, X_pred, model1)

if __name__ == "__main__":
    main()













