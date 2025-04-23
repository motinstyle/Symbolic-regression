import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Callable, List, Tuple

def generate_dataset(quadrants: List[int], n_points: int = 200) -> torch.Tensor:
    """
    Генерирует 2D датасет в указанных квадрантах.
    
    Args:
        quadrants: Список квадрантов (1,2,3,4)
        n_points: Количество точек
    
    Returns:
        torch.Tensor формы [n_points * len(quadrants), 2]
    """
    points_per_quadrant = n_points // len(quadrants)
    data_list = []
    
    for q in quadrants:
        if q == 1:  # Первый квадрант (+,+)
            x = torch.tensor(np.random.uniform(0, 10, (points_per_quadrant, 2)), dtype=torch.float32)
        elif q == 2:  # Второй квадрант (-,+)
            x = torch.tensor(np.random.uniform(-10, 0, (points_per_quadrant, 1)), dtype=torch.float32)
            y = torch.tensor(np.random.uniform(0, 10, (points_per_quadrant, 1)), dtype=torch.float32)
            x = torch.cat([x, y], dim=1)
        elif q == 3:  # Третий квадрант (-,-)
            x = torch.tensor(np.random.uniform(-10, 0, (points_per_quadrant, 2)), dtype=torch.float32)
        elif q == 4:  # Четвертый квадрант (+,-)
            x = torch.tensor(np.random.uniform(0, 10, (points_per_quadrant, 1)), dtype=torch.float32)
            y = torch.tensor(np.random.uniform(-10, 0, (points_per_quadrant, 1)), dtype=torch.float32)
            x = torch.cat([x, y], dim=1)
        else:
            raise ValueError(f"Неизвестный квадрант: {q}")
        
        data_list.append(x)
    
    return torch.cat(data_list, dim=0)

def target_func(x: torch.Tensor) -> torch.Tensor:
    """
    Целевая функция для оптимизации.
    В этом примере используем полином второй степени.
    """
    return 2 * x[:, 0]**2 + 3 * x[:, 1]**2 + x[:, 0] * x[:, 1] + 5

def optimize_with_normalization(X_data: torch.Tensor, y_target: torch.Tensor, normalization_type: str, 
                               steps: int = 1000, lr: float = 0.05) -> Tuple[List[float], torch.Tensor]:
    """
    Оптимизирует предсказание с использованием указанного метода нормализации.
    
    Args:
        X_data: Входной датасет
        y_target: Целевые значения
        normalization_type: Тип нормализации ('mean_std', 'minmax', 'tanh')
        steps: Количество шагов оптимизации
        lr: Скорость обучения
    
    Returns:
        Tuple из: [список потерь, оптимизированный X]
    """
    losses = []
    
    if normalization_type == 'mean_std':
        # Нормализация по среднему и стандартному отклонению
        X_mean = X_data.mean(dim=0)
        X_std = X_data.std(dim=0)
        X_std[X_std == 0] = 1.0  # Защита от деления на ноль
        
        Z = ((X_data - X_mean) / X_std).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([Z], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            X_optim = Z * X_std + X_mean
            loss = ((target_func(X_optim) - y_target) ** 2).mean()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
        X_final = Z.detach() * X_std + X_mean
        
    elif normalization_type == 'minmax':
        # Min-Max нормализация
        X_min = X_data.min(dim=0).values
        X_max = X_data.max(dim=0).values
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0  # Защита от деления на ноль
        
        Z = ((X_data - X_min) / X_range).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([Z], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            X_optim = Z * X_range + X_min
            loss = ((target_func(X_optim) - y_target) ** 2).mean()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
        X_final = Z.detach() * X_range + X_min
        
    elif normalization_type == 'tanh':
        # tanh mapping нормализация
        X_min = X_data.min(dim=0).values
        X_max = X_data.max(dim=0).values
        X_center = 0.5 * (X_min + X_max)
        X_scale = 0.5 * (X_max - X_min)
        
        # Инвертируем tanh трансформацию для инициализации Z
        Z_init = torch.atanh((X_data - X_center) / X_scale.clamp(min=1e-6))
        Z = Z_init.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([Z], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            X_optim = X_center + X_scale * torch.tanh(Z)
            loss = ((target_func(X_optim) - y_target) ** 2).mean()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
        X_final = (X_center + X_scale * torch.tanh(Z)).detach()
    
    else:
        raise ValueError(f"Неизвестный тип нормализации: {normalization_type}")
    
    return losses, X_final

def visualize_dataset(X_data: torch.Tensor, ax, title: str = None):
    """Визуализирует датасет с указанием квадрантов."""
    ax.scatter(X_data[:, 0], X_data[:, 1], alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.grid(True, linestyle='--', alpha=0.7)
    if title:
        ax.set_title(title)

def run_experiment(quadrants: List[int], title: str):
    """
    Проводит эксперимент для указанных квадрантов и визуализирует результаты.
    
    Args:
        quadrants: Список квадрантов (1,2,3,4)
        title: Заголовок для графика
    """
    # Генерируем датасет
    X_data = generate_dataset(quadrants)
    y_target = target_func(X_data)
    
    # Создаем фигуру для визуализации
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Визуализируем исходный датасет
    ax_data = fig.add_subplot(gs[0, 0])
    visualize_dataset(X_data, ax_data, "Исходный датасет")
    
    # Подготавливаем графики для потерь и результатов
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_results = fig.add_subplot(gs[1, :])
    
    # Запускаем оптимизацию с разными типами нормализации
    norm_types = ['mean_std', 'minmax', 'tanh']
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, norm_type in enumerate(norm_types):
        losses, X_optimized = optimize_with_normalization(X_data, y_target, norm_type)
        
        # Визуализируем потери
        ax_loss.plot(losses, label=norm_type, color=colors[i])
        
        # Визуализируем результаты оптимизации
        ax_results.scatter(X_data[:, 0], X_data[:, 1], alpha=0.3, label=f'Исходные', color='gray')
        ax_results.scatter(X_optimized[:, 0], X_optimized[:, 1], alpha=0.7, 
                         label=f'Оптимизированные ({norm_type})', color=colors[i], marker=markers[i])
    
    # Настройка графика потерь
    ax_loss.set_title("Сравнение потерь при оптимизации")
    ax_loss.set_xlabel("Итерация")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_yscale('log')
    ax_loss.legend()
    ax_loss.grid(True)
    
    # Настройка графика результатов
    ax_results.set_title("Результаты оптимизации")
    ax_results.set_xlabel("X1")
    ax_results.set_ylabel("X2")
    ax_results.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax_results.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax_results.legend()
    ax_results.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# Запускаем эксперименты для разных комбинаций квадрантов
def main():
    # По одному квадранту
    for q in range(1, 5):
        run_experiment([q], f"Данные в {q}-м квадранте")
    
    # Данные в двух квадрантах
    run_experiment([1, 3], "Данные в 1-м и 3-м квадрантах")
    run_experiment([2, 4], "Данные в 2-м и 4-м квадрантах")
    
    # Данные во всех квадрантах
    run_experiment([1, 2, 3, 4], "Данные во всех квадрантах")

if __name__ == "__main__":
    main()