import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import time
from typing import Tuple, List, Dict, Union

# Добавляем пути для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../sr')))

# Импорт из модулей SR
from sr.sr_tree import Node, Tree, FUNCTIONS
from sr.nodes import *
from sr.parameters import *

# Устанавливаем параметры
CONST_OPT = False
REQUIRES_GRAD = True  # Изменено на True для работы eval_inv_error_torch

def build_tree_x0_squared_plus_2x1():
    """
    Создает дерево для функции x[0]^2 + 2*x[1]
    
    Структура дерева:
    - Корень: sum_ (x[0]^2, 2*x[1])
      - Левый узел: pow2_ (x[0])
        - x[0]
      - Правый узел: mult_ (2, x[1])
        - Левый узел: const 2
        - Правый узел: x[1]
    """
    # Создаем листовые узлы
    x0_node = Node("var", 1, FUNCTIONS['ident_'].func, [0], requires_grad=REQUIRES_GRAD)
    x1_node = Node("var", 1, FUNCTIONS['ident_'].func, [1], requires_grad=REQUIRES_GRAD)
    const_2_node = Node("ident", 1, FUNCTIONS['const_'].func, torch.tensor([2.0]), requires_grad=False)
    
    # Создаем узел x[0]^2
    x0_squared = Node("func", 1, FUNCTIONS['pow2_'].func, [x0_node], requires_grad=REQUIRES_GRAD)
    
    # Создаем узел 2*x[1]
    mult_2_x1 = Node("func", 2, FUNCTIONS['mult_'].func, [const_2_node, x1_node], requires_grad=REQUIRES_GRAD)
    
    # Создаем корневой узел (x[0]^2 + 2*x[1])
    root_node = Node("func", 2, FUNCTIONS['sum_'].func, [x0_squared, mult_2_x1], requires_grad=REQUIRES_GRAD)
    
    # Создаем дерево
    tree = Tree(root_node, requires_grad=REQUIRES_GRAD)
    tree.num_vars = 2  # Устанавливаем количество переменных
    tree.update_var_counts()  # Обновляем счетчики переменных
    
    return tree

def load_dataset(filename='sum_of_2pow.csv'):
    """Загружает dataset из CSV файла"""
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../new_datasets', filename))
    data = pd.read_csv(file_path)
    
    # Извлекаем входные признаки (X) и целевые значения (y)
    X_data = data.iloc[:, :-1].values
    y_data = data.iloc[:, -1].values
    
    return X_data, y_data

def wrapped_model_func(x: Union[np.ndarray, float], target_y: float, tree: Tree) -> float:
    """
    Вычисляет разницу между выходом дерева и целевым значением в квадрате
    
    Args:
        x: Входные данные
        target_y: Целевое значение
        tree: Дерево модели
    Returns:
        Квадрат разницы между выходом дерева и целевым значением
    """

    # print("input wrapped_model_func type(x), type(target_y), type(tree):", type(x), type(target_y), type(tree))

    try:
        # print("x.type:", type(x))
        # print("x.shape:", x.shape)
        # Обеспечиваем правильную форму входных данных
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
            x_input = np.array([x]).reshape(1, -1)
        else:
            x_input = x.reshape(1, -1)
        
        # print("x_input.type:", type(x_input))
        # print("x_input.shape:", x_input.shape)

        # Преобразуем входной массив в тензор для forward
        x_tensor = to_tensor(x_input)
        # print("x_tensor.type:", type(x_tensor))
        
        # Прямой проход через модель - numpy массив на входе
        output = tree.forward(x_tensor)
        # print("output.type:", type(output))
        
        # Приводим выход к скаляру, если это массив numpy
        if isinstance(output, np.ndarray):
            if output.size > 1:
                output = output.flatten()[0]
            else:
                output = output.item() if hasattr(output, 'item') else output.flatten()[0]
        # Если это torch тензор, преобразуем его в numpy и затем в скаляр
        elif hasattr(output, 'detach'):
            output_np = output.detach().numpy()
            if output_np.size > 1:
                output = output_np.flatten()[0]
            else:
                output = output_np.item() if hasattr(output_np, 'item') else output_np.flatten()[0]
        
        # Вычисляем квадрат разницы
        diff = (output - target_y) ** 2
        
        return diff
    except Exception as e:
        # Возвращаем высокое значение в случае ошибки
        print(f"Ошибка в wrapped_model_func: {e}")
        return 1e10

def eval_inv_error(X_data, y_data, tree, num_points=100):
    """
    Оценивает обратную ошибку модели
    
    Args:
        X_data: Входные данные
        y_data: Целевые значения
        tree: Дерево модели
        num_points: Количество точек для вычисления ошибки
    
    Returns:
        Средняя обратная ошибка, предсказанные входные данные
    """
    # Если указано больше точек, чем есть в данных, используем все точки
    if num_points > len(y_data):
        num_points = len(y_data)
    
    # Случайно выбираем указанное количество точек
    indices = np.random.choice(len(y_data), size=num_points, replace=False)
    X_data_sampled = torch.tensor(X_data[indices])
    y_data_sampled = torch.tensor(y_data[indices])
    
    inv_losses = []
    predictions = []
    
    # Оцениваем обратную ошибку для каждой выбранной точки
    for i, (x, y) in enumerate(zip(X_data_sampled, y_data_sampled)):
        try:
            # Начальное приближение: используем реальные входные данные
            x0 = np.array(x)
            
            # Целевое значение для текущей точки
            target = y

            # print("type(x), type(x0), type(target):", type(x), type(x0), type(target))
            
            # Функция оптимизации для текущей цели
            def objective(x_input):
                return wrapped_model_func(x_input, target, tree)
            
            # objective(x)
            # quit()
            
            # Выполняем оптимизацию для поиска оптимального входа
            result = minimize(
                objective,
                x0,
                # x,
                method='Nelder-Mead',
                options={'maxiter': 100}
            )
            
            # Извлекаем результат
            x_pred = result.x
            predictions.append(x_pred)
            
            # Вычисляем обратную ошибку: среднеквадратическую ошибку между оригинальным и предсказанным входом
            # print("x_pred.type:",
            inv_loss = np.mean((x_pred - x.detach().numpy()) ** 2)
            inv_losses.append(inv_loss)
            
            if i < 5:  # Выводим детали для первых 5 точек
                print(f"Точка {i}:")
                print(f"  Оригинальный вход: {x}")
                
                # Преобразуем вход в numpy для вызова forward
                x_numpy = to_numpy(x.reshape(1, -1))
                model_output = tree.forward(x_numpy)
                if hasattr(model_output, 'detach'):
                    model_output = model_output.detach().numpy()
                print(f"  Выход модели: {model_output}")
                
                print(f"  Целевое значение: {y}")
                print(f"  Найденный вход: {x_pred}")
                
                # Снова преобразуем вход для второго вызова forward
                x_pred_numpy = to_numpy(x_pred.reshape(1, -1))
                pred_output = tree.forward(x_pred_numpy)
                if hasattr(pred_output, 'detach'):
                    pred_output = pred_output.detach().numpy()
                print(f"  Выход для найденного входа: {pred_output}")
                
                print(f"  Обратная ошибка: {inv_loss}")
                print()
            
        except Exception as e:
            # Возвращаем высокое значение в случае ошибки
            print(f"Ошибка при оценке точки {i}: {e}")
            inv_losses.append(1e10)
            predictions.append(np.full_like(x, np.nan))
    
    # Вычисляем среднюю обратную ошибку
    avg_inv_loss = np.mean(inv_losses) if inv_losses else 1e10
    
    return avg_inv_loss, np.array(predictions), X_data_sampled, y_data_sampled

def to_numpy(x):
    """
    Преобразует вход в numpy массив, независимо от его исходного типа
    
    Args:
        x: Входные данные (np.ndarray, torch.Tensor и т.д.)
    Returns:
        numpy массив
    """
    if hasattr(x, 'detach'): # Это torch.Tensor
        return x.detach().numpy()
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif isinstance(x, (int, float)):
        return np.array([x], dtype=np.float32)
    else:
        return np.array(x, dtype=np.float32)

def visualization_plot_2d(X_data, original_y, X_pred, tree):
    """
    Визуализирует оригинальные и предсказанные данные
    
    Args:
        X_data: Оригинальные входные данные
        original_y: Оригинальные выходные данные
        X_pred: Предсказанные входные данные
        tree: Дерево модели
    """
    # Преобразуем все входные данные в numpy массивы
    X_data_np = to_numpy(X_data)
    original_y_np = to_numpy(original_y)
    X_pred_np = to_numpy(X_pred)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Вычисляем выходы модели для оригинальных и предсказанных входов
    original_outputs = []
    for x in X_data_np:
        output = tree.forward(x.reshape(1, -1))
        if hasattr(output, 'detach'):
            output = output.detach().numpy()
        original_outputs.append(output.item() if hasattr(output, 'item') else output.flatten()[0])
    
    predicted_outputs = []
    for x in X_pred_np:
        output = tree.forward(x.reshape(1, -1))
        if hasattr(output, 'detach'):
            output = output.detach().numpy()
        predicted_outputs.append(output.item() if hasattr(output, 'item') else output.flatten()[0])
    
    original_outputs = np.array(original_outputs)
    predicted_outputs = np.array(predicted_outputs)
    
    # 3D точечный график: оригинальные и предсказанные точки
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_data_np[:, 0], X_data_np[:, 1], original_y_np, color='blue', label='Оригинальные точки', alpha=0.7)
    ax1.scatter(X_pred_np[:, 0], X_pred_np[:, 1], predicted_outputs, color='red', label='Предсказанные точки', alpha=0.7)
    
    # Соединяем соответствующие точки линиями
    for i in range(len(X_data_np)):
        ax1.plot([X_data_np[i, 0], X_pred_np[i, 0]],
                 [X_data_np[i, 1], X_pred_np[i, 1]],
                 [original_y_np[i], predicted_outputs[i]],
                 'k-', alpha=0.2)
    
    ax1.set_title('Оригинальные vs. Предсказанные точки')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('Y')
    ax1.legend()
    
    # Диаграмма рассеяния: оригинальные vs предсказанные X₁
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_data_np[:, 0], X_pred_np[:, 0], alpha=0.7)
    
    # Линия идеального предсказания
    min_x = min(np.min(X_data_np[:, 0]), np.min(X_pred_np[:, 0]))
    max_x = max(np.max(X_data_np[:, 0]), np.max(X_pred_np[:, 0]))
    ax2.plot([min_x, max_x], [min_x, max_x], 'r--', label='Идеальное предсказание')
    
    ax2.set_title('Оригинальные vs. Предсказанные X₁')
    ax2.set_xlabel('Оригинальный X₁')
    ax2.set_ylabel('Предсказанный X₁')
    ax2.grid(True)
    ax2.legend()
    
    # Диаграмма рассеяния: оригинальные vs предсказанные X₂
    ax3 = fig.add_subplot(133)
    ax3.scatter(X_data_np[:, 1], X_pred_np[:, 1], alpha=0.7)
    
    # Линия идеального предсказания
    min_x = min(np.min(X_data_np[:, 1]), np.min(X_pred_np[:, 1]))
    max_x = max(np.max(X_data_np[:, 1]), np.max(X_pred_np[:, 1]))
    ax3.plot([min_x, max_x], [min_x, max_x], 'r--', label='Идеальное предсказание')
    
    ax3.set_title('Оригинальные vs. Предсказанные X₂')
    ax3.set_xlabel('Оригинальный X₂')
    ax3.set_ylabel('Предсказанный X₂')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
def plot_error_histogram(X_data, X_pred):
    """
    Строит гистограмму ошибок предсказания
    
    Args:
        X_data: Оригинальные входные данные
        X_pred: Предсказанные входные данные
    """
    # Преобразуем входные данные в numpy массивы
    X_data_np = to_numpy(X_data)
    X_pred_np = to_numpy(X_pred)
    
    # Вычисляем ошибки по каждой координате
    errors_x1 = np.abs(X_data_np[:, 0] - X_pred_np[:, 0])
    errors_x2 = np.abs(X_data_np[:, 1] - X_pred_np[:, 1])
    
    # Евклидово расстояние
    euclidean_errors = np.sqrt(np.sum((X_data_np - X_pred_np)**2, axis=1))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Гистограмма ошибок для X₁
    axes[0].hist(errors_x1, bins=20, alpha=0.7)
    axes[0].set_title('Распределение ошибок X₁')
    axes[0].set_xlabel('Абсолютная ошибка')
    axes[0].set_ylabel('Частота')
    axes[0].grid(True, alpha=0.3)
    
    # Гистограмма ошибок для X₂
    axes[1].hist(errors_x2, bins=20, alpha=0.7)
    axes[1].set_title('Распределение ошибок X₂')
    axes[1].set_xlabel('Абсолютная ошибка')
    axes[1].set_ylabel('Частота')
    axes[1].grid(True, alpha=0.3)
    
    # Гистограмма евклидовых расстояний
    axes[2].hist(euclidean_errors, bins=20, alpha=0.7)
    axes[2].set_title('Распределение евклидовых расстояний')
    axes[2].set_xlabel('Евклидово расстояние')
    axes[2].set_ylabel('Частота')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_model_surface(tree, X_data, y_data, X_pred):
    """
    Рисует поверхность модели и оригинальные/предсказанные точки
    
    Args:
        tree: Дерево модели
        X_data: Оригинальные входные данные
        y_data: Оригинальные выходные данные
        X_pred: Предсказанные входные данные
    """
    # Преобразуем входные данные в numpy массивы
    X_data_np = to_numpy(X_data)
    y_data_np = to_numpy(y_data)
    X_pred_np = to_numpy(X_pred)
    
    # Создаем сетку для построения поверхности
    x_min, x_max = X_data_np[:, 0].min() - 1, X_data_np[:, 0].max() + 1
    y_min, y_max = X_data_np[:, 1].min() - 1, X_data_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    
    # Вычисляем значения функции на сетке
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            input_point = np.array([[xx[i, j], yy[i, j]]])
            output = tree.forward(input_point)
            if hasattr(output, 'detach'):
                output = output.detach().numpy()
            zz[i, j] = output.item() if hasattr(output, 'item') else output.flatten()[0]
    
    # Вычисляем выходы для предсказанных входов
    predicted_outputs = []
    for x in X_pred_np:
        output = tree.forward(x.reshape(1, -1))
        if hasattr(output, 'detach'):
            output = output.detach().numpy()
        predicted_outputs.append(output.item() if hasattr(output, 'item') else output.flatten()[0])
    predicted_outputs = np.array(predicted_outputs)
    
    # Рисуем поверхность и точки
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Поверхность модели
    surf = ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')
    
    # Оригинальные точки
    ax.scatter(X_data_np[:, 0], X_data_np[:, 1], y_data_np, color='blue', label='Оригинальные точки', s=20, alpha=0.7)
    
    # Предсказанные точки
    ax.scatter(X_pred_np[:, 0], X_pred_np[:, 1], predicted_outputs, color='red', label='Предсказанные точки', s=20, alpha=0.7)
    
    # Соединяем соответствующие точки линиями
    for i in range(len(X_data_np)):
        ax.plot([X_data_np[i, 0], X_pred_np[i, 0]],
                [X_data_np[i, 1], X_pred_np[i, 1]],
                [y_data_np[i], predicted_outputs[i]],
                'k-', alpha=0.2)
    
    ax.set_title('Поверхность модели с оригинальными и предсказанными точками')
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_zlabel('Y')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def compare_inverse_error_methods(tree: Tree, X_data: np.ndarray, y_data: np.ndarray, num_points: int = 50) -> Dict:
    """
    Сравнивает три метода оценки обратной ошибки:
    1. eval_inv_error из tree_inv_check.py
    2. eval_inv_error из sr_tree.py
    3. eval_inv_error_torch из sr_tree.py
    
    Args:
        tree: Дерево модели
        X_data: Входные данные
        y_data: Целевые значения
        num_points: Количество точек для тестирования
        
    Returns:
        Dict: Словарь с результатами сравнения
    """
    # Если указано больше точек, чем есть в данных, используем все точки
    if num_points > len(y_data):
        num_points = len(y_data)
    
    # Случайно выбираем указанное количество точек
    indices = np.random.choice(len(y_data), size=num_points, replace=False)
    X_data_sampled = X_data[indices]
    y_data_sampled = y_data[indices]
    
    # Подготовка данных в формате, необходимом для sr_tree.py функций
    combined_data = np.column_stack((X_data_sampled, y_data_sampled))
    
    results = {}
    
    # 1. Оценка с использованием eval_inv_error из tree_inv_check.py
    print("\n1. Оценка с использованием eval_inv_error из tree_inv_check.py...")
    start_time = time.time()
    avg_inv_loss, X_pred, X_sampled, y_sampled = eval_inv_error(X_data, y_data, tree, num_points)
    end_time = time.time()
    
    results['tree_inv_check'] = {
        'error': avg_inv_loss,
        'time': end_time - start_time,
        'X_pred': X_pred
    }
    
    print(f"  Средняя ошибка: {avg_inv_loss:.6f}")
    print(f"  Время выполнения: {end_time - start_time:.3f} сек")
    
    # 2. Оценка с использованием eval_inv_error из sr_tree.py
    print("\n2. Оценка с использованием eval_inv_error из sr_tree.py...")
    start_time = time.time()
    sr_inv_error = tree.eval_inv_error(combined_data)
    end_time = time.time()
    
    results['sr_tree_inv_error'] = {
        'error': sr_inv_error,
        'time': end_time - start_time
    }
    
    print(f"  Средняя ошибка: {sr_inv_error:.6f}")
    print(f"  Время выполнения: {end_time - start_time:.3f} сек")
    
    # 3. Оценка с использованием eval_inv_error_torch из sr_tree.py
    print("\n3. Оценка с использованием eval_inv_error_torch из sr_tree.py...")
    start_time = time.time()
    torch_inv_error = tree.eval_inv_error_torch(combined_data)
    end_time = time.time()
    
    results['sr_tree_inv_error_torch'] = {
        'error': torch_inv_error,
        'time': end_time - start_time
    }
    
    print(f"  Средняя ошибка: {torch_inv_error:.6f}")
    print(f"  Время выполнения: {end_time - start_time:.3f} сек")
    
    return results

def plot_comparison_results(results: Dict):
    """
    Строит графики для сравнения результатов методов
    
    Args:
        results: Словарь с результатами сравнения
    """
    method_names = list(results.keys())
    errors = [results[method]['error'] for method in method_names]
    times = [results[method]['time'] for method in method_names]
    
    # Переформатируем метки для лучшей читаемости
    method_labels = [
        'eval_inv_error\n(tree_inv_check.py)', 
        'eval_inv_error\n(sr_tree.py)', 
        'eval_inv_error_torch\n(sr_tree.py)'
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # График средних ошибок
    bar1 = axes[0].bar(method_labels, errors, alpha=0.7, color=['blue', 'green', 'red'])
    axes[0].set_ylabel('Средняя ошибка')
    axes[0].set_title('Сравнение средних ошибок')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar in bar1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    # График времени выполнения
    bar2 = axes[1].bar(method_labels, times, alpha=0.7, color=['blue', 'green', 'red'])
    axes[1].set_ylabel('Время выполнения (сек)')
    axes[1].set_title('Сравнение времени выполнения')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar in bar2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # Создаем график отношения ошибки к времени (эффективность)
    efficiency = [errors[i] / times[i] for i in range(len(errors))]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(method_labels, efficiency, alpha=0.7, color=['blue', 'green', 'red'])
    plt.ylabel('Ошибка/Время (ниже - лучше)')
    plt.title('Эффективность методов (отношение ошибки к времени)')
    plt.grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.show()

def main():
    # Создаем дерево для функции x[0]^2 + 2*x[1]
    tree = build_tree_x0_squared_plus_2x1()
    print(f"Создано дерево для функции: {tree.to_math_expr()}")
    print(f"Количество переменных: {tree.num_vars}")
    print()
    
    # Загружаем набор данных
    X_data, y_data = load_dataset('sum_of_2pow.csv')
    print(f"Загружен набор данных: {len(X_data)} точек")
    print(f"Диапазон X₁: [{X_data[:, 0].min():.2f}, {X_data[:, 0].max():.2f}]")
    print(f"Диапазон X₂: [{X_data[:, 1].min():.2f}, {X_data[:, 1].max():.2f}]")
    print(f"Диапазон Y: [{y_data.min():.2f}, {y_data.max():.2f}]")
    print()
    
    # Сравниваем методы оценки обратной ошибки
    print("Сравниваем методы оценки обратной ошибки...")
    results = compare_inverse_error_methods(tree, X_data, y_data, num_points=50)
    
    # Визуализируем результаты сравнения
    print("\nВизуализация результатов сравнения...")
    plot_comparison_results(results)
    
    # Выполняем оценку обратной ошибки с tree_inv_check.py методом для визуализации
    print("\nОцениваем обратную ошибку для визуализации...")
    avg_inv_loss, X_pred, X_sampled, y_sampled = eval_inv_error(X_data, y_data, tree, num_points=100)
    print(f"Средняя обратная ошибка: {avg_inv_loss:.6f}")
    print()
    
    # Визуализируем результаты
    print("Визуализация результатов...")
    visualization_plot_2d(X_sampled, y_sampled, X_pred, tree)
    plot_error_histogram(X_sampled, X_pred)
    plot_model_surface(tree, X_sampled, y_sampled, X_pred)

if __name__ == "__main__":
    main()
