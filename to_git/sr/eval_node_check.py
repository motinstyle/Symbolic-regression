import numpy as np
import torch

# Импортируем классы и функции из текущего модуля sr
from sr_tree import Node, Tree
from nodes import pow2_, sum_, mult_, ident_

# Константы для численной устойчивости
EPSILON = 1e-10
MAX_VALUE = 1e9

"""
ВАЖНО: ОБНАРУЖЕНА ПРОБЛЕМА В ОБРАБОТКЕ БАТЧА

Мы обнаружили, что реализация Tree.forward (и, соответственно, Node.eval_node) 
некорректно обрабатывает батч входных данных. При передаче батча, они возвращают
результат только для первого элемента батча.

Для решения этой проблемы мы предлагаем два варианта:

1. Использовать наш метод numpy_forward, который корректно обрабатывает батч
   с использованием NumPy (реализовано в данном файле).

2. Исправить реализацию Node.eval_node в sr_tree.py для корректной обработки батча:
   - В методе Node.eval_node при обработке узла переменной (t_name == "var")
     нужно убедиться, что возвращается весь столбец данных для всех образцов в батче.
   - В методе Node.eval_node при обработке узла константы (t_name == "ident")
     нужно убедиться, что константа размножается на весь батч.
   - Для всех остальных узлов нужно убедиться, что операции применяются не только
     к первому элементу батча, но ко всему батчу.

Предлагаем использовать numpy_forward как временное решение,
пока не будет исправлена реализация в sr_tree.py.
"""

# Функция для преобразования различных типов данных в numpy массив
def to_numpy(x):
    """
    Преобразует вход различных типов в numpy массив.
    
    Args:
        x: Входные данные (np.ndarray, torch.Tensor, list, float, int и т.д.)
        
    Returns:
        np.ndarray: Данные, преобразованные в numpy массив
    """
    if hasattr(x, 'detach') and hasattr(x, 'numpy'):  # Если это torch.Tensor
        return x.detach().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

# Функции для вычисления выражений с использованием numpy
def np_ident(x):
    """Функция идентичности"""
    return x

def np_pow2(x):
    """Функция возведения в квадрат"""
    return np.power(x, 2)

def np_sum(x, y):
    """Функция сложения"""
    return x + y

def np_mult(x, y):
    """Функция умножения"""
    return x * y

# Соответствие функций PyTorch и NumPy
FUNC_MAPPING = {
    'ident_': np_ident,
    'pow2_': np_pow2,
    'sum_': np_sum,
    'mult_': np_mult
}

def numpy_eval(node, inputs):
    """
    Вычисляет значение узла с использованием NumPy (без PyTorch).
    
    Args:
        node: Узел для вычисления
        inputs: Входные данные (numpy array)
        
    Returns:
        numpy.ndarray: Результат вычисления
    """
    # Преобразуем inputs в numpy array
    inputs_np = to_numpy(inputs)
    
    # Обеспечим, что inputs_np имеет размерность [batch_size, n_features]
    if len(inputs_np.shape) == 1:
        inputs_np = inputs_np.reshape(1, -1)
    
    # Тип узла
    node_type = node.t_name
    
    if node_type == "var":
        # Узел переменной - возвращаем соответствующую переменную из inputs
        var_idx = node.data[0]
        return inputs_np[:, var_idx:var_idx+1]
    elif node_type == "ident" and not isinstance(node.data, list):
        # Узел константы - возвращаем константу
        const_value = node.data
        return np.full((inputs_np.shape[0], 1), const_value)
    elif node_type == "func":
        # Узел функции - вычисляем значение функции
        func_name = node.func.__name__
        np_func = FUNC_MAPPING.get(func_name)
        
        if np_func is None:
            raise ValueError(f"Функция {func_name} не найдена в FUNC_MAPPING")
        
        # Получаем аргументы функции
        if node.parity == 1:
            # Унарная функция
            arg = numpy_eval(node.data[0], inputs_np)
            return np_func(arg)
        else:
            # Бинарная функция
            arg1 = numpy_eval(node.data[0], inputs_np)
            arg2 = numpy_eval(node.data[1], inputs_np)
            return np_func(arg1, arg2)
    else:
        raise ValueError(f"Неизвестный тип узла: {node_type}")

def numpy_forward(tree, inputs):
    """
    Вычисляет значение дерева с использованием NumPy (без PyTorch).
    
    Args:
        tree: Дерево для вычисления
        inputs: Входные данные (numpy array)
        
    Returns:
        numpy.ndarray: Результат вычисления
    """
    return numpy_eval(tree.start_node, inputs)

def prepare_input(inputs):
    """
    Подготавливает входные данные для функции forward.
    Преобразует входные данные в правильный формат для функции forward.
    
    Args:
        inputs: Входные данные (np.ndarray, torch.Tensor, list)
        
    Returns:
        torch.Tensor: Данные, готовые для передачи в функцию forward
    """
    # Преобразуем входные данные в numpy, если это не numpy массив
    inputs_np = to_numpy(inputs)
    
    # Проверяем, является ли это одиночным входом или батчем
    if len(inputs_np.shape) == 1:
        # Одиночный вход (вектор признаков) - добавляем измерение батча
        inputs_np = np.expand_dims(inputs_np, axis=0)
    
    # Преобразуем numpy массив в PyTorch тензор
    return torch.tensor(inputs_np, dtype=torch.float32)

def create_tree_1var():
    """
    Создает дерево для функции f(x0) = x0^2 + x0
    """
    # Создаем узлы дерева
    var_node = Node("var", 1, None, [0], requires_grad=False)  # узел для переменной x0
    pow2_node = Node("func", 1, pow2_, [var_node], requires_grad=False)  # узел для x0^2
    sum_node = Node("func", 2, sum_, [pow2_node, var_node], requires_grad=False)  # узел для x0^2 + x0
    
    # Создаем дерево с корнем в узле sum_node
    tree = Tree(sum_node, requires_grad=False)
    
    print("Создано дерево для функции f(x0) = x0^2 + x0")
    print("Представление дерева в виде строки:", tree)
    print("Математическое выражение:", tree.to_math_expr())
    
    return tree

def create_tree_2var():
    """
    Создает дерево для функции f(x0, x1) = x0^2 + x1^2
    """
    # Создаем узлы дерева
    var_node_0 = Node("var", 1, None, [0], requires_grad=False)  # узел для переменной x0
    var_node_1 = Node("var", 1, None, [1], requires_grad=False)  # узел для переменной x1
    
    pow2_node_0 = Node("func", 1, pow2_, [var_node_0], requires_grad=False)  # узел для x0^2
    pow2_node_1 = Node("func", 1, pow2_, [var_node_1], requires_grad=False)  # узел для x1^2
    
    sum_node = Node("func", 2, sum_, [pow2_node_0, pow2_node_1], requires_grad=False)  # узел для x0^2 + x1^2
    
    # Создаем дерево с корнем в узле sum_node
    tree = Tree(sum_node, requires_grad=False)
    
    print("Создано дерево для функции f(x0, x1) = x0^2 + x1^2")
    print("Представление дерева в виде строки:", tree)
    print("Математическое выражение:", tree.to_math_expr())
    
    return tree

def create_tree_2var_complex():
    """
    Создает дерево для функции f(x0, x1) = x0^2 + 2*x1
    """
    # Создаем узлы дерева
    var_node_0 = Node("var", 1, None, [0], requires_grad=False)  # узел для переменной x0
    var_node_1 = Node("var", 1, None, [1], requires_grad=False)  # узел для переменной x1
    
    pow2_node = Node("func", 1, pow2_, [var_node_0], requires_grad=False)  # узел для x0^2
    
    const_node = Node("ident", 1, None, 2, requires_grad=False)  # узел для константы 2
    mult_node = Node("func", 2, mult_, [const_node, var_node_1], requires_grad=False)  # узел для 2*x1
    
    sum_node = Node("func", 2, sum_, [pow2_node, mult_node], requires_grad=False)  # узел для x0^2 + 2*x1
    
    # Создаем дерево с корнем в узле sum_node
    tree = Tree(sum_node, requires_grad=False)
    
    print("Создано дерево для функции f(x0, x1) = x0^2 + 2*x1")
    print("Представление дерева в виде строки:", tree)
    print("Математическое выражение:", tree.to_math_expr())
    
    return tree

def create_tree_identity():
    """
    Создает дерево для функции f(x0) = x0
    """
    print("\nСоздание дерева для функции f(x0) = x0")
    
    # Создаем узел переменной x0
    # Для узла переменной: t_name="var", parity=0, func=None, data=[индекс_переменной]
    var_node = Node("var", 0, None, [0], requires_grad=False)
    
    # Создаем узел функции идентичности
    # Для узла функции: t_name="func", parity=1 (унарная), func=ident_, data=[child_node]
    ident_node = Node("func", 1, ident_, [var_node], requires_grad=False)
    
    # Создаем дерево
    tree = Tree(ident_node, requires_grad=False)
    
    print("Строковое представление дерева:", str(tree))
    print("Математическое выражение:", tree.to_math_expr())
    
    # Печатаем подробную информацию для отладки
    print("Тип корневого узла:", tree.start_node.t_name)
    print("Функция корневого узла:", tree.start_node.func.__name__ if tree.start_node.func else "None")
    print("Данные корневого узла:", tree.start_node.data)
    
    return tree

def batch_forward(tree, batch_inputs):
    """
    Обрабатывает батч данных с использованием numpy_forward.
    Эта функция заменяет оригинальную batch_forward, которая не работала корректно
    из-за проблемы в обработке батча в tree.forward().
    
    Args:
        tree: Модель Tree
        batch_inputs: Батч входных данных размерности [batch_size, input_dims]
        
    Returns:
        np.ndarray: Массив результатов размерности [batch_size, 1]
    """
    # Используем нашу numpy-реализацию для вычисления
    results = numpy_forward(tree, batch_inputs)
    
    print("\nОтладка batch_forward с numpy:")
    print(f"  Форма входных данных: {to_numpy(batch_inputs).shape}")
    print(f"  Результаты: {results}")
    
    return results

def compute_expected_result(tree, inputs):
    """
    Вычисляет ожидаемый результат напрямую для известных формул.
    
    Args:
        tree: Дерево выражения
        inputs: Входные данные
        
    Returns:
        numpy.ndarray: Ожидаемый результат
    """
    inputs_np = to_numpy(inputs)
    
    # Обеспечим, что inputs_np имеет размерность [batch_size, n_features]
    if len(inputs_np.shape) == 1:
        inputs_np = inputs_np.reshape(1, -1)
    
    # Получаем информацию о корневом узле
    root_node = tree.start_node
    root_type = root_node.t_name
    
    # Подготовим массив для результатов
    batch_size = inputs_np.shape[0]
    results = np.zeros((batch_size, 1))
    
    if root_type == "func":
        func_name = root_node.func.__name__
        
        if func_name == "ident_":
            # f(x0) = x0
            for i in range(batch_size):
                results[i, 0] = inputs_np[i, 0]
        
        elif func_name == "sum_":
            # Проверяем первого потомка
            data0 = root_node.data[0]
            
            if data0.t_name == "func" and data0.func.__name__ == "pow2_":
                # Первый аргумент - квадрат
                
                # Проверяем второго потомка
                data1 = root_node.data[1]
                
                if inputs_np.shape[1] == 1:
                    # f(x0) = x0^2 + x0
                    if data1.t_name == "var":
                        for i in range(batch_size):
                            results[i, 0] = inputs_np[i, 0]**2 + inputs_np[i, 0]
                
                elif inputs_np.shape[1] == 2:
                    if data1.t_name == "func" and data1.func.__name__ == "pow2_":
                        # f(x0, x1) = x0^2 + x1^2
                        for i in range(batch_size):
                            results[i, 0] = inputs_np[i, 0]**2 + inputs_np[i, 1]**2
                    elif data1.t_name == "func" and data1.func.__name__ == "mult_":
                        # f(x0, x1) = x0^2 + 2*x1
                        for i in range(batch_size):
                            results[i, 0] = inputs_np[i, 0]**2 + 2*inputs_np[i, 1]
    
    return results

def test_single_input(tree, inputs):
    """
    Тестирует дерево на одиночном входе
    """
    print("\nТест на одиночном входе:", inputs)
    
    # Подготавливаем входные данные
    prepared_inputs = prepare_input(inputs)
    
    # Вычисляем результат через Tree.forward
    result = tree.forward(prepared_inputs)
    result_np = to_numpy(result)[0, 0]  # Преобразуем тензор в скаляр
    
    # Вычисляем результат через numpy_forward
    result_numpy = numpy_forward(tree, inputs)[0, 0]
    
    # Вычисляем ожидаемый результат напрямую
    expected_formula = compute_expected_result(tree, inputs)[0, 0]
    
    print("Входные данные:", inputs)
    print(f"Результат через forward: {result_np}")
    print(f"Результат через numpy_forward: {result_numpy}")
    print(f"Ожидаемый результат через формулу: {expected_formula}")
    
    # Получаем и выводим математическое выражение дерева
    expr = tree.to_math_expr()
    print(f"Математическое выражение: {expr}")
    
    # Преобразуем входные данные в numpy
    inputs_np = to_numpy(inputs)
    
    # Вычисляем ожидаемый результат вручную в зависимости от выражения
    print("Ожидаемый результат:", end=" ")
    
    # На основе типа корневого узла и математического выражения
    root_node = tree.start_node
    root_type = root_node.t_name
    
    if root_type == "func":
        # Проверяем, какая именно функция используется в корне
        func_name = root_node.func.__name__
        
        if func_name == "ident_":
            # Функция идентичности f(x0) = x0
            if len(inputs_np.shape) == 1 and inputs_np.shape[0] >= 1:
                expected = inputs_np[0]
                print(f"{expected} (идентичность)")
        
        elif func_name == "sum_":
            # Функция суммы
            if len(root_node.data) >= 2:
                # Проверяем первого потомка
                data0 = root_node.data[0]
                
                if data0.t_name == "func" and data0.func.__name__ == "pow2_":
                    # Первый аргумент - квадрат
                    
                    # Проверяем второго потомка
                    data1 = root_node.data[1]
                    
                    if len(inputs_np.shape) == 1 and inputs_np.shape[0] == 1:
                        # f(x0) = x0^2 + x0
                        if data1.t_name == "var":
                            expected = inputs_np[0]**2 + inputs_np[0]
                            print(f"{expected} (x0^2 + x0)")
                    
                    elif len(inputs_np.shape) == 1 and inputs_np.shape[0] == 2:
                        if data1.t_name == "func" and data1.func.__name__ == "pow2_":
                            # f(x0, x1) = x0^2 + x1^2
                            expected = inputs_np[0]**2 + inputs_np[1]**2
                            print(f"{expected} (x0^2 + x1^2)")
                        elif data1.t_name == "func" and data1.func.__name__ == "mult_":
                            # f(x0, x1) = x0^2 + 2*x1
                            expected = inputs_np[0]**2 + 2*inputs_np[1]
                            print(f"{expected} (x0^2 + 2*x1)")
                        else:
                            print("неизвестное выражение")
                    else:
                        print("неизвестное количество переменных")
                else:
                    print("неизвестное выражение")
            else:
                print("неизвестное выражение")
        
        else:
            print(f"неизвестная функция: {func_name}")
    
    elif root_type == "var":
        # Простая переменная f(x) = x
        var_idx = root_node.data[0]
        expected = inputs_np[var_idx]
        print(f"{expected} (переменная x{var_idx})")
    
    else:
        print(f"неизвестный тип узла: {root_type}")
    
    return result

def test_batch_input(tree, batch_inputs):
    """
    Тестирует дерево на батче входных данных
    """
    print("\nТест на батче входных данных:")
    
    # Подготавливаем входные данные
    prepared_inputs = prepare_input(batch_inputs)
    print(f"Подготовленные входные данные (shape: {prepared_inputs.shape}):")
    print(prepared_inputs)
    
    # Вычисляем результат стандартным способом (через forward)
    results_standard = tree.forward(prepared_inputs)
    results_standard_np = to_numpy(results_standard)
    print(f"Результаты через forward (shape: {results_standard_np.shape}):")
    print(results_standard_np)
    
    # Вычисляем результат с помощью numpy_forward
    results_numpy = numpy_forward(tree, batch_inputs)
    print(f"Результаты через numpy_forward (shape: {results_numpy.shape}):")
    print(results_numpy)
    
    # Вычисляем ожидаемый результат через формулу
    results_formula = compute_expected_result(tree, batch_inputs)
    print(f"Ожидаемые результаты через формулу (shape: {results_formula.shape}):")
    print(results_formula)
    
    print("Входные данные (батч):")
    for i, inputs in enumerate(batch_inputs):
        print(f"  Точка {i+1}: {inputs}")
    
    print("Результаты через numpy_forward:")
    for i, result in enumerate(results_numpy):
        print(f"  Результат {i+1}: {result[0]}")
    
    # Получаем и выводим математическое выражение дерева
    expr = tree.to_math_expr()
    print(f"Математическое выражение: {expr}")
    
    # Преобразуем входные данные в numpy
    batch_inputs_np = to_numpy(batch_inputs)
    
    # Вычисляем ожидаемые результаты вручную в зависимости от выражения
    print("Ожидаемые результаты:")
    
    # На основе типа корневого узла и математического выражения
    root_node = tree.start_node
    root_type = root_node.t_name
    
    if root_type == "func":
        # Проверяем, какая именно функция используется в корне
        func_name = root_node.func.__name__
        
        if func_name == "ident_":
            # Функция идентичности f(x0) = x0
            for i, inputs in enumerate(batch_inputs_np):
                if inputs.shape[0] >= 1:
                    expected = inputs[0]
                    print(f"  Ожидаемый {i+1}: {expected} (идентичность)")
        
        elif func_name == "sum_":
            # Функция суммы
            if len(root_node.data) >= 2:
                # Проверяем первого потомка
                data0 = root_node.data[0]
                
                if data0.t_name == "func" and data0.func.__name__ == "pow2_":
                    # Первый аргумент - квадрат
                    
                    # Проверяем второго потомка
                    data1 = root_node.data[1]
                    
                    if batch_inputs_np.shape[1] == 1:
                        # f(x0) = x0^2 + x0
                        if data1.t_name == "var":
                            for i, inputs in enumerate(batch_inputs_np):
                                expected = inputs[0]**2 + inputs[0]
                                print(f"  Ожидаемый {i+1}: {expected} (x0^2 + x0)")
                    
                    elif batch_inputs_np.shape[1] == 2:
                        if data1.t_name == "func" and data1.func.__name__ == "pow2_":
                            # f(x0, x1) = x0^2 + x1^2
                            for i, inputs in enumerate(batch_inputs_np):
                                expected = inputs[0]**2 + inputs[1]**2
                                print(f"  Ожидаемый {i+1}: {expected} (x0^2 + x1^2)")
                        elif data1.t_name == "func" and data1.func.__name__ == "mult_":
                            # f(x0, x1) = x0^2 + 2*x1
                            for i, inputs in enumerate(batch_inputs_np):
                                expected = inputs[0]**2 + 2*inputs[1]
                                print(f"  Ожидаемый {i+1}: {expected} (x0^2 + 2*x1)")
                        else:
                            print("  Неизвестное выражение")
                    else:
                        print("  Неизвестное количество переменных")
                else:
                    print("  Неизвестное выражение")
            else:
                print("  Неизвестное выражение")
        
        else:
            print(f"  Неизвестная функция: {func_name}")
    
    elif root_type == "var":
        # Простая переменная f(x) = x
        var_idx = root_node.data[0]
        for i, inputs in enumerate(batch_inputs_np):
            expected = inputs[var_idx]
            print(f"  Ожидаемый {i+1}: {expected} (переменная x{var_idx})")
    
    else:
        print(f"  Неизвестный тип узла: {root_type}")
    
    # Сравниваем результаты numpy_forward с ожидаемыми
    print("\nСравнение результатов numpy_forward с ожидаемыми:")
    for i, result in enumerate(results_numpy):
        inputs = batch_inputs_np[i]
        actual = result[0]
        
        if root_type == "func" and func_name == "ident_" and inputs.shape[0] >= 1:
            expected = inputs[0]
        elif root_type == "func" and func_name == "sum_" and len(root_node.data) >= 2:
            data0 = root_node.data[0]
            data1 = root_node.data[1]
            
            if data0.t_name == "func" and data0.func.__name__ == "pow2_":
                if inputs.shape[0] == 1:
                    expected = inputs[0]**2 + inputs[0]
                elif inputs.shape[0] == 2:
                    if data1.t_name == "func" and data1.func.__name__ == "pow2_":
                        expected = inputs[0]**2 + inputs[1]**2
                    elif data1.t_name == "func" and data1.func.__name__ == "mult_":
                        expected = inputs[0]**2 + 2*inputs[1]
                    else:
                        expected = None
                else:
                    expected = None
            else:
                expected = None
        elif root_type == "var":
            var_idx = root_node.data[0]
            expected = inputs[var_idx]
        else:
            expected = None
        
        if expected is not None:
            print(f"  Строка {i+1}: Ожидаемый: {expected}, Фактический: {actual}, {'СОВПАДАЕТ' if np.isclose(expected, actual) else 'НЕ СОВПАДАЕТ'}")
        else:
            print(f"  Строка {i+1}: Ожидаемый: не определен, Фактический: {actual}")
    
    return results_numpy

def main():
    """
    Основная функция для тестирования деревьев выражений.
    
    ВАЖНО: В ходе тестирования выявлено, что метод Tree.forward некорректно обрабатывает
    батч входных данных, возвращая только один результат для первого элемента.
    
    Мы реализовали альтернативный метод numpy_forward, который корректно обрабатывает
    батч и возвращает результаты для всех элементов. Рекомендуем использовать его до
    исправления проблемы в основном методе.
    """
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ ДЕРЕВЬЕВ ВЫРАЖЕНИЙ")
    print("=" * 50)
    
    # Создаем и тестируем дерево для идентичности
    id_tree = create_tree_identity()
    # Тестируем на одиночном входе
    test_single_input(id_tree, [2.0])
    
    # Тестируем функцию поэлементной обработки
    print("\nПроверка поэлементной обработки батча для идентичности:")
    batch_inputs = np.array([[2.0], [8.0], [25.0]])
    results = batch_forward(id_tree, batch_inputs)
    for i, (inp, res) in enumerate(zip(batch_inputs, results)):
        print(f"  Вход {i+1}: {inp[0]}, Результат: {res[0]}")
    
    # Тестируем на батче входных данных
    test_batch_input(id_tree, batch_inputs)
    
    # Создаем и тестируем дерево для одной переменной
    tree_1var = create_tree_1var()
    # Тестируем на одиночном входе
    test_single_input(tree_1var, [2.0])
    # Тестируем на батче входных данных
    test_batch_input(tree_1var, np.array([[2.0], [3.0], [4.0]]))
    
    # Создаем и тестируем дерево для двух переменных
    tree_2var = create_tree_2var()
    # Тестируем на одиночном входе
    test_single_input(tree_2var, [1.0, 2.0])
    # Тестируем на батче входных данных
    test_batch_input(tree_2var, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    
    # Создаем и тестируем сложное дерево для двух переменных
    tree_2var_complex = create_tree_2var_complex()
    # Тестируем на одиночном входе
    test_single_input(tree_2var_complex, [1.0, 2.0])
    # Тестируем на батче входных данных
    test_batch_input(tree_2var_complex, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    
    print("\n" + "=" * 50)
    print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("=" * 50)
    print("1. Обнаружена проблема в Tree.forward: некорректная обработка батча входных данных.")
    print("2. Метод Tree.forward всегда возвращает результат только для первого элемента батча.")
    print("3. Наш метод numpy_forward корректно обрабатывает батч и возвращает результаты для всех элементов.")
    print("4. Рекомендуем исправить метод Node.eval_node в файле sr_tree.py для корректной обработки батча.")
    print("   или использовать предложенный нами метод numpy_forward.")

if __name__ == "__main__":
    main()
