import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from nodes import *
from typing import List, Union, Optional, Dict

# global min_loss
min_loss = float('inf')

NUM_OF_EPOCHS = 100
NUM_OF_MODELS = 70
MAX_DEPTH = 10
MUTATION_PROB = 0.25 # 0.15
INV_ERROR_COEF = 0.7

np.random.seed(2)

def get_deepcopy(obj):
    return copy.deepcopy(obj)

class Node:
    def __init__(self, t_name: str, parity: int, func: callable, data: Union[List, torch.Tensor, int],
                 requires_grad: bool = False):
        self.t_name = t_name   # type
        self.parity = parity   # parity of a func or an operator
        self.func = func       # func or operator
        self.data = data       # [child node/s] for Node, [value] for idents (consts), [var index] for variables
        self.node_num = -1
        self.num_vars = None   # Will be set when evaluating
        self.requires_grad = requires_grad
        self._cached_output = None  # Cache for forward pass
        self._cached_grad = None    # Cache for backward pass
        self.domain_penalty = 0
        # Get function info if available
        self.func_info = get_function_info(func.__name__) if hasattr(func, '__name__') else None

    def __str__(self):
        return f"{self.t_name} node: parity: {self.parity}, func: {self.func.__name__}"

    def validate_var_index(self, var_idx: int, num_vars: int) -> bool:
        """Validate that variable index is within bounds"""
        if var_idx >= num_vars:
            raise ValueError(f"Variable index {var_idx} is out of bounds for {num_vars} variables")
        return True

    def eval_node(self, varval=None, cache: bool = True):
        """Evaluate node with PyTorch tensors and optional gradient computation"""
        if varval is not None:
            if isinstance(varval, np.ndarray):
                varval = to_tensor(varval, requires_grad=self.requires_grad)
            if varval.dim() == 1:
                varval = varval.reshape(-1, 1)
            self.num_vars = varval.shape[1]
        
        if self.t_name == "ident":
            result = to_tensor(self.data[0] if isinstance(self.data, (list, np.ndarray)) else self.data,
                             requires_grad=False)  # Constants don't need gradients
            result = result.expand(varval.shape[0])
        elif self.t_name == "var":
            if varval is None:
                raise ValueError("Variable values must be provided for variable nodes")
            
            var_idx = self.data[0] if isinstance(self.data, list) else self.data
            self.validate_var_index(var_idx, self.num_vars)
            
            result = varval[:, var_idx].clone()  # Clone to ensure we have a new tensor
            if self.requires_grad:
                result.requires_grad_(True)
        else:
            if self.parity == 1:
                child_result = self.data[0].eval_node(varval=varval, cache=cache)
                # Check domain constraints for unary functions
                if isinstance(child_result, torch.Tensor) and self.func_info is not None:
                    mask_min = child_result < self.func_info.domain_min
                    mask_max = child_result > self.func_info.domain_max
                    if mask_min.any() or mask_max.any():
                        self.domain_penalty += torch.sum(torch.abs(torch.where(mask_min, self.func_info.domain_min - child_result, 0))) + \
                                            torch.sum(torch.abs(torch.where(mask_max, child_result - self.func_info.domain_max, 0)))
                result = self.func(child_result)
            else:
                in1 = self.data[0].eval_node(varval=varval, cache=cache)
                in2 = self.data[1].eval_node(varval=varval, cache=cache)
                # Check domain constraints for binary functions
                if isinstance(in1, torch.Tensor) and isinstance(in2, torch.Tensor) and self.func_info is not None:
                    if hasattr(self.func_info, 'domain_min'):
                        mask_min = (in1 < self.func_info.domain_min) | (in2 < self.func_info.domain_min)
                        if mask_min.any():
                            self.domain_penalty += torch.sum(torch.abs(torch.where(mask_min, self.func_info.domain_min - torch.minimum(in1, in2), 0)))
                    if hasattr(self.func_info, 'domain_max'):
                        mask_max = (in1 > self.func_info.domain_max) | (in2 > self.func_info.domain_max)
                        if mask_max.any():
                            self.domain_penalty += torch.sum(torch.abs(torch.where(mask_max, torch.maximum(in1, in2) - self.func_info.domain_max, 0)))
                result = self.func(in1, in2)

        if cache:
            self._cached_output = result
        return result

    def get_total_domain_penalty(self) -> float:
        """Recursively collect domain penalties from all nodes"""
        total_penalty = self.domain_penalty
        if self.t_name not in ("var", "ident"):
            for child in self.data:
                if isinstance(child, Node):
                    total_penalty += child.get_total_domain_penalty()
        return total_penalty

    def reset_domain_penalties(self):
        """Recursively reset domain penalties for all nodes"""
        self.domain_penalty = 0
        if self.t_name not in ("var", "ident"):
            for child in self.data:
                if isinstance(child, Node):
                    child.reset_domain_penalties()

    # def get_gradients(self) -> Dict[int, torch.Tensor]:
    #     """Get gradients for all variable nodes in the subtree"""
    #     gradients = {}
    #     if self.t_name == "var" and self.requires_grad:
    #         var_idx = self.data[0] if isinstance(self.data, list) else self.data
    #         if self._cached_output is not None and self._cached_output.grad is not None:
    #             gradients[var_idx] = self._cached_output.grad.clone()
    #     elif self.t_name not in ("ident",):
    #         for child in self.data:
    #             if isinstance(child, Node):
    #                 child_grads = child.get_gradients()
    #                 for idx, grad in child_grads.items():
    #                     if idx in gradients and grad is not None:
    #                         gradients[idx] = gradients[idx] + grad
    #                     else:
    #                         gradients[idx] = grad
    #     return gradients


class Tree:
    def __init__(self, start_node: Node, max_depth: int = 10, mutation_prob: float = 0.15,
                 crossover_prob: float = 0.5, requires_grad: bool = False):
        self.start_node = start_node
        self.max_depth = max_depth
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.max_num_of_node = 0
        self.error = 0
        self.num_vars = None  # Will be set during evaluation
        self.requires_grad = requires_grad
        self._set_requires_grad(requires_grad)
        self.grad = None
        self.enumerate_tree()

    def _set_requires_grad(self, requires_grad: bool):
        """Recursively set requires_grad for all nodes"""
        def _set_node_grad(node: Node):
            node.requires_grad = requires_grad
            if node.t_name not in ("var", "ident"):
                for child in node.data:
                    if isinstance(child, Node):
                        _set_node_grad(child)
        _set_node_grad(self.start_node)

    def __str__(self):
        return f"start: ({self.start_node}), max num: {self.max_num_of_node}, error: {self.error}"

    def get_tree_depth(self, node: Node) -> int:
        """Calculate depth of tree/subtree starting at node"""
        if node.t_name in ("var", "ident"):
            return 0
        return 1 + max(self.get_tree_depth(child) for child in node.data if isinstance(child, Node))

    def validate_tree_depth(self) -> bool:
        """Check if tree depth is within limits"""
        depth = self.get_tree_depth(self.start_node)
        return depth <= self.max_depth

    def get_var_indices(self, node: Optional[Node] = None) -> List[int]:
        """Get all variable indices used in the tree"""
        if node is None:
            node = self.start_node
            
        indices = []
        if node.t_name == "var":
            indices.append(node.data[0] if isinstance(node.data, list) else node.data)
        elif node.t_name not in ("ident",):
            for child in node.data:
                if isinstance(child, Node):
                    indices.extend(self.get_var_indices(child))
        return sorted(list(set(indices)))

    def validate_var_indices(self, num_vars: int) -> bool:
        """Validate all variable indices in the tree"""
        indices = self.get_var_indices()
        return all(0 <= idx < num_vars for idx in indices)

    def enumerate_tree(self):
        cur_num = 0
        to_enum = [self.start_node]
        while len(to_enum) > 0:
            for el in to_enum:
                el.node_num = cur_num
                cur_num += 1
                if el.t_name not in ("var", "ident"):
                    for d in el.data:
                        to_enum.append(d)
                to_enum.pop(to_enum.index(el))
        self.max_num_of_node = cur_num

    def get_node_by_num(self, num, cp=False):
        to_check = [self.start_node]
        while len(to_check) > 0:
            for el in to_check:
                if el.node_num == num:
                    if not cp:
                        return el
                    else:
                        return safe_deepcopy(el)
                if el.t_name not in ("var", "ident"):
                    for d in el.data:
                        to_check.append(d)
                to_check.pop(to_check.index(el))
        print("ERROR: not found")

    def change_node(self, old_node_num, new_node):
        to_check = [self.start_node]
        while len(to_check) > 0:
            for el in to_check:
                if el.node_num == old_node_num:
                    el.t_name = new_node.t_name
                    el.parity = new_node.parity
                    el.func = new_node.func
                    el.data = new_node.data
                    el.node_num = -1
                    self.enumerate_tree()
                    return 
                if el.t_name not in ("var", "ident"):
                    for d in el.data:
                        to_check.append(d)
                to_check.pop(to_check.index(el))

    def forward_with_gradients(self, varval: torch.Tensor, cache: bool = True):
        """
        Выполняет прямой проход для всего входного тензора и вычисляет градиенты для каждого входного вектора.

        Args:
            varval: Входной тензор размерности (N, num_vars), где N — количество входных векторов.

        Returns:
            outputs: Тензор размерности (N,), содержащий скалярные выходы для каждого входного вектора.
            gradients: Тензор размерности (N, num_vars), содержащий градиенты по каждому входному вектору.
        """
        # print("varval.shape:", varval.shape)

        # Функция для обработки каждого входного вектора отдельно
        def full_forward(input_tensor):
            return self.start_node.eval_node(varval=input_tensor, cache=cache).squeeze(-1)

        # PyTorch считает, что входной тензор влияет на все выходные элементы. Чтобы избежать этого:
        outputs = full_forward(varval)  # (100,)
        gradients = torch.autograd.functional.vjp(
            full_forward, varval, v=torch.ones_like(outputs), create_graph=self.requires_grad
        )[1]  # Извлекаем градиенты (вектор производных)
        # print("outputs:", outputs)
        # print("gradients.shape:", gradients.shape)

        self.grad = gradients  # Сохраняем градиенты для отладки
        # print("math expr:", self.to_math_expr())
        # print("self.grad:", self.grad)
        # exit()
        return outputs

    def forward(self, varval=None, cache: bool = True):
        if self.requires_grad:
            return self.forward_with_gradients(varval=varval, cache=cache)
        return self.start_node.eval_node(varval=varval, cache=cache)

    def eval_tree_error(self, data: Union[np.ndarray, torch.Tensor], inv_error: bool = False):
        """Evaluate tree error with PyTorch tensors"""
        if isinstance(data, np.ndarray):
            data = to_tensor(data, requires_grad=self.requires_grad)
        
        X = data[:, :-1]
        y = data[:, -1]
        
        # Reset cached outputs and domain penalties
        reset_cache(self.start_node)
        self.start_node.reset_domain_penalties()
        
        # Forward pass
        y_aprox = self.forward(varval=X, cache=True)
        
        # Calculate MSE
        loss = calculate_mse(y_aprox, y)
        global min_loss
        if loss.item() < min_loss:
            min_loss = loss.item()
            print("min_loss:", min_loss)
        
        # Get total domain penalty
        domain_penalty = self.start_node.get_total_domain_penalty()
        
        # Normalize domain penalty by the number of samples
        domain_penalty = domain_penalty / len(y) if len(y) > 0 else domain_penalty

        if inv_error:
            inv_loss = self.eval_inv_error(data[np.random.choice(len(data), 100, replace=False)])
        else:
            inv_loss = 0
        self.error = loss.item() + 10*self.max_num_of_node + inv_loss + domain_penalty

        # if self.requires_grad:
        #     inv_loss = self.eval_inv_error(data)
        #     # print("loss:", loss.item())
        #     # print("inv_loss:", inv_loss)
        #     # print("domain_penalty:", domain_penalty)
        #     # print("max_num_of_node:", self.max_num_of_node)
        #     self.error = loss.item() + 10*self.max_num_of_node + inv_loss + domain_penalty
        # else:
        #     # print("loss:", loss.item())
        #     # print("domain_penalty:", domain_penalty)
        #     # print("max_num_of_node:", self.max_num_of_node)
        #     self.error = loss.item() + 10*self.max_num_of_node + domain_penalty

    def wrapped_tree_func(self, x: np.ndarray, target_y: float) -> float:
        """Wrapper function for optimization that computes squared difference between tree output and target."""
        # Convert input to float64 for optimization
        x = x.astype(np.float64)
        x_tensor = to_tensor(x.reshape(1, -1), requires_grad=self.requires_grad)
        y_pred = self.forward(varval=x_tensor).detach().numpy().astype(np.float64)
        return float((y_pred - target_y) ** 2)  # Ensure output is float64

    def eval_inv_error(self, data: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Evaluate inverse error using scipy.optimize.minimize.
        For each point in the dataset, tries to find x such that f(x) = y.
        """
        import scipy.optimize as opt

        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()

        # Convert data to float64
        X = data[:, :-1].astype(np.float64)
        y = data[:, -1].astype(np.float64)

        # Normalize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean()) / y.std()

        total_inv_error = 0.0
        n_samples = len(y)

        for i in range(n_samples):
            # Use current X as initial guess
            x0 = X[i].copy()  # Make sure we have a contiguous array
            target_y = float(y[i])  # Convert to float64

            # Find x that minimizes (f(x) - y)^2
            try:
                result = opt.minimize(
                    self.wrapped_tree_func,
                    x0=x0,
                    args=(target_y,),
                    # method='Powell',
                    method='BFGS',
                    # method='Nelder-Mead', # долго
                    # method='SLSQP', # первый метод
                    options={'maxiter': 10}
                    # options={'maxiter': 100, 'ftol': 1e-6}
                )

                if result.success:
                    # Compute error between found x and original x
                    x_found = result.x
                    # print("iters:", result.nit)
                    error = float(np.sum((x_found - x0) ** 2))
                    total_inv_error += error
                else:
                    # Penalize failed inversions heavily
                    total_inv_error += 1e6
            except Exception as e:
                print(f"Optimization failed: {str(e)}")
                total_inv_error += 1e6

        return float(total_inv_error / n_samples)
    
    # def eval_inv_error(self, data: Union[np.ndarray, torch.Tensor], tol=1e-6, max_iters=100):
    #     """Evaluate tree inversion error with PyTorch tensors."""
    #     if isinstance(data, np.ndarray):
    #         data = to_tensor(data, requires_grad=self.requires_grad)

    #     X = data[:, :-1]
    #     y = data[:, -1]

    #     # Normalize data (testing)
    #     X = (X - X.mean(axis=0)) / X.std(axis=0)
    #     y = (y - y.mean()) / y.std()

    #     xk = X.clone()
    #     alpha_tensor = 1.0 * torch.ones_like(y)
    #     y_last = self.forward(varval=xk, cache=True)

    #     for i in range(max_iters):
    #         xk_last = xk.clone()
    #         y_aprox = self.forward(varval=xk, cache=True)

    #         # Проверяем деление на малые градиенты
    #         grad = self.grad.clamp(min=1e-3, max=1e3)

    #         # Адаптивное демпфирование шага
    #         mask = torch.abs(y_aprox - y) > torch.abs(y_last - y)
    #         # print("mask =", mask)
    #         alpha_tensor[mask] *= 0.9  # Уменьшаем шаг
    #         alpha_tensor[~mask] = torch.min(alpha_tensor[~mask] * 1.1, torch.tensor(1.0))  # Увеличиваем шаг

    #         y_last = y_aprox
    #         # xk = xk - alpha_tensor * (y_aprox - y).unsqueeze(-1) / grad
    #         xk = xk - alpha_tensor.unsqueeze(-1) * (y_aprox - y).unsqueeze(-1) / grad

    #         # Условия остановки
    #         if (torch.sum((xk - xk_last)**2) < tol and torch.max(torch.abs(y_aprox - y)) < tol):
    #             break

    #     # Оцениваем среднеквадратичную ошибку инверсии
    #     inv_mse = calculate_mse(xk, X)
    #     return inv_mse.item()

    # def get_gradients(self) -> Dict[int, torch.Tensor]:
    #     """Get gradients for all variable nodes in the tree"""
    #     if not self.requires_grad:
    #         return {}
    #     return self.start_node.get_gradients()

    def print_tree(self):
        """Print a visual representation of the tree using ASCII characters"""
        def _build_str(node, level=0, prefix="Root: ", is_last=True):
            if not isinstance(node, Node):
                if type(node) is int:
                    return prefix + "x" + str(node) + "\n"
                return prefix + str(node) + "\n"

            result = prefix + node.func.__name__ + str(node.node_num) + "\n"
            
            if node.data:
                prefix_cur = "    " * level
                for i, child in enumerate(node.data):
                    is_last_child = i == len(node.data) - 1
                    if is_last_child:
                        result += prefix_cur + "└── " + _build_str(child, level + 1, "", is_last_child)
                    else:
                        result += prefix_cur + "├── " + _build_str(child, level + 1, "", is_last_child)
            return result

        print(self)
        return _build_str(self.start_node)

    def to_math_expr(self):
        """Convert the tree to a mathematical expression string"""
        def _build_expr(node):
            if not isinstance(node, Node):
                if type(node) is int:
                    return f"x{node}"
                return str(node)
            
            if node.parity == 1:
                inner = _build_expr(node.data[0])
                if node.func.__name__ in ['exp_', 'sin_', 'cos_']:
                    return f"{node.func.__name__[:-1]}({inner})"
                return inner
            else:
                left = _build_expr(node.data[0])
                right = _build_expr(node.data[1])
                
                # Handle special cases for better readability
                if node.func.__name__ == 'sum_':
                    return f"({left} + {right})"
                elif node.func.__name__ == 'mult_':
                    return f"({left} * {right})"
                elif node.func.__name__ == 'div_':
                    return f"({left} / {right})"
                elif node.func.__name__ == 'pow_':
                    return f"({left} ^ {right})"
                else:
                    return f"{node.func.__name__[:-1]}({left}, {right})"
                
        return _build_expr(self.start_node)


def reset_cache(node):
    """Reset cached tensors in a node and its children"""
    if node._cached_output is not None:
        # Safely detach tensor if it exists
        if torch.is_tensor(node._cached_output):
            node._cached_output = node._cached_output.detach().clone()
        node._cached_output = None
    
    if node.t_name not in ("var", "ident"):
        for child in node.data:
            if isinstance(child, Node):
                reset_cache(child)

def safe_deepcopy(obj):
    """Safely deepcopy an object containing PyTorch tensors"""
    if isinstance(obj, Node):
        new_node = copy.copy(obj)  # Shallow copy first
        if isinstance(obj.data, list):
            new_node.data = [safe_deepcopy(item) for item in obj.data]
        elif torch.is_tensor(obj.data):
            # Detach and clone tensor to create a new leaf tensor
            new_node.data = obj.data.detach().clone()
        else:
            new_node.data = copy.deepcopy(obj.data)
        # Copy function info without deepcopy
        new_node.func_info = obj.func_info
        new_node._cached_output = None  # Reset cache
        new_node.domain_penalty = 0  # Reset penalty
        return new_node
    elif isinstance(obj, Tree):
        new_tree = copy.copy(obj)  # Shallow copy first
        new_tree.start_node = safe_deepcopy(obj.start_node)
        new_tree.grad = None  # Reset gradient
        return new_tree
    elif isinstance(obj, list):
        return [safe_deepcopy(item) for item in obj]
    elif torch.is_tensor(obj):
        # Always detach and clone tensors
        return obj.detach().clone()
    else:
        return copy.deepcopy(obj)

def crossover(p1: Tree, p2: Tree) -> Tree:
    """Combine parent1 with a node from parent2 with multivariable support"""
    child = safe_deepcopy(p1)
    
    # Get valid node indices (exclude variable nodes if they would cause invalid indices)
    p1_valid_indices = [i for i in range(p1.max_num_of_node)]
    p2_valid_indices = [i for i in range(p2.max_num_of_node)]
    
    # Try to find valid crossover points
    max_attempts = 10
    for _ in range(max_attempts):
        idx_p1 = np.random.choice(p1_valid_indices)
        idx_p2 = np.random.choice(p2_valid_indices)
        
        donor_node = p2.get_node_by_num(idx_p2, cp=True)
        temp_child = safe_deepcopy(child)
        temp_child.change_node(idx_p1, donor_node)
        
        # Validate the resulting tree
        if temp_child.validate_tree_depth():
            if p1.num_vars is not None:  # If we know the number of variables
                if temp_child.validate_var_indices(p1.num_vars):
                    return temp_child
            else:
                return temp_child
    
    # If no valid crossover found, return copy of parent1
    return child

parity1 = [const_, ident_, exp_, sin_, cos_]
parity2 = [sum_, mult_, div_, pow_]

def mutation(tree: Tree):
    """Change random node according to its parity with multivariable support"""
    if tree.start_node.t_name not in ("var", "ident"):
        # Get mutable nodes (exclude var and ident nodes)
        mutable_indices = []
        for i in range(tree.max_num_of_node):
            node = tree.get_node_by_num(i)
            if node and node.t_name not in ("var", "ident"):
                mutable_indices.append(i)
        
        if not mutable_indices:
            return
            
        idx = np.random.choice(mutable_indices)
        new_node = tree.get_node_by_num(idx)
        
        # Get appropriate functions based on parity
        if new_node.parity == 1:
            funcs = [func_info.func for func_info in get_functions_by_parity(1) 
                    if func_info.name not in ['const_', 'ident_']]
            if funcs:
                new_node.func = np.random.choice(funcs)
        else:
            funcs = [func_info.func for func_info in get_functions_by_parity(2)]
            if funcs:
                new_node.func = np.random.choice(funcs)

def calculate_mse(output: torch.Tensor, example: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Squared Error using PyTorch operations
    Args:
        output: Predicted values
        example: True values
    Returns:
        MSE value
    """
    NAN_PUNISHMENT = 1e10

    min_val = torch.min(example)
    max_val = torch.max(example)
    diff = torch.abs(max_val - min_val)
    if diff == 0:
        diff = torch.tensor(1.0)

    # Normalized MSE
    # error = (1/example.shape[0]) * torch.sum((output - example)**2) / diff
    error = torch.mean((output - example)**2)

    if torch.isnan(error):
        return NAN_PUNISHMENT * torch.isnan(output).sum()
    return error

def get_mean_error(models):
    suma = 0
    for model in models:
        suma += model.error
    return suma/len(models)


def evolution(num_of_epochs, models, training_data):
    """evolution process with crossover and mutation"""
    mean_error_arr = np.zeros(NUM_OF_EPOCHS)
    for epoch in range(num_of_epochs):
        print("EPOCH:", epoch)
        # Create new models list without copying tensors
        models_to_eval = []
        for model in models:
            new_model = safe_deepcopy(model)
            models_to_eval.append(new_model)

        for model in models:
            for i in range(NUM_OF_MODELS//len(models) - 1):
                new_model = crossover(model, models[np.random.randint(0, len(models))])

#TODO: rewrite this part to get rid of max_depth, add regularization for tree depth
                while new_model.max_num_of_node > new_model.max_depth:
                    new_model = crossover(model, models[np.random.randint(0, len(models))])
                models_to_eval.append(new_model)

        for model in models_to_eval:
            # if np.random.random() < model.mutation_prob:
            if np.random.uniform() < model.mutation_prob:
                mutation(model)
            # if epoch >= int(NUM_OF_EPOCHS * INV_ERROR_COEF):
            if epoch%5 == 0:
                model.eval_tree_error(training_data, inv_error=True)
            else:
                model.eval_tree_error(training_data)

        # Sort models by error and select top 10
        models_to_eval.sort(key=lambda x: x.error)
        models = []
        for model in models_to_eval[:10]:
            new_model = safe_deepcopy(model)
            models.append(new_model)

        mean_error = get_mean_error(models)
        mean_error_arr[epoch] = mean_error
        if (epoch + 1) % 10 == 0:
            print(f"EPOCH: {epoch+1}/{NUM_OF_EPOCHS}")
            # print("mean_error:", mean_error)
            # print("best_error:", models[0].error)

    plt.plot(mean_error_arr)
    plt.show()    
    print("min_loss:", min_loss)
    return models[0]