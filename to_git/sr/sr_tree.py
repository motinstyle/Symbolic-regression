import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from nodes import *
from typing import List, Union, Optional, Dict
from parameters import *
import random
import sympy as sp
# global min_rmse_loss for the best model
# min_rmse_loss = float('inf')

# get deepcopy of an object
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
        self.var_count = 0      # Counter for variables under this node
        self.domain_penalty = 0
        
        # Get function info if available
        self.func_info = get_function_info(func.__name__) if hasattr(func, '__name__') else None

        # Initialize constants if required
        if CONST_OPT:
            # Don't add constants for basic operators and const nodes
            if (self.func_info and self.func_info.name not in ['add_', 'sub_', 'mul_', 'div_']):
                # Random initialization of constants in range [-10, 10]
                self.scale = torch.tensor([np.random.uniform(-10, 10)], requires_grad=requires_grad)  # Multiplicative constant
                self.bias = torch.tensor([np.random.uniform(-10, 10)], requires_grad=requires_grad)   # Additive constant
            else:
                self.scale = 1
                self.bias = 0
        else:
            self.scale = 1
            self.bias = 0

    # string representation of the node (type, parity, function name)
    def __str__(self):
        return f"{self.t_name} node: parity: {self.parity}, func: {self.func.__name__}"

    # validate variable index is within bounds (less than num_vars)
    def validate_var_index(self, var_idx: int, num_vars: int) -> bool:
        """Validate that variable index is within bounds"""
        if var_idx >= num_vars:
            raise ValueError(f"Variable index {var_idx} is out of bounds for {num_vars} variables")
        return True

    # evaluate node value for a given input
    def eval_node(self, varval=None, cache: bool = True):
        """Evaluate node with PyTorch tensors and optional gradient computation"""
        if varval is not None:
            if isinstance(varval, np.ndarray):
                varval = to_tensor(varval, requires_grad=self.requires_grad)
            if varval.dim() == 1:
                varval = varval.reshape(-1, 1)
            
        # Return cached result if available
        if cache and self._cached_output is not None:
            return self._cached_output

        # Evaluate based on node type
        if self.t_name == "var":
            if varval is None:
                raise ValueError("Variable values must be provided for variable nodes")
            var_idx = self.data[0]
            self.validate_var_index(var_idx, varval.shape[1])
            result = varval[:, var_idx:var_idx+1]
        elif self.t_name == "ident":
            if isinstance(self.data, (int, float)):
                result = torch.tensor([[float(self.data)]], dtype=torch.float32)
            else:
                result = self.data.reshape(1, -1)
            result = result.repeat(varval.shape[0] if varval is not None else 1, 1)
        else:  # Function node
            if self.parity == 1:
                child_result = self.data[0].eval_node(varval, cache)
                result = self.func(child_result)
            else:
                left_result = self.data[0].eval_node(varval, cache)
                right_result = self.data[1].eval_node(varval, cache)
                result = self.func(left_result, right_result)

        # Apply constants if required
        if CONST_OPT and isinstance(self.scale, torch.Tensor):
            result = self.scale * result + self.bias

        if cache:
            self._cached_output = result
        return result

    # calculate total domain penalty for the node
    def get_total_domain_penalty(self) -> float:
        """Recursively collect domain penalties from all nodes"""
        total_penalty = self.domain_penalty
        if self.t_name not in ("var", "ident"):
            for child in self.data:
                if isinstance(child, Node):
                    total_penalty += child.get_total_domain_penalty()
        return total_penalty

    # reset domain penalties for all nodes for new generation
    def reset_domain_penalties(self):
        """Recursively reset domain penalties for all nodes"""
        self.domain_penalty = 0
        if self.t_name not in ("var", "ident"):
            for child in self.data:
                if isinstance(child, Node):
                    child.reset_domain_penalties()

    # convert node to mathematical expression string
    def to_math_expr(self):
        """Convert node to mathematical expression string"""
        if self.t_name == "var":
            expr = f"x{self.data[0]}"
        elif self.t_name == "ident":
            if isinstance(self.data[0], int):
                expr = f"x{self.data[0]}"
            else:
                expr = str(float(self.data[0]) if isinstance(self.data, torch.Tensor) else self.data)
        else:
            if self.parity == 1:
                child_expr = self.data[0].to_math_expr()
                if self.func_info:
                    func_name = self.func_info.name.replace('_', '')
                    # Special handling for power functions
                    if func_name == 'pow2':
                        expr = f"({child_expr})**2"
                    elif func_name == 'pow3':
                        expr = f"({child_expr})**3"
                    elif func_name == 'neg':
                        expr = f"-{child_expr}"
                    else:
                        expr = f"{func_name}({child_expr})"
                else:
                    # Handle power functions in func.__name__
                    func_name = self.func.__name__.replace('_', '')
                    if func_name == 'pow2':
                        expr = f"({child_expr})**2"
                    elif func_name == 'pow3':
                        expr = f"({child_expr})**3"
                    elif func_name == 'neg':
                        expr = f"-{child_expr}"
                    else:
                        expr = f"{func_name}({child_expr})"
            else:
                left_expr = self.data[0].to_math_expr()
                right_expr = self.data[1].to_math_expr()
                if self.func_info:
                    op = self.func_info.name.replace('_', '')
                    if op == 'sum':
                        expr = f"({left_expr} + {right_expr})"
                    elif op == 'mult':
                        expr = f"{left_expr} * {right_expr}"
                    elif op == 'div':
                        expr = f"({left_expr} / {right_expr})"
                    else:
                        expr = f"{op}({left_expr}, {right_expr})"
                else:
                    expr = f"{self.func.__name__}({left_expr}, {right_expr})"

        # Add constants if required and they exist
        if CONST_OPT:
            scale = self.scale.item()
            bias = self.bias.item()
            if scale != 1 or bias != 0:
                if scale != 1:
                    expr = f"{scale}*({expr})"
                if bias != 0:
                    expr = f"({expr} + {bias})"

        return expr

    # get all constants in the node and its children (for optimization)
    def get_constants(self):
        """Get all constants in the node and its children"""
        constants = []
        if CONST_OPT and isinstance(self.scale, torch.Tensor):
            constants.extend([self.scale, self.bias])
        
        if self.t_name == "func":
            for child in self.data:
                constants.extend(child.get_constants())
        
        return constants

    # simplify the expression by combining constants where possible
    def simplify(self):
        """Simplify the expression by combining constants where possible"""
        if not CONST_OPT:
            return
            
        if self.t_name == "func" and self.func_info:
            # Recursively simplify children first
            for child in self.data:
                child.simplify()
            
            # Handle addition case
            if self.func_info.name == 'add_':
                left, right = self.data
                # If both children have constants, combine them
                if isinstance(left.scale, torch.Tensor) and isinstance(right.scale, torch.Tensor):
                    # Combine like terms
                    self.scale = left.scale + right.scale
                    self.bias = left.bias + right.bias
                    # Reset children's constants
                    left.scale = 1
                    left.bias = 0
                    right.scale = 1
                    right.bias = 0
            #TODO: add other cases for simplification

class Tree:
    def __init__(self,
                 start_node: Node,
                 max_depth: int = MAX_DEPTH, 
                 mutation_prob: float = MUTATION_PROB,
                 crossover_prob: float = 0.5, 
                 requires_grad: bool = False):
        
        self.start_node = start_node
        self.max_depth = max_depth
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        self.error_is_set = False
        
        self.error = 0
        self.forward_loss = 0
        self.inv_loss = 0
        self.domain_loss = 0
        self.max_num_of_node = 0
        self.error = 0
        self.requires_grad = requires_grad
        self._set_requires_grad(requires_grad)
        self.enumerate_tree()
        self.update_var_counts()  # Count variables under each node
        self.num_vars = self.start_node.var_count  # Will be set during evaluation
        # self.str_variables = [f'x{i}' for i in range(self.num_vars)]
        self.symbols = sp.symbols([f'x{i}' for i in range(self.num_vars)])
        self.depth = self.get_tree_depth(self.start_node)  # Store current depth
        
        # Кэшированное математическое выражение
        self.math_expr = self.to_math_expr()

        # NSGA-II
        self.crowding_distance = 0
        self.domination_count = 0
        self.dominated_models = None
        self.is_unique = True

        self.grad = None

    # recursively set requires_grad for all nodes
    def _set_requires_grad(self, requires_grad: bool):
        """Recursively set requires_grad for all nodes"""
        def _set_node_grad(node: Node):
            node.requires_grad = requires_grad
            if node.t_name not in ("var", "ident"):
                for child in node.data:
                    if isinstance(child, Node):
                        _set_node_grad(child)
        _set_node_grad(self.start_node)

    # string representation of the tree (start node, max number of nodes, error)
    def __str__(self):
        return f"start: ({self.start_node}), max_node_num: {self.max_num_of_node}, depth: {self.depth}, error: {self.error}"

    def __repr__(self):
        return self.__str__()

    # calculate depth of tree/subtree starting at node
    def get_tree_depth(self, node: Node) -> int:
        """
        Calculate the depth of the tree starting from the given node,
        treating nodes with zero variables as leaf nodes regardless of structure.
        """
        if node is None:
            return 0
        
        # Variable nodes are leaf nodes
        if node.t_name == "var" or (node.t_name == "ident" and not isinstance(node.data, list)):
            return 0
            
        # If this node has zero variables under it, treat it as a leaf node
        if node.var_count == 0:
            return 0
        
        if self.is_constant_node(node):
            return 0
        
        # For other nodes, get the maximum depth from children
        max_child_depth = 0
        if isinstance(node.data, list):
            for child in node.data:
                child_depth = self.get_tree_depth(child)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth + 1

    def update_var_counts(self):
        """Update the variable counts for all nodes in the tree"""
        def count_vars(node):
            if node is None:
                return 0
            
            # Reset the counter for this node
            node.var_count = 0
            
            # If it's a variable node, count as 1
            if node.t_name == "var":
                node.var_count = 1
                return 1
                
            # For other nodes, sum up from children
            if isinstance(node.data, list):
                for child in node.data:
                    node.var_count += count_vars(child)
            
            return node.var_count
            
        count_vars(self.start_node)

    def update_depth(self):
        """Update the current depth of the tree"""
        self.update_var_counts()  # First update variable counts
        self.depth = self.get_tree_depth(self.start_node)

    def validate_tree_depth(self) -> bool:
        """Check if the tree depth is within the allowed maximum"""
        self.update_depth()
        return self.depth <= self.max_depth

    # get all variable indices used in the tree
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

    # validate all variable indices in the tree
    def validate_var_indices(self, num_vars: int) -> bool:
        """Validate all variable indices in the tree"""
        indices = self.get_var_indices()
        return all(0 <= idx < num_vars for idx in indices)

    # enumerate all nodes in the tree
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

    # get node by node number
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

    # change node with old_node_num to new_node
    def change_node(self, old_node_num, new_node):
        """Change a node in the tree and update tree properties"""
        nodes = []
        def find_and_replace(node):
            # Проверяем, является ли node объектом класса Node
            if not isinstance(node, Node):
                return node  # Если нет, просто возвращаем его без изменений
            
            if node.node_num == old_node_num:
                return new_node
            if isinstance(node.data, list):
                new_children = []
                for child in node.data:
                    new_children.append(find_and_replace(child))
                node.data = new_children
            return node

        self.start_node = find_and_replace(self.start_node)
        self.enumerate_tree()
        self.update_var_counts()  # Update var counts after node change
        self.update_depth()  # Update depth after node change

    # forward pass with gradients if gradient is required (for inverse pass)
    def forward_with_gradients(self, varval: torch.Tensor, cache: bool = True):
        """
        Performs forward pass for the entire input tensor and calculates gradients for each input vector.

        Args:
            varval: Input tensor of shape (N, num_vars), where N is the number of input vectors.
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
        # print("math expr:", self.math_expr)
        # print("self.grad:", self.grad)
        # exit()
        return outputs

    # forward pass (choose variant of forward pass based on requires_grad)
    def forward(self, varval=None, cache: bool = True):
        if self.requires_grad:
            return self.forward_with_gradients(varval=varval, cache=cache)
        return self.start_node.eval_node(varval=varval, cache=cache)

    # optimize constants in the tree using Nelder-Mead method if CONST_OPT is True
    def optimize_constants(self, data: Union[np.ndarray, torch.Tensor], max_iter: int = 500) -> None:
        """Optimize constants in the tree using Nelder-Mead method"""
        if not CONST_OPT:
            return

        # Convert data to numpy if it's a tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        
        # Get all constants from the tree
        constants = self.start_node.get_constants()
        # print("constants:", constants)
        if not constants:
            return
        
        # Convert constants to numpy array for optimization
        initial_params = np.array([const.item() for const in constants])
        n_params = len(initial_params)
        
        # Define optimization objective function
        def objective(params):
            # Update constants in the tree
            param_idx = 0
            def update_constants(node):
                nonlocal param_idx
                if CONST_OPT and isinstance(node.scale, torch.Tensor):
                    node.scale = torch.tensor([params[param_idx]], dtype=torch.float32)
                    param_idx += 1
                    node.bias = torch.tensor([params[param_idx]], dtype=torch.float32)
                    param_idx += 1
                if node.t_name == "func":
                    for child in node.data:
                        update_constants(child)
            
            update_constants(self.start_node)
            
            # Evaluate error with current constants
            X = data[:, :-1]
            y_true = data[:, -1]
            y_pred = self.forward(X).detach().numpy()
            return np.mean((y_pred - y_true) ** 2)
        
        # Run Nelder-Mead optimization
        from scipy.optimize import minimize
        result = minimize(objective,
                          initial_params, 
                          method='Powell',
                          options={'maxiter': max_iter, 'disp': False}
                          )
        
        # Update tree with optimized constants
        final_params = result.x
        # print("final_params:", final_params)
        param_idx = 0
        def update_final_constants(node):
            nonlocal param_idx
            if CONST_OPT and isinstance(node.scale, torch.Tensor):
                node.scale = torch.tensor([final_params[param_idx]], dtype=torch.float32)
                param_idx += 1
                node.bias = torch.tensor([final_params[param_idx]], dtype=torch.float32)
                param_idx += 1
            if node.t_name == "func":
                for child in node.data:
                    update_final_constants(child)
        
        update_final_constants(self.start_node)
        self.math_expr = self.to_math_expr()

    # evaluate tree error 
    def eval_tree_error(self, data: Union[np.ndarray, torch.Tensor], inv_error: bool = False):
        """Evaluate tree error without constant optimization"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        self.error_is_set = True

        # Original error evaluation logic
        X = data[:, :-1]
        y_true = data[:, -1].reshape(-1, 1)
        
        # Initialize domain_loss
        self.domain_loss = 0.0
        
        try:
            # print("X.shape:", X.shape)
            # print("y_true.shape:", y_true.shape)
            y_pred = self.forward(X)
            # print("y_pred.shape:", y_pred.shape)
            # print("y_pred.reshape(-1, 1).shape:", y_pred.reshape(-1, 1).shape)

            # quit()
            # Calculate forward_loss in any case (RMSE)
            forward_loss = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
            self.forward_loss = forward_loss
            
            if inv_error:
                # Calculate inversion error only if it's required
                inv_loss = self.eval_inv_error(data)
                self.inv_loss = inv_loss
                # error = (1 - INV_ERROR_COEF) * forward_loss + INV_ERROR_COEF * inv_loss
                error = forward_loss + inv_loss
            else:
                # Use only forward_loss
                error = forward_loss
                
                # # Add domain penalty if applicable
                # domain_penalty = self.start_node.get_total_domain_penalty()
                # if domain_penalty > 0:
                #     error += domain_penalty
                # self.domain_loss = domain_penalty

            if np.isnan(error) or np.isinf(error):
                error = float('inf')
            
            self.error = error
            
            return error
        
        except Exception as e:
            print(f"Error in eval_tree_error: {e}")
            self.error = float('inf')
            self.forward_loss = float('inf')
            if inv_error:
                self.inv_loss = float('inf')
            return float('inf')

    # wrapper function for optimization that computes squared difference between tree output and target
    def wrapped_tree_func(self, x: np.ndarray, target_y: float) -> float:
        """Wrapper function for optimization that computes squared difference between tree output and target."""
        # Convert input to float64 for optimization
        x = x.astype(np.float64)
        x_tensor = to_tensor(x.reshape(1, -1), requires_grad=self.requires_grad)
        y_pred = self.forward(varval=x_tensor).detach().numpy().astype(np.float64)
        
        # Ensure we have a scalar output (MSE)
        if y_pred.size > 1:
            # If y_pred is not a scalar, compute MSE
            return float(np.mean((y_pred.flatten() - target_y) ** 2))
        else:
            # If y_pred is a scalar or a single-element array
            return float(((y_pred.item() if hasattr(y_pred, 'item') else y_pred[0]) - target_y) ** 2)

    # evaluate inverse error using scipy.optimize.minimize (for inverse pass/error)
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
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0  # Avoid division by zero
        X = (X - X_mean) / X_std
        
        y_mean = y.mean()
        y_std = y.std()
        if y_std == 0:
            y_std = 1.0  # Avoid division by zero
        y = (y - y_mean) / y_std

        total_inv_error = 0.0
        # n_samples = min(100, len(y))  # Limit samples to prevent excessive computation
        # samples_indices = np.random.choice(len(y), n_samples, replace=False)
        
        # Get optimization method
        method = getattr(self, 'inv_error_method', 'BFGS')
        
        # for i in samples_indices:
        for i in range(len(y)):
            # Use current X as initial guess
            x0 = X[i].copy()  # Make sure we have a contiguous array
            target_y = float(y[i])  # Convert to float64

            # Find x that minimizes (f(x) - y)^2
            try:
                result = opt.minimize(
                    self.wrapped_tree_func,
                    x0=x0,
                    args=(target_y,),
                    method=method,
                    options={'maxiter': 20, 'disp': False}
                )

                if result.success:
                    # Compute error between found x and original x
                    x_found = result.x
                    print("x_found:", x_found, "x0:", x0, "error:", float(np.sum((x_found - x0) ** 2)))
                    error = float(np.sum((x_found - x0) ** 2))
                    total_inv_error += error
                else:
                    # Penalize failed inversions with a reasonable penalty
                    total_inv_error += 10.0
            except Exception as e:
                # Print error but continue with other samples
                # print(f"Optimization failed: {str(e)}")
                total_inv_error += 10.0  # Penalize errors
                continue

        # Return average inverse error
        # return total_inv_error / n_samples
        return total_inv_error / len(y)

    # print tree in a visual representation (as a bash tree command in terminal)
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

    # convert tree to mathematical expression string
    def to_math_expr(self):
        """Convert the tree to a mathematical expression string"""
        def _build_expr(node):
            if not isinstance(node, Node):
                if type(node) is int:
                    return f"x{node}"
                return str(node)
            
            # Handle variable nodes
            if node.t_name == "var":
                if isinstance(node.data, list) and len(node.data) == 1 and isinstance(node.data[0], int):
                    return f"x{node.data[0]}"
                return f"x{node.data}"
            
            # Handle constant/identity nodes
            if node.t_name == "ident":
                if isinstance(node.data, torch.Tensor):
                    return f"{node.data.item()}"
                return str(node.data)
            
            # Build base expression for function nodes
            if node.parity == 1:
                inner = _build_expr(node.data[0])
                # Handle all unary functions
                func_name = node.func.__name__
                if func_name.endswith('_'):  # Remove trailing underscore
                    func_name = func_name[:-1]
                # Special handling for power functions
                if func_name == 'pow2':
                    expr = f"({inner})**2"
                elif func_name == 'pow3':
                    expr = f"({inner})**3"
                elif func_name == 'neg':
                    expr = f"-({inner})"
                else:
                    expr = f"{func_name}({inner})"
            else:
                left = _build_expr(node.data[0])
                right = _build_expr(node.data[1])
                
                # Handle binary operations with special formatting
                func_name = node.func.__name__
                if func_name.endswith('_'):
                    func_name = func_name[:-1]
                
                if func_name == 'sum':
                    expr = f"({left} + {right})"
                elif func_name == 'mult':
                    expr = f"{left} * {right}"
                elif func_name == 'div':
                    expr = f"({left} / {right})"
                # elif func_name == 'pow':
                #     expr = f"({left})**({right})"
                else:
                    expr = f"{func_name}({left}, {right})"
                
            # Add constants if required and they exist
            if CONST_OPT and isinstance(node.scale, torch.Tensor):
                scale = node.scale.item()
                bias = node.bias.item()
                if scale != 1 or bias != 0:
                    if scale != 1:
                        expr = f"{scale}*({expr})"
                    if bias != 0:
                        expr = f"({expr} + {bias})"
            return expr
                
        return _build_expr(self.start_node)

    def is_constant_node(self, node, data=None, tolerance=1e-6):
        """
        Check if a node behaves as a constant across different inputs.
        
        Args:
            node: The node to check
            data: Not used, kept for backward compatibility
            tolerance: Floating point comparison tolerance
            
        Returns:
            bool: True if the node is constant, False otherwise
        """
        # Handle None node
        if node is None:
            return False
        
        # Fast check 1: if node is directly a constant or has var_count 0
        if node.t_name == "ident" and not isinstance(node.data, list):
            return True
        
        # Fast check 2: var_count == 0 means no variables in the subtree
        if hasattr(node, 'var_count') and node.var_count == 0:
            return True
        
        # Fast check 3: if node is a variable, it's not constant
        if node.t_name == "var":
            return False

        # SLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOW
        # Check if the node is constant by taking derivatives and checking if they are zero
        # symbols = sp.symbols(self.str_variables)
        # print("symbols:", symbols, "node:", node)
        # temp_tree = Tree(safe_deepcopy(node))
        # expr = node.to_math_expr()
        # print("expr:", expr) 
        # # quit()
        # if expr:
        #     for i in range(self.num_vars):
        #         res = sp.simplify(sp.diff(expr, self.symbols[i]))
        #         print("res:", res)
        #         if res != 0:
        #             return False
            
        
        # Determine number of variables
        num_vars = self.num_vars if hasattr(self, 'num_vars') and self.num_vars is not None else 1
        if hasattr(node, 'var_count'):
            num_vars = min(num_vars, node.var_count)
        
        # If there are no variables, it's a constant node
        if num_vars == 0:
            return True
        
        try:
            # from parameters import NUM_OF_POINTS_FOR_CONST_NODE_CHECK, CONST_NODE_CHECK_MAGNITUDE
            
            # Generate random points from uniform distribution within range [-CONST_NODE_CHECK_MAGNITUDE, CONST_NODE_CHECK_MAGNITUDE]
            magnitude = CONST_NODE_CHECK_MAGNITUDE
            
            # Create random test points (NUM_OF_POINTS_FOR_CONST_NODE_CHECK × num_vars)
            test_points = torch.rand(NUM_OF_POINTS_FOR_CONST_NODE_CHECK, num_vars) * 2 * magnitude - magnitude
            
            # Evaluate the node at each test point
            try:
                # Evaluate all test points at once
                values = node.eval_node(test_points, cache=False)
                
                if isinstance(values, torch.Tensor):
                    # Convert tensor to numpy array
                    values = values.detach().numpy()
                
                # Check if all values are approximately equal
                if len(values) == 0:
                    return False
                
                # Compare all values to the first value
                first_value = values[0]
                max_diff = np.max(np.abs(values - first_value))
                
                return max_diff <= tolerance
                
            except Exception as e:
                print(f"Error evaluating node: {e}")
                return False
            
        except Exception as e:
            print(f"Error in is_constant_node: {e}")
            return False

    # def is_effective_constant_node(self, node, data, sample_ratio=0.1, tolerance=1e-6):
    #     """
    #     Checks if a node behaves as a constant on the provided dataset.
        
    #     Unlike is_constant_node which checks if a node is mathematically constant,
    #     this method checks if a node produces approximately the same output for
    #     all points in the actual dataset, which is useful for simplification.
        
    #     Args:
    #         node: The node to check
    #         data: Dataset tensor to evaluate on, shape [N, D+1] or [N, D]
    #         sample_ratio: Ratio of data points to sample (0.1 = 10%)
    #         tolerance: Tolerance for floating point comparisons
            
    #     Returns:
    #         bool: True if the node produces the same output for all sampled data points
            
    #     Example:
    #         >>> # A node that is not mathematically constant but might be constant
    #         >>> # for a specific dataset (e.g., sin(100*x) on a small x range)
    #         >>> tree.is_effective_constant_node(node, data)
    #     """
    #     # Apply the basic constant checks first
    #     if self.is_constant_node(node, None):  # Pass None to only check var_count
    #         return True
        
    #     # If no data provided, we can't determine
    #     if data is None or len(data) == 0:
    #         return False
        
    #     try:
    #         # Determine input data format
    #         if data.shape[1] > 1 and hasattr(node, 'var_count'):
    #             # Check if the last column might be the target
    #             if data.shape[1] == node.var_count + 1:
    #                 X_data = data[:, :-1]  # All but last column
    #             else:
    #                 X_data = data  # Use all as input
    #         else:
    #             X_data = data
            
    #         # Sample data points (use all for small datasets)
    #         n_samples = max(2, min(int(len(X_data) * sample_ratio), 100))
            
    #         if len(X_data) <= n_samples:
    #             sample_indices = np.arange(len(X_data))
    #         else:
    #             sample_indices = np.random.choice(len(X_data), n_samples, replace=False)
            
    #         sampled_data = X_data[sample_indices]
            
    #         # Evaluate node on sampled data
    #         results = []
    #         for i in range(len(sampled_data)):
    #             point = sampled_data[i]
    #             if isinstance(point, np.ndarray) and len(point.shape) == 1:
    #                 point = point.reshape(1, -1)
                    
    #             result = node.eval_node(point, cache=False)
                
    #             # Extract scalar value
    #             if isinstance(result, torch.Tensor):
    #                 result = result.item() if result.numel() == 1 else result.flatten()[0]
                    
    #             results.append(result)
            
    #         # Check if all results are the same within tolerance
    #         results = np.array(results)
    #         if np.isnan(results).any() or np.isinf(results).any():
    #             return False
            
    #         max_diff = np.max(np.abs(results - results[0]))
    #         return max_diff <= tolerance
            
    #     except Exception as e:
    #         print(f"Error in is_effective_constant_node: {str(e)}")
    #         return False

# reset cached tensors in a node and its children
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

# safely deepcopy an object containing PyTorch tensors
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
        new_tree.math_expr = obj.math_expr
        return new_tree
    elif isinstance(obj, list):
        return [safe_deepcopy(item) for item in obj]
    elif torch.is_tensor(obj):
        # Always detach and clone tensors
        return obj.detach().clone()
    else:
        return copy.deepcopy(obj)

# crossover two trees (combine parent1 with a node from parent2 with multivariable support)
def crossover(p1: Tree, p2: Tree) -> Tree:
    """
    Perform crossover between two parent trees.
    Returns a new tree that is a copy of p1 with one of its subtrees replaced by a subtree from p2.
    """
    # Create a copy of the first parent
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
                    temp_child.math_expr = temp_child.to_math_expr()
                    return temp_child
            else:
                temp_child.math_expr = temp_child.to_math_expr()
                return temp_child
    
    # If no valid crossover found, return copy of parent1
    return child

# mutate a tree (change random node according to its parity with multivariable support)
def mutation(tree: Tree):
    """Change random node according to its parity with multivariable support"""
    
    mutant = safe_deepcopy(tree)
    
    if mutant.start_node.t_name not in ("var", "ident"):
        # Get mutable nodes (exclude var and ident nodes)
        mutable_indices = []
        for i in range(mutant.max_num_of_node):
            node = mutant.get_node_by_num(i)
            if node and node.t_name not in ("var", "ident"):
                mutable_indices.append(i)
        
        if not mutable_indices:
            return
            
        idx = np.random.choice(mutable_indices)
        new_node = mutant.get_node_by_num(idx)
        
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
    
        mutant.math_expr = mutant.to_math_expr()
    return mutant

def mutation_constant(tree: Tree):
    """Change random constant in the tree"""
    
    mutant = safe_deepcopy(tree)
    
    mutable_indices = []

    if mutant.start_node.t_name not in ("var", "ident"):
        # Get mutable nodes (exclude var and ident nodes)
        for i in range(mutant.max_num_of_node):
            node = mutant.get_node_by_num(i)
            if node and node.t_name == "ident":
                mutable_indices.append(i)
    
    if not mutable_indices:
        return tree
    
    idx = np.random.choice(mutable_indices)
    node = mutant.get_node_by_num(idx)
    node.data = node.data + np.random.normal(0, 1)
    mutant.math_expr = mutant.to_math_expr()
    return mutant

# calculate Mean Squared Error using PyTorch operations
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

def get_rmse_stats(models):
    """Calculate RMSE statistics for a list of models"""
    errors = [model.error for model in models]  # Already RMSE values
    return {
        'all_rmse': errors,
        'mean_rmse': np.mean(errors),
        'median_rmse': np.median(errors),
        'best_rmse': min(errors)
    }

def compute_domination(models: List[Tree]):
    """
    Compute domination relationships between models and mark duplicates.
    
    Args:
        models: List of Tree models to analyze
    """
    # Reset all parameters
    for m in models:
        m.domination_count = 0
        m.dominated_models = []
        m.is_unique = True

    n = len(models)
    # Step 1: Identify duplicates by comparing mathematical expressions
    for i in range(n):
        for j in range(i+1, n):
            if models[i].math_expr == models[j].math_expr or \
                (models[i].error_is_set and models[j].error_is_set and \
                 np.isclose(models[i].forward_loss, models[j].forward_loss, atol=1e-6) and \
                 models[i].depth == models[j].depth and \
                 np.isclose(models[i].domain_loss, models[j].domain_loss, atol=1e-6)):
                models[j].is_unique = False

    # for model in models:
    #     print(model.math_expr, model.is_unique, end="\n")

    # Step 2: Compute domination only among unique models
    for i in range(n):
        if not models[i].is_unique:
            continue
        for j in range(n):
            if i == j or not models[j].is_unique:
                continue
            # Check if model i dominates model j
            if (models[i].forward_loss <= models[j].forward_loss and \
                # models[i].max_num_of_node <= models[j].max_num_of_node and \
                models[i].depth <= models[j].depth and \
                models[i].domain_loss <= models[j].domain_loss and \
                (models[i].forward_loss < models[j].forward_loss or \
                 models[i].depth < models[j].depth or \
                 models[i].domain_loss < models[j].domain_loss)):
                
                models[i].dominated_models.append(models[j])
                models[j].domination_count += 1

def calculate_crowding_distance(front: List[Tree]):
    """
    Calculate crowding distance for models in a front.
    Modifies the models in place by setting their crowding_distance attribute.
    """
    if not front:
        return
    
    num_models = len(front)
    if num_models < 2:
        front[0].crowding_distance = float('inf')
        return
    
    # Initialize crowding distances
    for model in front:
        model.crowding_distance = 0
    
    # Calculate crowding distance for each objective
    # objectives = ['forward_loss', 'inv_loss', 'max_num_of_node', 'domain_loss']
    # objectives = ['forward_loss', 'inv_loss', 'max_num_of_node']
    objectives = ['forward_loss', 'inv_loss', 'depth']
    

    for objective in objectives:
        # Sort front by the objective
        front.sort(key=lambda x: getattr(x, objective))
        
        # Set infinite distance to boundary points
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        # Calculate crowding distance for intermediate points
        obj_range = getattr(front[-1], objective) - getattr(front[0], objective)
        if obj_range == 0:
            obj_range = 1  # Avoid division by zero
            
        for i in range(1, num_models - 1):
            distance = (getattr(front[i + 1], objective) - getattr(front[i - 1], objective)) / obj_range
            front[i].crowding_distance += distance

def non_dominated_sort(models: List[Tree]) -> List[List[Tree]]:
    """
    Perform non-dominated sorting (NSGA-II) on models with uniqueness check.
    
    Args:
        models: List of Tree models to sort
    
    Returns:
        List of fronts, where each front is a list of non-dominated models
    """
    if not models:
        return []
    
    # Compute domination relationships and mark duplicates
    compute_domination(models)
    
    # Filter only unique models
    unique_models = [m for m in models if m.is_unique]
    if not unique_models:
        return []
    
    fronts = [[]]  # Initialize with empty first front
    
    # First front: models not dominated by any other model
    first_front = [m for m in unique_models if m.domination_count == 0]
    if not first_front:
        return []
    
    fronts[0] = first_front
    current_front = 0
    
    while current_front < len(fronts):
        next_front = []
        for model in fronts[current_front]:
            for dominated in model.dominated_models:
                dominated.domination_count -= 1
                if dominated.domination_count == 0:
                    next_front.append(dominated)
        
        current_front += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts

def model_selection(models: List[Tree], params: Optional[Dict] = None) -> List[Tree]:
    """
    Select models based on specified method (NSGA-II or top_k).
    
    Args:
        models: List of Tree models to select from
        params: Dictionary with selection parameters
    
    Returns:
        List of selected models
    """
    if not models:
        return []
    
    if params is None:
        params = {}
    
    population_size = params.get("population_size", POPULATION_SIZE)
    method = params.get("method", SELECTION_METHOD)
    
    try:
        if method == "NSGA-II":
            # Get sorted fronts with uniqueness check
            fronts = non_dominated_sort(models)
            print("len(fronts):", len(fronts))
            if not fronts:
                # Fall back to top_k if sorting fails
                valid_models = [m for m in models if m.is_unique and hasattr(m, 'error')]
                valid_models.sort(key=lambda x: x.error)
                return [safe_deepcopy(model) for model in valid_models[:population_size]]
            
            selected_models = []
            front_idx = 0
            
            # Add complete fronts while possible
            while front_idx < len(fronts) and len(selected_models) + len(fronts[front_idx]) <= population_size:
                current_front = fronts[front_idx]
                selected_models.extend([safe_deepcopy(model) for model in current_front])
                front_idx += 1
            
            # If we need more models and there are more fronts
            if len(selected_models) < population_size and front_idx < len(fronts):
                current_front = fronts[front_idx]
                # Calculate crowding distance for the last front
                calculate_crowding_distance(current_front)
                # Sort by crowding distance (higher is better)
                current_front.sort(key=lambda x: x.crowding_distance, reverse=True)
                # Add remaining models to reach population_size
                remaining = population_size - len(selected_models)
                selected_models.extend([safe_deepcopy(model) for model in current_front[:remaining]])
            
            return selected_models
        
        else:  # "top_k" selection
            # Sort by error, considering only unique models
            valid_models = [m for m in models if m.is_unique and hasattr(m, 'error')]
            valid_models.sort(key=lambda x: x.error)
            return [safe_deepcopy(model) for model in valid_models[:population_size]]
    
    except Exception as e:
        print(f"Warning: Model selection failed with error: {str(e)}")
        # Fall back to simple top_k selection
        valid_models = [m for m in models if hasattr(m, 'error')]
        valid_models.sort(key=lambda x: x.error)
        return [safe_deepcopy(model) for model in valid_models[:population_size]]

# optimize constants for a population of trees
def optimize_population_constants(models: List[Tree], training_data: Union[np.ndarray, torch.Tensor], max_iter: int = 50) -> None:
    """Optimize constants for a population of trees"""
    if not CONST_OPT:
        return
    
    print("Optimizing constants for population...")
    for model in models:
        model.optimize_constants(training_data, max_iter=max_iter)

# set evaluated error for models
def set_evaluated_error(models_to_eval: List[Tree], training_data: Union[np.ndarray, torch.Tensor], epoch: int):
    """Set evaluated error for models"""
    for model in models_to_eval:
        try:
            # Проверяем, нужно ли вычислять инверсную ошибку
            use_inv_error = epoch % INV_ERROR_EVAL_FREQ == 0 and model.requires_grad
            
            # Вычисляем ошибку модели
            model.eval_tree_error(training_data, inv_error=use_inv_error)
            
            # Проверяем валидность результатов
            if np.isnan(model.error) or np.isinf(model.error):
                print(f"Warning: Model {model.math_expr} has invalid error {model.error}")
                model.error = float('inf')
                model.forward_loss = float('inf')
                if hasattr(model, 'inv_loss'):
                    model.inv_loss = float('inf')
        except Exception as e:
            print(f"Error evaluating model {model.math_expr}: {e}")
            model.error = float('inf')
            model.forward_loss = float('inf')
            if use_inv_error:
                model.inv_loss = float('inf')

def tournament(model1: Tree, model2: Tree):
    """Tournament selection"""
    # Select two random models
    if model1.error < model2.error and \
        model1.depth < model2.depth and \
        model1.domain_loss < model2.domain_loss and \
        model1.forward_loss < model2.forward_loss and \
        model1.inv_loss < model2.inv_loss:
        return model1
    elif model1.error > model2.error and \
        model1.depth > model2.depth and \
        model1.domain_loss > model2.domain_loss and \
        model1.forward_loss > model2.forward_loss and \
        model1.inv_loss > model2.inv_loss:
        return model2
    else:
# TODO: if NSGA is used, else compute crowding distance for tournament selection =========================================
        model_with_max_crowding_distance = max(model1, model2, key=lambda x: x.crowding_distance)
        # print("model_with_max_crowding_distance:", model_with_max_crowding_distance.math_expr, "crowding_distance:", model_with_max_crowding_distance.crowding_distance)
        return model_with_max_crowding_distance
    
    

# evolution process with crossover, mutation, and constant optimization(if CONST_OPT is True)
def evolution(num_of_epochs, models, training_data):
    """evolution process with crossover, mutation, and constant optimization"""
    
    for model in models:
        print(model.math_expr, end="\n")

    # initialize arrays for tracking progress
    mean_rmse_arr = np.zeros(num_of_epochs)
    median_rmse_arr = np.zeros(num_of_epochs)
    best_rmse_arr = np.zeros(num_of_epochs)
    
    # initialize best model and best error
    best_model = safe_deepcopy(models[0])
    min_rmse_loss = float('inf')
    best_rmse = float('inf')

    # Model selection (choose method of selection based on params)
    selection_params = {
        "population_size": POPULATION_SIZE,
        "method":          SELECTION_METHOD
    }

    # select POPULATION_SIZE models for init population for evolution 
    print("Selecting models for init population...")
    models = model_selection(models, selection_params)
    # selection_params["population_size"] = NUM_OF_MODELS_TO_SELECT # uncomment if we want to select NUM_OF_MODELS_TO_SELECT models for init population
    
    # evolution loop
    for epoch in range(num_of_epochs):

        # Initialize set to track unique expressions
        unique_expressions = {model.math_expr for model in models}

        print("EPOCH:", epoch)
        
        # for model in models:
        #     print(model.math_expr, end="\n")
        
        # print("len(models):", len(models))
        # print("POPULATION_SIZE//len(models) - 1:", POPULATION_SIZE//len(models) - 1)
        # if epoch == 2:
        #     quit()

        # Create part of a new population (copy of the current population)
        models_to_eval = []
        for model in models:
            new_model = safe_deepcopy(model)
            models_to_eval.append(new_model)

        if epoch != 0: # for the first epoch we don't need to do anything
            

            # # =======hard selection, crossover, mutation, evaluation, selection=======
            # # Crossover (combine parent1 with a node from parent2 with multivariable support)
            # for model in models:
            #     # example: if we have 10 selected models, POPULATION_SIZE = 100,
            #     # we need to create 90 new models
            #     # 9*10 = len(models)*(POPULATION_SIZE//len(models) - 1)
                
            #     for _ in range(POPULATION_SIZE//len(models) - 1):
            #         new_model = crossover(model, models[np.random.randint(0, len(models))])
            #         while new_model.max_num_of_node > new_model.max_depth:
            #             new_model = crossover(model, models[np.random.randint(0, len(models))])
            #         models_to_eval.append(new_model)

            # # Mutation (change random node according to its parity with multivariable support)
            # for model in models_to_eval:
            #     if np.random.uniform() < model.mutation_prob:
            #         mutant = mutation(model)
            #         models_to_eval.append(mutant)
            # # ========================================================


            # =======crossover, then select the best models, then mutate=======
            # crossover
            for model in models:
                # Try up to 10 times to get a unique expression through crossover
                for _ in range(10):
                    new_model = crossover(model, tournament(models[np.random.randint(0, len(models))], models[np.random.randint(0, len(models))]))
                    while new_model.max_num_of_node > new_model.max_depth:
                        new_model = crossover(model, tournament(models[np.random.randint(0, len(models))], models[np.random.randint(0, len(models))]))
                    
                    expr = new_model.math_expr
                    if expr not in unique_expressions:
                        unique_expressions.add(expr)
                        models_to_eval.append(new_model)
                        break
            print(f"Number of unique expressions after crossover: {len(unique_expressions)}")

            # Evaluate models (choose variant of evaluation based on requires_grad)
            set_evaluated_error(models_to_eval, training_data, epoch)

            # select the best models
            models_to_eval = model_selection(models_to_eval, selection_params)
            unique_expressions = {model.math_expr for model in models_to_eval}

            # mutate from best models
            new_models = []
            for model in models_to_eval:
            # Try up to 10 times to get a unique expression through mutation
                for _ in range(10):
                    mutated_model = mutation(model)
                    if not CONST_OPT and np.random.uniform() < CONST_MUTATION_PROB:
                        mutated_model = mutation_constant(mutated_model)
                    expr = mutated_model.math_expr
                    if expr not in unique_expressions:
                        unique_expressions.add(expr)
                        new_models.append(mutated_model)
                        break
            models_to_eval.extend(new_models)
            print(f"Number of unique expressions after mutation: {len(unique_expressions)}")
            # print("len(models_to_eval) after mutation:", len(models_to_eval))
            # quit()
            # ========================================================


            # # =======crossover, mutation, evaluation, selection=======
            # # crossover
            # for model in models:
            #     new_model = crossover(model, models[np.random.randint(0, len(models))])
            #     while new_model.max_num_of_node > new_model.max_depth:
            #         new_model = crossover(model, models[np.random.randint(0, len(models))])
            #     models_to_eval.append(new_model)
            # # print("len(models_to_eval) after crossover:", len(models_to_eval))

            # # mutate
            # for model in models:
            #     models_to_eval.append(mutation(model))
            # # print("len(models_to_eval) after mutation:", len(models_to_eval))
            # # quit()
            # # ========================================================


            # # =======crossover, then mutate from all models, evaluate, select the best models=======
            # # crossover
            # for model in models:
            #     new_model = crossover(model, models[np.random.randint(0, len(models))])
            #     while new_model.max_num_of_node > new_model.max_depth:
            #         new_model = crossover(model, models[np.random.randint(0, len(models))])
            #     models_to_eval.append(new_model)
            # # print("len(models_to_eval) after crossover:", len(models_to_eval))
                
            # # mutate from all models
            # new_models = [] 
            # for model in models_to_eval:
            #     new_models.append(mutation(model))
            # models_to_eval = models_to_eval + new_models
            # # print("len(models_to_eval) after mutation:", len(models_to_eval))
            # # quit()
            # # ========================================================


        # Optimize constants periodically if enabled
        if CONST_OPT and epoch % CONST_OPT_FREQUENCY == 0:
            optimize_population_constants(models_to_eval, training_data, max_iter=50)

        # Evaluate models (choose variant of evaluation based on requires_grad)
        set_evaluated_error(models_to_eval, training_data, epoch)

        # Best models selection (choose method of selection based on params)
        selected_models = model_selection(models_to_eval, selection_params)
        
        # Ensure we have models
        if not selected_models:
            print(f"No selected models, using {selection_params['method']}")
            models_to_eval.sort(key=lambda x: x.error)
            models = [safe_deepcopy(model) for model in models_to_eval[:NUM_OF_MODELS_TO_SELECT]]
        else:
            models = selected_models

        models.sort(key=lambda x: x.error)

        # Track progress (RMSE statistics)
        rmse_stats = get_rmse_stats(models)
        mean_rmse_arr[epoch] = rmse_stats['mean_rmse']
        median_rmse_arr[epoch] = rmse_stats['median_rmse']
        best_rmse_arr[epoch] = rmse_stats['best_rmse']

        # Track best model
        if models[0].error < best_rmse:
            best_rmse = models[0].error
            best_model = safe_deepcopy(models[0])
            print("best_model:", best_model.math_expr, "best_rmse:", best_rmse)
            print("forward_loss:", best_model.forward_loss, "inv_loss:", best_model.inv_loss, "max_num_of_node:", best_model.max_num_of_node)

        # global min_rmse_loss
        print("min_rmse_loss:", min_rmse_loss)
        if models[0].forward_loss < min_rmse_loss:
            min_rmse_loss = models[0].forward_loss
            if min_rmse_loss < 0.0001:
                print("min_rmse_loss < 0.0001")
                print(f"Current best RMSE: {best_rmse:.6f}")
                print(f"Number of unique expressions: {len(unique_expressions)}")
                for model in models[:5]:
                    print(model.math_expr, "depth:", model.depth, "inv_loss:", model.inv_loss, end="\n")
                # quit()

        # Print best model and its expression every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Current best RMSE: {best_rmse:.6f}")
            print(f"Current mean RMSE: {rmse_stats['mean_rmse']:.6f}")
            print(f"Current median RMSE: {rmse_stats['median_rmse']:.6f}")
            print(f"Number of unique expressions: {len(unique_expressions)}")
            for model in models[:5]:
                print(model.math_expr, "depth:", model.depth, "inv_loss:", model.inv_loss, end="\n")

    # Store results
    best_model.error_history = mean_rmse_arr
    best_model.min_rmse_loss = min_rmse_loss
    best_model.best_rmse = best_rmse
    
    # Plot error history using the new visualization function
    from visual import plot_rmse_history
    plot_rmse_history(mean_rmse_arr, median_rmse_arr, best_rmse_arr)
    
    # print final results
    print(f"Final min_rmse_loss: {min_rmse_loss}")
    print(f"Best RMSE achieved: {best_rmse:.6f}")
    print(f"Final mean RMSE: {rmse_stats['mean_rmse']:.6f}")
    print(f"Final median RMSE: {rmse_stats['median_rmse']:.6f}")
    print(f"Total number of unique expressions found: {len(unique_expressions)}")

    # return best model
    print("selected_models:", [model.math_expr for model in models])
    print("\n")
    print("sorted_selected_models:", [model.math_expr for model in sorted(models, key=lambda x: x.error)])
    for i, model in enumerate(sorted(models, key=lambda x: x.error)):
        print(f"{i}: {model.math_expr}")
        print(f"   RMSE: {model.error:.6f}")
        print(f"   forward_loss: {model.forward_loss:.6f}")
        print(f"   inv_loss: {getattr(model, 'inv_loss', 0):.6f}")
        print(f"   domain_loss: {model.domain_loss:.6f}")
        print(f"   complexity: {model.max_num_of_node}")
    return models[0]