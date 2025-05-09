import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from nodes import *
from typing import List, Union, Optional, Dict, Tuple, Callable
from parameters import *
import sympy as sp
import scipy.optimize as opt
from dataclasses import dataclass, field
from visual import plot_scatter, plot_results

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
        self.var_names = set() 
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
                varval = to_tensor(varval, requires_grad=False)
            if varval.dim() == 1:
                varval = varval.reshape(-1, 1)

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

        # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        self.mse = 0

        self.forward_loss = 0
        self.inv_loss = 0
        self.abs_loss = 0
        self.spatial_abs_loss = 0
        # self.domain_loss = 0
        self.max_num_of_node = 0
        # self.error = 0
        self.requires_grad = requires_grad
        self._set_requires_grad(requires_grad)
        self.enumerate_tree()
        self.update_var_counts()  # Count variables under each node
        self.num_vars = len(self.start_node.var_names)  # Will be set during evaluation (number of unique variables)
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

        self.error_history = None
        self.min_forward_loss = None
        self.best_sum_of_losses = None

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
                node.var_names.add(node.data[0])
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


    def reset_losses(self):
        """Reset all losses to zero"""
        self.error_is_set = False
        self.error = 0

        # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        self.mse = 0

        self.forward_loss = 0
        self.inv_loss = 0
        self.abs_loss = 0
        self.spatial_abs_loss = 0
        # self.domain_loss = 0

    # forward pass with gradients if gradient is required (for inverse pass)
    def forward_with_gradients(self, varval: torch.Tensor, cache: bool = True):
        """
        Performs forward pass for the entire input tensor and calculates gradients for each input vector.

        Args:
            varval: Input tensor of shape (N, num_vars), where N is the number of input vectors.
        """

        # Функция для обработки каждого входного вектора отдельно
        def full_forward(input_tensor):
            return self.start_node.eval_node(varval=input_tensor, cache=cache).squeeze(-1)

        # PyTorch считает, что входной тензор влияет на все выходные элементы. Чтобы избежать этого:
        outputs = full_forward(varval)  # (100,)
        gradients = torch.autograd.functional.vjp(
            full_forward, varval, v=torch.ones_like(outputs), create_graph=self.requires_grad
        )[1]  # Извлекаем градиенты (вектор производных)

        self.grad = gradients  # Сохраняем градиенты для отладки
        return outputs

    # forward pass (choose variant of forward pass based on requires_grad)
    def forward(self, varval=None, cache: bool = True, new_requires_grad: bool = False):
        if self.requires_grad and new_requires_grad: 
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
    def eval_tree_error(self, data: Union[np.ndarray, torch.Tensor], forward_error: bool = False, inv_error: bool = False, abs_error: bool = False, spatial_abs_error: bool = False):
        """Evaluate tree error without constant optimization"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        self.error_is_set = True

        error = 0.0

        # Original error evaluation logic
        X = data[:, :-1]
        y_true = data[:, -1].reshape(-1, 1)
        
        # Initialize domain_loss
        # self.domain_loss = 0.0
        
        # evaluate tree on whole dataset
        y_pred = self.forward(X)

        try:
            # Calculate forward_loss in any case (RMSE)
            # forward_loss = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
            forward_loss = torch.mean((y_pred - y_true) ** 2).item()
            if np.isinf(self.forward_loss):
                print("forward_loss is inf for model:", self.to_math_expr())
            elif np.isnan(self.forward_loss):
                print("forward_loss is nan for model:", self.to_math_expr())

            # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            self.mse = forward_loss
            if forward_error:
                self.forward_loss = forward_loss
                error += forward_loss
            # if forward_error:
            #     # # evaluate tree on whole dataset
            #     # y_pred = self.forward(X)

            #     # Calculate forward_loss in any case (RMSE)
            #     # forward_loss = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
            #     forward_loss = torch.mean((y_pred - y_true) ** 2).item()
            #     if np.isinf(self.forward_loss):
            #         print("forward_loss is inf for model:", self.to_math_expr())
            #     elif np.isnan(self.forward_loss):
            #         print("forward_loss is nan for model:", self.to_math_expr())

            #     # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            #     self.forward_loss = forward_loss
            #     error += forward_loss

            if inv_error:
                # Calculate inversion error only if it's required
                # total_inv_loss = self.eval_inv_error(data)
                total_inv_loss = self.eval_inv_error_torch(data)
                # print(f"got total_inv_loss = {total_inv_loss} for model: {self.to_math_expr()}")
                if np.isinf(total_inv_loss):
                    print("total_inv_loss is inf for model:", self.to_math_expr())
                elif np.isnan(total_inv_loss):
                    print("total_inv_loss is nan for model:", self.to_math_expr())
                self.inv_loss = total_inv_loss
                error += total_inv_loss

            if spatial_abs_error and abs_error:
                # print("y_pred.type():", y_pred.type())
                # print("y_pred.dim():", y_pred.dim())           
                predicted_limits = torch.clamp(torch.max(y_pred) - torch.min(y_pred), -MAX_VALUE, MAX_VALUE)
                
                # print("predicted_limits:", predicted_limits)
                # print("predicted_limits.dim() in eval_tree_error:", predicted_limits.dim())
                # print("predicted_limits.type() in eval_tree_error:", predicted_limits.type())
                total_spatial_abs_loss, total_abs_loss = self.eval_abs_inv_error_torch(data, pred_limits=predicted_limits)
                # print("total_spatial_abs_loss.type():", type(total_spatial_abs_loss))
                # print("total_abs_loss.type():", type(total_abs_loss))
                # print("total_spatial_abs_loss.dim():", total_spatial_abs_loss.dim())
                # print("total_abs_loss.dim():", total_abs_loss.dim())
                self.abs_loss = total_abs_loss
                self.spatial_abs_loss = total_spatial_abs_loss
                if np.isinf(total_abs_loss):
                    print("total_abs_loss is inf for model:", self.to_math_expr())
                elif np.isnan(total_abs_loss):
                    print("total_abs_loss is nan for model:", self.to_math_expr())
                if np.isinf(total_spatial_abs_loss):
                    print("total_spatial_abs_loss is inf for model:", self.to_math_expr())
                elif np.isnan(total_spatial_abs_loss):
                    print("total_spatial_abs_loss is nan for model:", self.to_math_expr())
                error += total_abs_loss
                error += total_spatial_abs_loss

            elif abs_error:
                # calculate abs error using numpy
                # total_abs_loss = self.eval_abs_inv_error(data)
                
                # calculate abs error using torch
                predicted_limits = torch.clamp(torch.max(y_pred) - torch.min(y_pred), -MAX_VALUE, MAX_VALUE)                
                total_abs_loss = self.eval_abs_inv_error_torch(data, pred_limits=predicted_limits, error_type="abs")
                self.abs_loss = total_abs_loss
                if np.isinf(total_abs_loss):
                    print("total_abs_loss is inf for model:", self.to_math_expr())
                elif np.isnan(total_abs_loss):
                    print("total_abs_loss is nan for model:", self.to_math_expr())
                error += total_abs_loss

            elif spatial_abs_error:
                # calculate abs error using torch
                predicted_limits = torch.clamp(torch.max(y_pred) - torch.min(y_pred), -MAX_VALUE, MAX_VALUE)                
                total_spatial_abs_loss = self.eval_abs_inv_error_torch(data, pred_limits=predicted_limits, error_type="spatial")
                self.spatial_abs_loss = total_spatial_abs_loss
                if np.isinf(total_spatial_abs_loss):
                    print("total_spatial_abs_loss is inf for model:", self.to_math_expr())
                elif np.isnan(total_spatial_abs_loss):
                    print("total_spatial_abs_loss is nan for model:", self.to_math_expr())
                error += total_spatial_abs_loss

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
            if abs_error:
                self.abs_loss = float('inf')
            if spatial_abs_error:
                self.spatial_abs_loss = float('inf')
            # quit()
            return float('inf')

    # wrapper function for optimization that computes squared difference between tree output and target
    def wrapped_tree_func(self, x: np.ndarray, target_y: float) -> float:
        """Wrapper function for optimization that computes squared difference between tree output and target."""
        # Convert input to float64 for optimization
        x = x.astype(np.float64)
        x_tensor = to_tensor(x.reshape(1, -1), requires_grad=self.requires_grad)
        y_pred = self.forward(varval=np.array(x_tensor.detach().numpy())).detach().numpy().astype(np.float64)

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

        # print("type(data):", type(data))

        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()

        data_limits = data.max(axis=0) - data.min(axis=0)
        max_error = np.sum(data_limits[:-1])
        # print("data_limits:", data_limits)
        # print("max_error:", max_error)
        # quit()

        # print("type(data):", type(data))

        # Convert data to float64
        X = data[:, :-1].astype(np.float64)
        y = data[:, -1].astype(np.float64)

        # print("X.shape:", X.shape)
        # print("y.shape:", y.shape)
        # quit()

        # # Normalize data
        # X_mean = X.mean(axis=0)
        # X_std = X.std(axis=0)
        # X_std[X_std == 0] = 1.0  # Avoid division by zero
        # X = (X - X_mean) / X_std
        
        # y_mean = y.mean()
        # y_std = y.std()
        # if y_std == 0:
        #     y_std = 1.0  # Avoid division by zero
        # y = (y - y_mean) / y_std

        total_inv_error = 0.0
        # n_samples = min(100, len(y))  # Limit samples to prevent excessive computation
        # samples_indices = np.random.choice(len(y), n_samples, replace=False)
        
        # Get optimization method
        method = getattr(self, 'inv_error_method', 'Nelder-Mead')
        
        # for i in samples_indices:
        for i in range(len(y)):
            # Use current X as initial guess
            x0 = X[i].copy()  # Make sure we have a contiguous array
            target_y = float(y[i])  # Convert to float64

            # print("x0:", x0)
            # print("target_y:", target_y)
            # print("type(x0):", type(x0))
            # print("type(target_y):", type(target_y))
            # quit()

            # Find x that minimizes (f(x) - y)^2
            try:
                # print("x0:", x0)
                # print("target_y:", target_y)
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
                    # print("self.forward(varval=x_found):", self.forward(varval=x_found))
                    # print("self.to_math_expr():", self.to_math_expr(), "self.depth:", self.depth)
                    # print("x_found:", x_found)
                    # print("x0:", x0)
                    # print("type(x_found):", type(x_found))
                    # print("type(x0):", type(x0))
                    if np.isclose(x_found, x0, atol=1e-6):
                        if np.isclose(self.forward(varval=x_found), target_y, atol=1e-6):
                            error = 0.0
                        else:
                            error = max_error
                    else:

                        error = np.sqrt(np.sum((x_found - x0) ** 2))
                    # print("error:", error)
                    # quit()
                    total_inv_error += error
                else:
                    # Penalize failed inversions with a reasonable penalty
                    total_inv_error += max_error
            except Exception as e:
                # Print error but continue with other samples
                print(f"Optimization failed: {str(e)}")
                total_inv_error += max_error  # Penalize errors
                continue
            # quit()

        # Return average inverse error
        # return total_inv_error / n_samples
        # print("total_inv_error", total_inv_error)
        return total_inv_error / len(y)
        # return total_inv_error

    def eval_inv_error_torch(self, data: Union[np.ndarray, torch.Tensor], steps=50, lr=0.001) -> float:
        """
        Evaluate inverse error using PyTorch optimization.
        Measures how well inputs can be recovered from outputs using inverse modeling.
        Adds checks for gradient sensitivity and model flatness.
        """

        if self.inv_loss > 0:
            return self.inv_loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32).to(device)
        else:
            data = data.clone().detach().to(device)

        X_data = data[:, :-1].clone().detach()
        y_target = data[:, -1].clone().detach()

        data_limits = X_data.max(dim=0).values - X_data.min(dim=0).values
        tolerance = 0.01 * (y_target.max() - y_target.min())
        max_error = (torch.norm(data_limits).item())**2

        # Return max_error if the model is constant
        if self.depth == 0 or self.is_constant_node(self.start_node):
            print(f"Tree {self.to_math_expr()} is constant, returning max_error = {max_error}")
            return max_error

        y_pred_full = self.forward(X_data.to("cpu")).to(device)

        # Check if model output is nearly constant (flat model)
        if (y_pred_full.max() - y_pred_full.min()).item() < 1e-3:
            print("Model output range too small, likely constant. Returning max_error.")
            return max_error

        # Compute sensitivity (gradient of output w.r.t. inputs)
        X_data_for_grad = X_data.clone().detach().requires_grad_(True)
        y_pred_check = self.forward(X_data_for_grad.to("cpu")).to(device)
        grads = torch.autograd.grad(
            outputs=y_pred_check,
            inputs=X_data_for_grad,
            grad_outputs=torch.ones_like(y_pred_check),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        mean_grad = grads.norm(p=2, dim=1).mean()
        if mean_grad < 1e-5:
            print("Model gradient too small — not invertible. Returning max_error.")
            return max_error

        # Create mask where y_target is within range of model outputs
        mask_in_range = (y_target >= y_pred_full.min()) & (y_target <= y_pred_full.max())

        if mask_in_range.sum() == 0:
            print("All points are outside the model output range, returning max_error =", max_error)
            return max_error

        X_orig = X_data[mask_in_range].clone().detach()
        y_target_masked = y_target[mask_in_range]
        X_optim = X_orig.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([X_optim], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            y_pred = self.forward(X_optim.to("cpu")).to(device)

            loss = ((y_pred - y_target_masked) ** 2).mean() + (X_optim - X_orig).norm(p=2, dim=1).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss exploded in eval_inv_error_torch. Returning max_error.")
                return max_error

            loss.backward()
            torch.nn.utils.clip_grad_norm_([X_optim], max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            y_final = self.forward(X_optim.to("cpu")).to(device)

            y_error = ((y_final - y_target_masked) ** 2).mean()
            x_error = ((X_optim - X_orig) ** 2).mean()

            if y_error > tolerance:
                return max_error

            return x_error.item()
    # def eval_inv_error_torch(self, data: Union[np.ndarray, torch.Tensor], steps=50, lr=0.001) -> float:
    #     """
    #     Evaluate inverse error using PyTorch optimization.
    #     Hybrid initialization:
    #     - If X_orig predicts well -> start from X_orig
    #     - Else -> start from zeros (center of allowed region)
    #     """

    #     if self.inv_loss > 0:
    #         return self.inv_loss

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     if isinstance(data, np.ndarray):
    #         data = torch.tensor(data, dtype=torch.float32).to(device)
    #     else:
    #         data = data.clone().detach().to(device)

    #     # mean_x = data[:, :-1].mean(dim=0)
    #     # std_x = data[:, :-1].std(dim=0)
    #     # std_x[std_x == 0] = 1.0  # Avoid division by zero
    #     # data[:, :-1] = (data[:, :-1] - mean_x) / std_x

    #     # mean_y = data[:, -1].mean(dim=0)
    #     # std_y = data[:, -1].std(dim=0)
    #     # std_y[std_y == 0] = 1.0  # Avoid division by zero
    #     # data[:, -1] = (data[:, -1] - mean_y) / std_y

    #     # plot_scatter(data[:, :-1].to("cpu").numpy(), data[:, -1].to("cpu").numpy())

    #     X_data = data[:, :-1].clone().detach()
    #     y_target = data[:, -1].clone().detach()

    #     # Compute data limits
    #     data_limits = X_data.max(dim=0).values - X_data.min(dim=0).values
    #     tolerance = 0.01 * (y_target.max() - y_target.min())
    #     max_error = (torch.norm(data_limits).item())**2 # L2 norm

    #     # If tree is constant, return max error
    #     if self.depth == 0 or self.is_constant_node(self.start_node):
    #         print(f"Tree {self.to_math_expr()} is constant, returning max_error = {max_error}")
    #         return max_error

    #     y_pred_full = self.forward(X_data.to("cpu")).to(device)

    #     # plot_results(X_data.to("cpu").numpy(), y_target.to("cpu").numpy(), y_pred_full.to("cpu").numpy().flatten(), "Initial comparison")
        
    #     # try:
    #     #     plot_scatter(X_data.to("cpu").numpy(), y_pred_full.to("cpu").numpy().flatten(), "Initial")
    #     # except Exception as e:
    #     #     print(f"Error plotting scatter: {e}")
    #     # quit()        
        
    #     # Создаем маску с правильной формой
    #     mask_in_range = (y_target >= y_pred_full.min()) & (y_target <= y_pred_full.max())

    #     # If all points are outside the range, return max error
    #     if mask_in_range.sum() == 0:
    #         print("All points are outside the range, returning max_error =", max_error)
    #         return max_error

    #     # Apply mask to data
    #     X_optim = X_data[mask_in_range].clone().detach().requires_grad_(True)
    #     y_target = y_target[mask_in_range]
            
    #     optimizer = torch.optim.Adam([X_optim], lr=lr)

    #     for _ in range(steps):
    #         optimizer.zero_grad()
    #         y_pred = self.forward(X_optim.to("cpu")).to(device)
                
    #         loss = ((y_pred - y_target) ** 2).mean() + (X_optim - X_data[mask_in_range]).norm(p=2, dim=1).mean() # MSE + L2 norm

    #         if torch.isnan(loss) or torch.isinf(loss):
    #             print("Loss exploded in eval_inv_error_torch during optimization for model:", self.to_math_expr(), "returning max_error =", max_error)
    #             return max_error

    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_([X_optim], max_norm=1.0)
    #         optimizer.step()

    #     with torch.no_grad():
    #         y_final = self.forward(X_optim.to("cpu")).to(device)
                
    #         diff = torch.abs(y_final - y_target)
    #         failed_mask = diff > tolerance
    #         success_mask = ~failed_mask

    #         # try:
    #         #     plot_scatter(X_optim.to("cpu").numpy(), y_final.to("cpu").numpy().flatten(), "Optimized")
    #         # except Exception as e:
    #         #     print(f"Error plotting scatter: {e}")
    #         # quit()

    #         # X_optim = X_optim*std_x + mean_x

    #         # spatial_error = torch.norm(X_optim - X_data[mask_in_range], dim=1)
    #         spatial_error = torch.mean((X_optim - X_data[mask_in_range])**2)
    #         error = torch.where(success_mask, spatial_error, torch.full_like(spatial_error, max_error))
    #         # print(f"error.mean().item() for model {self.math_expr} is: {error.mean().item()}")
    #         return error.mean().item()

    def eval_abs_inv_error(self, data: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Вычисляет абсолютную обратную ошибку (RMSE) между исходными и предсказанными входами.
        Использует ортогональные расстояния от точек до поверхности модели.
        
        Args:
            data: Входные данные формата [X, y], где X - входные переменные, y - выходное значение
            
        Returns:
            float: Среднеквадратичная ошибка (RMSE) между исходными и найденными входами
        """
        # Преобразуем в numpy массив, если передан torch.Tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
            
        # Разделяем входные и выходные значения
        X = data[:, :-1].astype(np.float64)
        y = data[:, -1].astype(np.float64)
        
        # Вычисляем предельные значения для ошибок
        data_limits = data.max(axis=0) - data.min(axis=0)
        max_error = np.sum(data_limits[:-1])  # Максимальная ошибка для случаев неудачной оптимизации
        
        # Получаем метод оптимизации (или используем Nelder-Mead по умолчанию)
        method = getattr(self, 'abs_inv_error_method', 'Nelder-Mead')
        
        total_sq_error = 0.0
        num_points = len(y)
        
        # Для отладки: количество успешных и неудачных оптимизаций
        successful_optimizations = 0
        failed_optimizations = 0
        
        for i in range(num_points):
            # print(f"i: {i}")
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
                output = self.forward(varval=u_input)
                
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
                    method=method,
                    options={'maxiter': 100, 'disp': False}
                )
                
                if result.success:
                    successful_optimizations += 1
                    # Извлекаем найденную точку
                    x_found = result.x
                    fun_error = result.fun
                    
                    # Вычисляем квадрат расстояния между исходной и найденной точками входа
                    # sq_error = np.sum((x_found - x0)**2) + (self.forward(varval=torch.tensor(x_found).reshape(1, -1).float()) - target_y)**2
                    sq_error = fun_error
                    total_sq_error += sq_error

                    # Выводим детали для первых 5 точек
                    if i < 5:
                        # Преобразуем для вывода
                        x_input = x_found.reshape(1, -1)
                        model_output = self.forward(varval=x_input)
                        if hasattr(model_output, 'detach'):
                            model_output = model_output.detach().numpy()
                            
                        # print(f"Точка {i}:")
                        # print(f"  Оригинальный вход: {x0}")
                        # print(f"  Целевое значение: {target_y}")
                        # print(f"  Найденный вход: {x_found}")
                        # print(f"  Выход для найденного входа: {model_output}")
                        # print(f"  Ортогональное расстояние: {np.sqrt(sq_error)}")
                        # print()
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
        print(f"Успешных оптимизаций: {successful_optimizations}/{num_points} ({successful_optimizations/num_points*100:.1f}%)")
        print(f"Неудачных оптимизаций: {failed_optimizations}/{num_points} ({failed_optimizations/num_points*100:.1f}%)")
                
        # Возвращаем среднеквадратичную ошибку (RMSE)
        rmse = np.sqrt(total_sq_error / num_points)
        print(f"Абсолютная обратная ошибка (RMSE): {rmse:.6f}")
        return rmse

    def eval_abs_inv_error_torch(self, data: Union[np.ndarray, torch.Tensor], steps=50, lr=0.1, tol=1e-3, penalty=None, pred_limits=None, error_type=None) -> Union[float, Tuple[float, float]]:
        """
        Векторизованная версия абсолютной обратной ошибки (RMSE).
        Использует PyTorch для оптимизации. Штрафует несошедшиеся точки.

        Args:
            data: torch.Tensor или np.ndarray формы [N, D+1] — входы и выход.
            steps: Количество итераций оптимизации.
            lr: Скорость обучения.
            tol: Допуск к f(x') ≈ y.
            penalty: Штраф за неудачные оптимизации. Если None — берётся из диапазона X.

        Returns:
            float: RMSE между [x, f(x)] и восстановленным [x']
        """
        # Определяем устройство - используем CUDA, если доступно

        # print("in eval_abs_inv_error_torch with error_type =", error_type)

        # print("pred_limits in eval_abs_inv_error_torch 1:", pred_limits)

        if error_type == "abs" and self.abs_loss > 0:
            return self.abs_loss
        elif error_type == "spatial" and self.spatial_abs_loss > 0:
            return self.spatial_abs_loss
        elif self.abs_loss > 0 and self.spatial_abs_loss > 0:
            return (self.spatial_abs_loss, self.abs_loss)
        
        # print("pred_limits in eval_abs_inv_error_torch 2:", pred_limits)

        # print("before device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # print("pred_limits in eval_abs_inv_error_torch 3:", pred_limits)
        # Подготавливаем данные - переносим на нужное устройство
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32).to(device)
        else:
            data = data.to(device=device, dtype=torch.float32)


        # mean_x = data[:, :-1].mean(dim=0)
        # std_x = data[:, :-1].std(dim=0)
        # std_x[std_x == 0] = 1.0  # Avoid division by zero
        # data[:, :-1] = (data[:, :-1] - mean_x) / std_x

        # mean_y = data[:, -1].mean(dim=0)
        # std_y = data[:, -1].std(dim=0)
        # std_y[std_y == 0] = 1.0  # Avoid division by zero
        # data[:, -1] = (data[:, -1] - mean_y) / std_y    

        # print("pred_limits in eval_abs_inv_error_torch 4:", pred_limits)

        # Разделяем входные и выходные данные
        # print("before X_orig = data[:, :-1].clone().detach()")
        X_orig = data[:, :-1].clone().detach()
        y_target = data[:, -1].clone().detach() 

        # print("pred_limits in eval_abs_inv_error_torch 5:", pred_limits)

        # define max_error
        # print("pred_limits is None:", pred_limits is None)
        if pred_limits is None:
            pred_limits = torch.tensor(2*MAX_VALUE, dtype=torch.float32, device=device)
            # print("setted pred_limits to 2*MAX_VALUE")

        # print("pred_limits in eval_abs_inv_error_torch 6:", pred_limits)

        # print("before pred_limits.to(device=device, dtype=torch.float32).unsqueeze(0):", pred_limits)
        pred_limits = pred_limits.to(device=device, dtype=torch.float32).unsqueeze(0)
        # print("pred_limits in eval_abs_inv_error_torch 7:", pred_limits)
        # print("after pred_limits.to(device=device, dtype=torch.float32).unsqueeze(0):", pred_limits)
        # print("pred_limits.type():", pred_limits.type())
        # pred_limits = torch.max(pred_limits, torch.max(y_target.abs(), dim=0).values)
        pred_limits = torch.max(pred_limits, torch.max(y_target, dim=0).values - torch.min(y_target, dim=0).values)
        # print("y_target.isnan().any()", y_target.isnan().any())
        # print("pred_limits in eval_abs_inv_error_torch 8:", pred_limits)
        # print("pred_limits = torch.max(pred_limits, torch.max(y_target, dim=0).values - torch.min(y_target, dim=0).values)", pred_limits)
        # print(f"pred_limits: {pred_limits}")
        spatial_limits = torch.max(X_orig, dim=0).values - torch.min(X_orig, dim=0).values
        # print("pred_limits in eval_abs_inv_error_torch 9:", pred_limits)
        # max_spatial_error = torch.norm(spatial_limits).item()
        max_spatial_error = (torch.norm(spatial_limits))**2
        # print("pred_limits in eval_abs_inv_error_torch 10:", pred_limits)
        # print(f"spatial_limits: {spatial_limits}")
        # print("spatial_limits in eval_abs_inv_error_torch:", spatial_limits)
        # print("pred_limits in eval_abs_inv_error_torch:", pred_limits)
        limits = torch.cat((spatial_limits, pred_limits), dim=0)
        # print("limits in eval_abs_inv_error_torch:", limits)
        # max_error = torch.norm(limits).item()
        max_error = (torch.norm(limits))**2
        
        # Calculate clip value based on data limits (10% of max range)
        clip_value = 0.1 * torch.max(spatial_limits).item()
        # print(f"max_error: {max_error}")
        # quit()

        # Создаем тензор для оптимизации с градиентами
        X_optim = X_orig.clone().detach().requires_grad_(True)

        # Создаем оптимизатор
        optimizer = torch.optim.Adam([X_optim], lr=lr)
        criterion = torch.nn.MSELoss()

        # Выполняем оптимизацию
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Вычисляем предсказания модели
            y_pred = self.forward(varval=X_optim.to("cpu")).to(device)
                
            # Вычисляем функцию потерь
            # Compute squared L2 norm of concatenated (x,y) vectors for each point in batch
            # loss = torch.mean(torch.sum((X_optim - X_orig)**2, dim=1) + (y_pred - y_target)**2)
            # print("X_optim.shape, y_pred.shape, X_orig.shape, y_target.shape", X_optim.shape, y_pred.shape, X_orig.shape, y_target.shape)
            # print("y_target.unsqueeze(1).shape:", y_target.unsqueeze(1).shape)
            # print("torch.cat((X_optim, y_pred), dim=1).shape:", torch.cat((X_optim, y_pred), dim=1).shape   )
            # print("torch.cat((X_orig, y_target.unsqueeze(1)), dim=1).shape:", torch.cat((X_orig, y_target.unsqueeze(1)), dim=1).shape)
            loss = criterion(torch.cat((X_optim, y_pred), dim=1), torch.cat((X_orig, y_target.unsqueeze(1)), dim=1))

            if torch.isnan(loss) or torch.isinf(loss):
                if error_type == "abs":
                    print("Loss exploded in eval_abs_inv_error_torch during optimization for model:", self.to_math_expr(), "returning max_error =", max_error)
                    # quit()
                    return max_error.to("cpu")
                elif error_type == "spatial":
                    print("Loss exploded in eval_abs_inv_error_torch during optimization for model:", self.to_math_expr(), "returning max_spatial_error =", max_spatial_error)
                    # quit()
                    return max_spatial_error.to("cpu")
                else:
                    print("Loss exploded in eval_abs_inv_error_torch during optimization for model:", self.to_math_expr(), "returning (max_spatial_error, max_error) =", (max_spatial_error, max_error))
                    # quit()
                    return (max_spatial_error.to("cpu"), max_error.to("cpu"))
            
            # Вычисляем градиенты и делаем шаг оптимизации
            loss.backward()
            # Apply gradient clipping based on data limits
            torch.nn.utils.clip_grad_value_([X_optim], clip_value)
            optimizer.step()

        # Вычисляем финальное f(x') после оптимизации
        with torch.no_grad():
            # Убедимся, что все тензоры на одном устройстве перед вычислениями

            if error_type == "abs":
                y_final = self.forward(varval=X_optim.to("cpu")).to(device)

                # compute error
                concatenated_error = torch.cat((X_optim - X_orig, y_final - y_target.unsqueeze(1)), dim=1)
                if concatenated_error.isnan().any() or concatenated_error.isinf().any():
                    print(f"[{self.to_math_expr()}] concatenated_error contains NaN/Inf — skipping, returning max_error = {max_error}")
                    return max_error.to("cpu")
                # rmse = torch.mean(torch.sqrt(torch.sum(concatenated_error**2, dim=1))).item()
                # rmse = torch.mean(torch.norm(concatenated_error, dim=1)).item()
                # rmse = torch.mean(torch.norm(concatenated_error, dim=1)).to("cpu")
                mse = torch.mean((concatenated_error)**2).to("cpu")
                if mse.isnan() or mse.isinf():
                    print(f"[{self.to_math_expr()}] mse contains NaN/Inf — skipping, returning max_error = {max_error}")
                    return max_error.to("cpu")
                # print("rmse1:", rmse)
                # rmse = torch.sqrt(torch.mean(torch.sum(concatenated_error**2, dim=1))).item()
                # print("rmse2:", rmse)
                # quit()
                if mse > max_error:
                    # print(f"[{self.to_math_expr()}] rmse ({rmse}) > max_error ({max_error}) — skipping, returning max_error = {max_error}")
                    return max_error.to("cpu")

            elif error_type == "spatial":
                # spatial_rmse = torch.mean(torch.norm(X_optim - X_orig, dim=1)).item()
                # spatial_rmse = torch.mean(torch.norm(X_optim - X_orig, dim=1))
                spatial_mse = torch.mean((X_optim - X_orig)**2)
                if spatial_mse.isnan() or spatial_mse.isinf():
                    print(f"[{self.to_math_expr()}] spatial_mse contains NaN/Inf — skipping, returning max_spatial_error = {max_spatial_error}")
                    return max_spatial_error.to("cpu")
                if spatial_mse > max_spatial_error:
                    return max_spatial_error.to("cpu")
                mse = spatial_mse.to("cpu")
            
            else: # error_type == "abs" and "spatial"
                y_final = self.forward(varval=X_optim.to("cpu")).to(device)
                # print("y_final.dim()", y_final.dim())

                # spatial_rmse = torch.mean(torch.norm(X_optim - X_orig, dim=1))
                spatial_mse = torch.mean((X_optim - X_orig)**2)
                # print("spatial_rmse.dim()", spatial_rmse.dim())
                if spatial_mse.isnan() or spatial_mse.isinf():
                    print(f"[{self.to_math_expr()}] spatial_mse contains NaN/Inf — skipping, setting max_spatial_error = {max_spatial_error}")
                    spatial_mse = max_spatial_error
                if spatial_mse > max_spatial_error:
                    spatial_mse = max_spatial_error
                # print("spatial_rmse.dim()", spatial_rmse.dim())

                # compute abs error
                concatenated_error = torch.cat((X_optim - X_orig, y_final - y_target.unsqueeze(1)), dim=1)
                # print("concatenated_error.dim()", concatenated_error.dim())
                if concatenated_error.isnan().any() or concatenated_error.isinf().any():
                    print(f"[{self.to_math_expr()}] concatenated_error contains NaN/Inf — skipping, returning (spatial_mse, max_error) = ({spatial_mse}, {max_error})")
                    return (spatial_mse.to("cpu"), max_error.to("cpu"))
                # rmse = torch.mean(torch.sqrt(torch.sum(concatenated_error**2, dim=1))).item()
                # abs_rmse = torch.mean(torch.norm(concatenated_error, dim=1))
                abs_mse = torch.mean((concatenated_error)**2)
                # print("abs_rmse.dim()", abs_rmse.dim())
                if abs_mse.isnan() or abs_mse.isinf():
                    print(f"[{self.to_math_expr()}] abs_mse contains NaN/Inf — skipping, returning (spatial_mse, max_error) = ({spatial_mse}, {max_error})")
                    return (spatial_mse.to("cpu"), max_error.to("cpu"))
                # print("rmse1:", rmse)
                # rmse = torch.sqrt(torch.mean(torch.sum(concatenated_error**2, dim=1))).item()
                # print("rmse2:", rmse)
                # quit()
                if abs_mse > max_error:
                    # print(f"[{self.to_math_expr()}] abs_mse ({abs_mse}) > max_error ({max_error}) — skipping, returning max_error = {max_error}")
                    abs_mse = max_error

                mse = (spatial_mse.to("cpu"), abs_mse.to("cpu"))
                # print("type(rmse[0]), type(rmse[1])", type(rmse[0]), type(rmse[1]))

            return mse

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
                    expr = f"({left}) + ({right})"
                elif func_name == 'mult':
                    expr = f"({left}) * ({right})"
                elif func_name == 'div':
                    expr = f"({left}) / ({right})"
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

    def is_constant_node(self, node, data=None, tolerance=1e-3):
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
        # if hasattr(node, 'var_count'):
        #     num_vars = min(num_vars, node.var_count)
        
        # If there are no variables, it's a constant node
        if num_vars == 0:
            return True
        
        try:
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
    child.reset_losses()
    
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
        temp_child.reset_losses()
        
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
def mutation(tree: Tree, desired_functions: List[str] = None):
    """Change random node according to its parity with multivariable support"""
    
    mutant = safe_deepcopy(tree)
    mutant.reset_losses()
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
            if desired_functions:
                funcs = [func_info.func for func_info in get_functions_by_parity(1) 
                        if func_info.name not in ['const_', 'ident_'] and func_info.name in desired_functions]
            else:
                funcs = [func_info.func for func_info in get_functions_by_parity(1) 
                        if func_info.name not in ['const_', 'ident_']]
            if funcs:
                new_node.func = np.random.choice(funcs)
        else:
            if desired_functions:
                funcs = [func_info.func for func_info in get_functions_by_parity(2)
                        if func_info.name in desired_functions]
            else:
                funcs = [func_info.func for func_info in get_functions_by_parity(2)]
            if funcs:
                new_node.func = np.random.choice(funcs)

        mutant.math_expr = mutant.to_math_expr()
    return mutant

def mutation_constant(tree: Tree):
    """Change random constant in the tree"""
    
    mutant = safe_deepcopy(tree)
    mutant.reset_losses()
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
# def calculate_mse(output: torch.Tensor, example: torch.Tensor) -> torch.Tensor:
#     """Calculate Mean Squared Error using PyTorch operations
#     Args:
#         output: Predicted values
#         example: True values
#     Returns:
#         MSE value
#     """
#     NAN_PUNISHMENT = 1e10

#     min_val = torch.min(example)
#     max_val = torch.max(example)
#     diff = torch.abs(max_val - min_val)
#     if diff == 0:
#         diff = torch.tensor(1.0)

#     # Normalized MSE
#     # error = (1/example.shape[0]) * torch.sum((output - example)**2) / diff
#     error = torch.mean((output - example)**2)

#     if torch.isnan(error):
#         return NAN_PUNISHMENT * torch.isnan(output).sum()
#     return error

def get_stats(models, requires_forward_error, requires_inv_error, requires_abs_error, requires_spatial_abs_error):
    """Calculate RMSE statistics for a list of models"""
    errors = [model.error for model in models]  # Already RMSE values

    # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    mse_errors = [model.mse for model in models]

    forward_errors = [model.forward_loss for model in models] if requires_forward_error else None
    inv_errors = [model.inv_loss for model in models] if requires_inv_error else None
    abs_errors = [model.abs_loss for model in models] if requires_abs_error else None
    spatial_abs_errors = [model.spatial_abs_loss for model in models] if requires_spatial_abs_error else None
    return {
        # 'all_rmse': errors,
        # 'all_forward_rmse': forward_errors,
        # 'all_inv_rmse': inv_errors,
        # 'all_abs_rmse': abs_errors,
        # 'all_spatial_abs_rmse': spatial_abs_errors,
        
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'best_error': min(errors),

        # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        'mean_mse': np.mean(mse_errors),
        'median_mse': np.median(mse_errors),
        'best_mse': min(mse_errors),

        'mean_forward_loss': np.mean(forward_errors) if requires_forward_error else None,
        'median_forward_loss': np.median(forward_errors) if requires_forward_error else None,
        'best_forward_loss': min(forward_errors) if requires_forward_error else None,
        
        'mean_inv_loss': np.mean(inv_errors) if requires_inv_error else None,   
        'median_inv_loss': np.median(inv_errors) if requires_inv_error else None,
        'best_inv_loss': min(inv_errors) if requires_inv_error else None,
        
        'mean_abs_loss': np.mean(abs_errors) if requires_abs_error else None,
        'median_abs_loss': np.median(abs_errors) if requires_abs_error else None,
        'best_abs_loss': min(abs_errors) if requires_abs_error else None,
        
        'mean_spatial_abs_loss': np.mean(spatial_abs_errors) if requires_spatial_abs_error else None,
        'median_spatial_abs_loss': np.median(spatial_abs_errors) if requires_spatial_abs_error else None,
        'best_spatial_abs_loss': min(spatial_abs_errors) if requires_spatial_abs_error else None,
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
                 models[i].depth == models[j].depth and \
                 # np.isclose(models[i].error, models[j].error, atol=1e-6) and \ # 'error' is sum of all losses so if we check all losses, we dont need this
                 np.isclose(models[i].forward_loss, models[j].forward_loss, atol=1e-6) and \
                 np.isclose(models[i].abs_loss, models[j].abs_loss, atol=1e-6) and \
                 np.isclose(models[i].spatial_abs_loss, models[j].spatial_abs_loss, atol=1e-6) and \
                 np.isclose(models[i].inv_loss, models[j].inv_loss, atol=1e-6)):
                #  np.isclose(models[i].domain_loss, models[j].domain_loss, atol=1e-6) and \
                models[j].is_unique = False

    # Step 2: Compute domination only among unique models
    for i in range(n):
        if not models[i].is_unique:
            continue
        for j in range(n):
            if i == j or not models[j].is_unique:
                continue
            # Check if model i dominates model j
            if (models[i].forward_loss <= models[j].forward_loss and \
                models[i].abs_loss <= models[j].abs_loss and \
                models[i].inv_loss <= models[j].inv_loss and \
                models[i].spatial_abs_loss <= models[j].spatial_abs_loss and \
                models[i].error <= models[j].error and \
                # models[i].max_num_of_node <= models[j].max_num_of_node and \
                models[i].depth <= models[j].depth and \
                
                (models[i].depth < models[j].depth or \
                 models[i].forward_loss < models[j].forward_loss or \
                 models[i].abs_loss < models[j].abs_loss or \
                 models[i].inv_loss < models[j].inv_loss or \
                 models[i].spatial_abs_loss < models[j].spatial_abs_loss or \
                 models[i].error < models[j].error)):
                # models[i].domain_loss <= models[j].domain_loss and \
                #  models[i].domain_loss < models[j].domain_loss or \
                
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
    objectives = ['forward_loss', 'inv_loss', 'abs_loss', 'spatial_abs_loss', 'error', 'depth'] # 'error' is sum of all losses
    

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
def set_evaluated_error(models_to_eval: List[Tree], 
                        training_data: Union[np.ndarray, torch.Tensor], 
                        epoch: int, 
                        requires_forward_error: bool, 
                        requires_inv_error: bool, 
                        requires_abs_error: bool, 
                        requires_spatial_abs_error: bool):
    """Set evaluated error for models"""
    print("len(models_to_eval):", len(models_to_eval))
    for model in models_to_eval:
        try:
            # Проверяем, нужно ли вычислять инверсную ошибку
            # use_inv_error = epoch % INV_ERROR_EVAL_FREQ == 0 and model.requires_grad #and epoch > 40
            use_forward_error = requires_forward_error
            use_inv_error = epoch % INV_ERROR_EVAL_FREQ == 0 and requires_inv_error #and epoch > 40
            use_abs_inv_error = epoch % ABS_INV_ERROR_EVAL_FREQ == 0 and requires_abs_error #and epoch > 40
            use_spatial_abs_inv_error = epoch % SPATIAL_INV_ERROR_EVAL_FREQ == 0 and requires_spatial_abs_error #and epoch > 40
            # Вычисляем ошибку модели
            model.eval_tree_error(training_data, 
                                  forward_error=use_forward_error,
                                  inv_error=use_inv_error, 
                                  abs_error=use_abs_inv_error, 
                                  spatial_abs_error=use_spatial_abs_inv_error)
            
            # Проверяем валидность результатов
            if np.isnan(model.error) or np.isinf(model.error):
                print(f"Warning in set_evaluated_error: Model {model.math_expr} depth={model.depth} has invalid error {model.error}")
                model.error = float('inf')
                model.forward_loss = float('inf')
                if requires_inv_error:
                    model.inv_loss = float('inf')
                if requires_abs_error:
                    model.abs_loss = float('inf')
                if requires_spatial_abs_error:
                    model.spatial_abs_loss = float('inf')
        except Exception as e:
            print(f"Error evaluating model {model.math_expr}: {e}")
            model.error = float('inf')
            model.forward_loss = float('inf')
            if use_inv_error:
                model.inv_loss = float('inf')
            if use_abs_inv_error:
                model.abs_loss = float('inf')
            if use_spatial_abs_inv_error:
                model.spatial_abs_loss = float('inf')

def tournament(model1: Tree, model2: Tree):
    """Tournament selection"""
    # Select two random models
    if  model1.depth < model2.depth and \
        model1.error < model2.error and \
        model1.forward_loss < model2.forward_loss and \
        model1.abs_loss < model2.abs_loss and \
        model1.inv_loss < model2.inv_loss and \
        model1.spatial_abs_loss < model2.spatial_abs_loss:
        # model1.domain_loss < model2.domain_loss and \
        return model1
    
    elif model1.depth > model2.depth and \
        model1.error > model2.error and \
        model1.forward_loss > model2.forward_loss and \
        model1.abs_loss > model2.abs_loss and \
        model1.inv_loss > model2.inv_loss and \
        model1.spatial_abs_loss > model2.spatial_abs_loss:
        # model1.domain_loss > model2.domain_loss and \
        return model2
    
    else:
# TODO: if NSGA is used, else compute crowding distance for tournament selection =========================================
        model_with_max_crowding_distance = max(model1, model2, key=lambda x: x.crowding_distance)
        # print("model_with_max_crowding_distance:", model_with_max_crowding_distance.math_expr, "crowding_distance:", model_with_max_crowding_distance.crowding_distance)
        return model_with_max_crowding_distance
    
    

# evolution process with crossover, mutation, and constant optimization(if CONST_OPT is True)
def evolution(num_of_epochs,
              models,
              training_data,
              population_size, 
              requires_forward_error, 
              requires_inv_error, 
              requires_abs_error, 
              requires_spatial_abs_error, 
              desired_functions=None) -> Tuple[Dict[str, Tree], Dict[str, float]]:
    """evolution process with crossover, mutation, and constant optimization"""
    
    # for model in models:
    #     print(model.math_expr, "num_vars:", model.num_vars, end="\n")

    # initialize arrays for tracking progress
    mean_sum_of_losses_arr = np.zeros(num_of_epochs)
    median_sum_of_losses_arr = np.zeros(num_of_epochs)
    best_sum_of_losses_arr = np.zeros(num_of_epochs)

    # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    mean_mse_arr = np.zeros(num_of_epochs)
    median_mse_arr = np.zeros(num_of_epochs)
    best_mse_arr = np.zeros(num_of_epochs)
    
    if requires_forward_error:
        mean_forward_loss_arr = np.zeros(num_of_epochs)
        median_forward_loss_arr = np.zeros(num_of_epochs)
        best_forward_loss_arr = np.zeros(num_of_epochs)
    
    if requires_inv_error:
        mean_inv_loss_arr = np.zeros(num_of_epochs)
        median_inv_loss_arr = np.zeros(num_of_epochs)
        best_inv_loss_arr = np.zeros(num_of_epochs)
    
    if requires_abs_error:
        mean_abs_loss_arr = np.zeros(num_of_epochs)
        median_abs_loss_arr = np.zeros(num_of_epochs)
        best_abs_loss_arr = np.zeros(num_of_epochs)
    
    if requires_spatial_abs_error:
        mean_spatial_abs_loss_arr = np.zeros(num_of_epochs)
        median_spatial_abs_loss_arr = np.zeros(num_of_epochs)
        best_spatial_abs_loss_arr = np.zeros(num_of_epochs)
    

    
    # initialize best model and best error
    # best_model = safe_deepcopy(models[0])
    min_forward_loss = float('inf')
    best_sum_of_losses = float('inf')

    # Model selection (choose method of selection based on params)

    # select POPULATION_SIZE models for init population for evolution 
    print("Evaluating models for init population...")
    set_evaluated_error(models, 
                        training_data, 
                        0, 
                        requires_forward_error, 
                        requires_inv_error, 
                        requires_abs_error, 
                        requires_spatial_abs_error)
    
    print("Selecting models for init population...")
    selection_params = {
        "population_size": population_size,
        "method":          SELECTION_METHOD # "NSGA-II" by default
    }

    # selection of models for init population////////////////////////////////////////////////////////////////////////////////////////////////////////
    models = model_selection(models, selection_params)
    # selection_params["population_size"] = NUM_OF_MODELS_TO_SELECT # uncomment if we want to select NUM_OF_MODELS_TO_SELECT models for init population

    # evolution loop////////////////////////////////////////////////////////////////////////////////////////////////////////
    print("Starting evolution...")
    for epoch in range(num_of_epochs):

        # Initialize set to track unique expressions
        unique_expressions = {model.math_expr for model in models}

        print("EPOCH:", epoch)
        
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
                    while new_model.depth > new_model.max_depth:
                        new_model = crossover(model, tournament(models[np.random.randint(0, len(models))], models[np.random.randint(0, len(models))]))
                    
                    expr = new_model.math_expr
                    if expr not in unique_expressions:
                        unique_expressions.add(expr)
                        models_to_eval.append(new_model)
                        break
            print(f"Number of unique expressions after crossover: {len(unique_expressions)}")

            # Evaluate models (choose variant of evaluation based on requires_grad)
            set_evaluated_error(models_to_eval, training_data, epoch, requires_forward_error, requires_inv_error, requires_abs_error, requires_spatial_abs_error)

            # select the best models
            models_to_eval = model_selection(models_to_eval, selection_params)
            unique_expressions = {model.math_expr for model in models_to_eval}

            # mutate from best models
            new_models = []
            for model in models_to_eval:
                # Try up to 10 times to get a unique expression through mutation
                for _ in range(10):
                    mutated_model = mutation(model, desired_functions)
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
        set_evaluated_error(models_to_eval, training_data, epoch, requires_forward_error, requires_inv_error, requires_abs_error, requires_spatial_abs_error)

        # Best models selection (choose method of selection based on params)
        selected_models = model_selection(models_to_eval, selection_params)
        
        # Ensure we have models in case of no selection by NSGA-II
        if not selected_models:
            print(f"No selected models, using {selection_params['method']}")
            models_to_eval.sort(key=lambda x: x.error)
            models = [safe_deepcopy(model) for model in models_to_eval[:NUM_OF_MODELS_TO_SELECT]]
        else:
            models = selected_models

        # Track progress (sum of losses statistics)
        epoch_stats = get_stats(models, 
                               requires_forward_error, 
                               requires_inv_error, 
                               requires_abs_error, 
                               requires_spatial_abs_error)
        mean_sum_of_losses_arr[epoch] = epoch_stats['mean_error']
        median_sum_of_losses_arr[epoch] = epoch_stats['median_error']
        best_sum_of_losses_arr[epoch] = epoch_stats['best_error']

        # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        mean_mse_arr[epoch] = epoch_stats['mean_mse']
        median_mse_arr[epoch] = epoch_stats['median_mse']
        best_mse_arr[epoch] = epoch_stats['best_mse']

        if requires_forward_error:
            mean_forward_loss_arr[epoch] = epoch_stats['mean_forward_loss']
            median_forward_loss_arr[epoch] = epoch_stats['median_forward_loss']
            best_forward_loss_arr[epoch] = epoch_stats['best_forward_loss']

        if requires_inv_error:
            mean_inv_loss_arr[epoch] = epoch_stats['mean_inv_loss']
            median_inv_loss_arr[epoch] = epoch_stats['median_inv_loss']
            best_inv_loss_arr[epoch] = epoch_stats['best_inv_loss']

        if requires_abs_error:
            mean_abs_loss_arr[epoch] = epoch_stats['mean_abs_loss']
            median_abs_loss_arr[epoch] = epoch_stats['median_abs_loss']
            best_abs_loss_arr[epoch] = epoch_stats['best_abs_loss']

        if requires_spatial_abs_error:
            mean_spatial_abs_loss_arr[epoch] = epoch_stats['mean_spatial_abs_loss']
            median_spatial_abs_loss_arr[epoch] = epoch_stats['median_spatial_abs_loss']
            best_spatial_abs_loss_arr[epoch] = epoch_stats['best_spatial_abs_loss']

        # sorting models by error and selecting the best model as a model with minimal error//////////////////////////////////////////////////////////////////////////////////////////////////////// 
        # models.sort(key=lambda x: x.error) # i need to get fronts[0] from nsga-ii selection
        best_front = None
        fronts = non_dominated_sort(models)
        if len(fronts) > 0:
            best_front = fronts[0]
        else:
            models.sort(key=lambda x: x.error) # i need to get fronts[0] from nsga-ii selection
            best_front = models

        best_models : Dict[str, Tree] = {}
        all_criterions = ["error", "mse", "forward_loss", "inv_loss", "abs_loss", "spatial_abs_loss"]
        required_criterions = [True, False, requires_forward_error, requires_inv_error, requires_abs_error, requires_spatial_abs_error] # True means that the error criterion is always required
        for criterion in all_criterions:
            if required_criterions[all_criterions.index(criterion)]:
                best_models[criterion] = min(best_front, key=lambda model: getattr(model, criterion))
                if epoch % 10 == 0:
                    print(f"Best model by {criterion}: {best_models[criterion].math_expr} - {getattr(best_models[criterion], criterion)}")
            else:
                best_models[criterion] = None

    # Store stats
    stats = {
        "mean_error_arr": mean_sum_of_losses_arr,
        "median_error_arr": median_sum_of_losses_arr,
        "best_error_arr": best_sum_of_losses_arr,

        # MSE stats here ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        "mean_mse_arr": mean_mse_arr,
        "median_mse_arr": median_mse_arr,
        "best_mse_arr": best_mse_arr,

        "mean_forward_loss_arr": mean_forward_loss_arr if requires_forward_error else None,
        "median_forward_loss_arr": median_forward_loss_arr if requires_forward_error else None,
        "best_forward_loss_arr": best_forward_loss_arr if requires_forward_error else None,

        "mean_inv_loss_arr": mean_inv_loss_arr if requires_inv_error else None,
        "median_inv_loss_arr": median_inv_loss_arr if requires_inv_error else None,
        "best_inv_loss_arr": best_inv_loss_arr if requires_inv_error else None,

        "mean_abs_loss_arr": mean_abs_loss_arr if requires_abs_error else None,
        "median_abs_loss_arr": median_abs_loss_arr if requires_abs_error else None,
        "best_abs_loss_arr": best_abs_loss_arr if requires_abs_error else None,

        "mean_spatial_abs_loss_arr": mean_spatial_abs_loss_arr if requires_spatial_abs_error else None,
        "median_spatial_abs_loss_arr": median_spatial_abs_loss_arr if requires_spatial_abs_error else None,
        "best_spatial_abs_loss_arr": best_spatial_abs_loss_arr if requires_spatial_abs_error else None,
    }
    
    # Plot error history using the new visualization function
    # if not SAVE_RESULTS:
    #     from visual import plot_rmse_history
    #     plot_rmse_history(mean_rmse_arr, median_rmse_arr, best_rmse_arr)
    
    # print final results
    # print(f"Final min_forward_loss: {min_forward_loss}")
    # print(f"Best sum of losses achieved: {best_sum_of_losses:.6f}")
    # print(f"Final mean sum of losses: {epoch_stats['mean_error']:.6f}")
    # print(f"Final median sum of losses: {epoch_stats['median_error']:.6f}")
    # print(f"Total number of unique expressions found: {len(unique_expressions)}")

    # return best model
    # print("selected_models:", [model.math_expr for model in models])
    # print("\n")
    # print("sorted_selected_models:", [model.math_expr for model in sorted(models, key=lambda x: x.error)])
    # for i, model in enumerate(sorted(models, key=lambda x: x.error)):
    #     print(f"{i}: {model.math_expr}")
    #     print(f"   sum of losses: {model.error:.6f}")
    #     print(f"   forward_loss: {model.forward_loss:.6f}")
    #     print(f"   inv_loss: {getattr(model, 'inv_loss', 0):.6f}")
    #     print(f"   abs_loss: {getattr(model, 'abs_loss', 0):.6f}")
    #     print(f"   spatial_abs_loss: {getattr(model, 'spatial_abs_loss', 0):.6f}")
    #     print(f"   depth: {model.depth}")
    #     # print(f"   domain_loss: {model.domain_loss:.6f}")
    #     print(f"   max_num_of_node: {model.max_num_of_node}")

    # return models[0], stats
    # print("best_models:", best_models)
    return best_models, stats