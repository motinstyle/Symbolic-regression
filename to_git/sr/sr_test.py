import pandas as pd
import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Dict

from nodes import FUNCTIONS, FunctionCategory, get_functions_by_category, get_functions_by_parity, to_tensor
from visual import plot_results
from sr_tree import *


def load_data(func_name: Optional[str] = None) -> List[Tuple[str, pd.DataFrame]]:
    """Load data from CSV files for specified function or all functions"""
    data_folder = "../datasets"
    test_func = [func for func in os.listdir(data_folder) if func.endswith(".csv")]
    test_func = [func.replace(".csv", "") for func in test_func]
    data_list = []
    target_funcs = [func_name] if func_name else test_func
    
    for func in target_funcs:
        try:
            data = pd.read_csv(os.path.join(data_folder, f"{func}.csv"))
            data_list.append((func, data))
        except FileNotFoundError:
            print(f"Warning: No data file found for function {func}")
    
    return data_list

def create_vars(data: pd.DataFrame, requires_grad: bool = False) -> List[Node]:
    """Create variable nodes based on input data dimensions"""
    num_vars = data.shape[1] - 1  # Last column is target
    return [Node("var", 1, FUNCTIONS['ident_'].func, [i], requires_grad=requires_grad) 
            for i in range(num_vars)]

def create_start_nodes(data: pd.DataFrame, requires_grad: bool = False) -> List[Node]:
    """Create initial nodes for model generation using function categories"""
    nodes_list = []
    vars = create_vars(data, requires_grad=requires_grad)
    num_vars = len(vars)
    nodes_list.extend(vars)

    # Add constant nodes
    prime_consts = [1, 2, 3]
    for val in prime_consts:
        nodes_list.append(Node("ident", 1, FUNCTIONS['const_'].func, 
                             torch.tensor([float(val)], dtype=torch.float32),
                             requires_grad=False))  # Constants don't need gradients
    
    # Add function nodes by category
    for category in FunctionCategory:
        category_funcs = get_functions_by_category(category)
        for func_info in category_funcs:
            if func_info.name in ['const_', 'ident_']:
                continue  # Skip as we handle these separately
                
            if func_info.parity == 1:  # Unary functions
                # Create unary functions with each variable
                for var in vars:
                    nodes_list.append(Node("func", 1, func_info.func, 
                                         [get_deepcopy(var)],
                                         requires_grad=requires_grad))
            else:  # Binary functions
                if num_vars >= 2:
                    # Create some basic combinations
                    for i in range(num_vars):
                        for j in range(i, num_vars):  # Including i=j for self-operations
                            nodes_list.append(Node("func", 2, func_info.func, 
                                                [get_deepcopy(vars[i]), get_deepcopy(vars[j])],
                                                requires_grad=requires_grad))
                            
                    # Add some combinations with constants
                    for var in vars:
                        for const in prime_consts:
                            const_node = Node("ident", 1, FUNCTIONS['const_'].func,
                                           torch.tensor([float(const)], dtype=torch.float32),
                                           requires_grad=False)
                            nodes_list.append(Node("func", 2, func_info.func, 
                                                [get_deepcopy(var), const_node],
                                                requires_grad=requires_grad))
    
    return nodes_list

def create_start_models(data: pd.DataFrame, 
                       requires_grad: bool = False,
                       population_size: int = 70,
                       max_depth: int = 3
                      ) -> List[Tree]:
    """Create diverse initial models"""
    models = create_diverse_population(
        data=data,
        population_size=population_size,
        max_depth=max_depth,
        requires_grad=requires_grad
    )
    return models

def evaluate_model(model: Tree, X_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Evaluate a model on input data using PyTorch tensors"""
    if isinstance(X_data, np.ndarray):
        X_data = to_tensor(X_data, requires_grad=model.requires_grad)
    if X_data.dim() == 1:
        X_data = X_data.reshape(-1, 1)
    # return model.start_node.eval_node(varval=X_data)
    return model.forward(varval=X_data)

def build_random_tree(
    num_vars: int,
    max_depth: int,
    all_functions: List[FunctionInfo],
    const_candidates: List[float],
    requires_grad: bool = False
) -> Node:
    """
    Recursively builds a random tree up to max_depth.
    - If depth=0 or random stop, returns either a variable or constant node
    - If depth>0, randomly selects a function and builds required number of subtrees
    """
    # Higher probability to stop at lower depths for shorter trees
    if max_depth <= 0 or np.random.rand() < 0.2:
        # Choose between variable and constant
        if np.random.rand() < 0.5:
            # Variable node
            var_idx = np.random.randint(0, num_vars)
            return Node("var", 1, FUNCTIONS['ident_'].func, [var_idx], requires_grad=requires_grad)
        else:
            # Constant node
            c_val = np.random.choice(const_candidates)
            c_tensor = torch.tensor([float(c_val)], dtype=torch.float32)
            return Node("ident", 1, FUNCTIONS['const_'].func, c_tensor, requires_grad=False)
    
    # Build function node
    func_info = np.random.choice(all_functions)
    # Exclude const_ and ident_ from function nodes
    while func_info.name in ['const_', 'ident_']:
        func_info = np.random.choice(all_functions)
    
    if func_info.parity == 1:
        # Unary function
        child = build_random_tree(num_vars, max_depth - 1, all_functions, const_candidates, requires_grad)
        return Node("func", 1, func_info.func, [child], requires_grad=requires_grad)
    else:
        # Binary function
        left = build_random_tree(num_vars, max_depth - 1, all_functions, const_candidates, requires_grad)
        right = build_random_tree(num_vars, max_depth - 1, all_functions, const_candidates, requires_grad)
        return Node("func", 2, func_info.func, [left, right], requires_grad=requires_grad)

def create_diverse_population(
    data: pd.DataFrame,
    population_size: int = 50,
    max_depth: int = 3,
    requires_grad: bool = False
) -> List[Tree]:
    """
    Generates a list of 'population_size' trees:
    1. Guarantees each function appears at least once as a root
    2. Fills remaining population with random trees
    """
    num_vars = data.shape[1] - 1
    # Collect all functions except const_ and ident_
    all_funcs = [f_info for name, f_info in FUNCTIONS.items() 
                if name not in ['const_', 'ident_']]
    
    # Set of possible constants
    const_candidates = [1, 2, 3, 0.5, 2.0]
    
    population = []
    
    # 1) Guarantee each function appears at least once as root
    for func_info in all_funcs:
        if func_info.parity == 1:
            # Unary function
            child_node = build_random_tree(num_vars, max_depth - 1, all_funcs, const_candidates, requires_grad)
            root_node = Node("func", 1, func_info.func, [child_node], requires_grad=requires_grad)
        else:
            # Binary function
            left_node = build_random_tree(num_vars, max_depth - 1, all_funcs, const_candidates, requires_grad)
            right_node = build_random_tree(num_vars, max_depth - 1, all_funcs, const_candidates, requires_grad)
            root_node = Node("func", 2, func_info.func, [left_node, right_node], requires_grad=requires_grad)
        
        tree = Tree(root_node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        population.append(tree)
    
    # 2) Fill remaining population with random trees
    while len(population) < population_size:
        node = build_random_tree(num_vars, max_depth, all_funcs, const_candidates, requires_grad)
        tree = Tree(node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        population.append(tree)
    
    return population

def run_evolution(X_data: Union[np.ndarray, torch.Tensor], 
                 Y_data: Union[np.ndarray, torch.Tensor], 
                 num_epochs: int = 100,
                 population_size: int = 70,
                 mutation_prob: float = 0.15,
                 max_depth: int = 10,
                 requires_grad: bool = False) -> Tree:
    """Run the evolutionary algorithm with the given parameters"""
    # Convert inputs to PyTorch tensors
    if isinstance(X_data, np.ndarray):
        X_data = to_tensor(X_data, requires_grad=requires_grad)
    if isinstance(Y_data, np.ndarray):
        Y_data = to_tensor(Y_data, requires_grad=False)  # Target doesn't need gradients
    
    if X_data.dim() == 1:
        X_data = X_data.reshape(-1, 1)
    
    data = torch.column_stack((X_data, Y_data))
    
    # Create DataFrame from detached tensors for initialization
    data_np = data.detach().numpy()
    start_models = create_start_models(
        pd.DataFrame(data_np),
        requires_grad=requires_grad,
        population_size=population_size,
        max_depth=4  # Initial trees are simpler
    )
    
    # Set evolution parameters
    for model in start_models:
        model.max_depth = max_depth
        model.mutation_prob = mutation_prob
    
    final_model = evolution(num_epochs, start_models, data)
    
    # Evaluate and visualize results
    Y_aprox = evaluate_model(final_model, X_data)
    
    # Convert tensors to numpy for plotting
    X_plot = X_data.detach().numpy()
    Y_true = Y_data.detach().numpy()
    Y_pred = Y_aprox.detach().numpy()
    
    # Plot results
    if X_data.shape[1] < 3:
        if X_data.shape[1] == 1:
            X_plot = X_plot.reshape(-1)  # Flatten for 1D plotting
        plot_results(X_plot, Y_true, Y_pred, "Model Comparison")
    
    # Print the final model
    print("\nFinal Model:")
    print(final_model.print_tree())
    print("\nMathematical Expression:")
    print(final_model.to_math_expr())
    
    if requires_grad:
        print("\nGradients:")
        gradients = final_model.grad
        print("gradients:", gradients)
        print("gradients.shape:", gradients.shape)

    return final_model


REQUIRES_GRAD = False
if __name__ == "__main__":
    # Example usage with single variable
    print("Single variable example:")
    X_data = np.linspace(-10, 10, 100)
    Y_data = np.power(X_data, 5)  # Example target function: x^5
    model1 = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD)
    # model1 = run_evolution(X_data, Y_data, requires_grad=False)
    print("model1.grad:", model1.grad)
    
    # Example usage with multiple variables
    print("\nMultiple variables example:")
    X_data = np.random.uniform(-5, 5, (100, 2))  # 2 variables
    Y_data = X_data[:, 0]**2 + X_data[:, 1]**2  # Example target function: x0^2 + x1^2
    # model2 = run_evolution(X_data, Y_data, requires_grad=True)
    model2 = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD)

    # Load and process dataset examples
    data_list = load_data()
    for func_name, data in data_list:
        print(f"\nProcessing function: {func_name}")
        # run_evolution(data.iloc[:, :-1].values, data.iloc[:, -1].values, requires_grad=True)
        run_evolution(data.iloc[:, :-1].values, data.iloc[:, -1].values, requires_grad=REQUIRES_GRAD)

