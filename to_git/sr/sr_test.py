import pandas as pd
import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Dict

from nodes import FUNCTIONS, FunctionCategory, get_functions_by_category, get_functions_by_parity, to_tensor
from visual import plot_results
from sr_tree import *
from parameters import *

# load data from csv files for specified function or all functions
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

# create variable nodes based on input data dimensions
def create_vars(data: pd.DataFrame, requires_grad: bool = False) -> List[Node]:
    """Create variable nodes based on input data dimensions"""
    num_vars = data.shape[1] - 1  # Last column is target
    return [Node("var", 1, FUNCTIONS['ident_'].func, [i], requires_grad=requires_grad) 
            for i in range(num_vars)]

# Create initial nodes for model generation using function categories
#TODO change and fix errors(not used, create trees for parity 2, wrong tree/node creation)///////////////////////////////////////////////////////
def create_start_nodes(data: pd.DataFrame, requires_grad: bool = False) -> List[Node]:
    """Create initial nodes for model generation using function categories"""
    nodes_list = []
    vars = create_vars(data, requires_grad=requires_grad)
    print("vars:", [var.data for var in vars])
    num_vars = len(vars)
    nodes_list.extend(vars)
    
    # Add function nodes by category
    for category in FunctionCategory:
        category_funcs = get_functions_by_category(category)
        print(f"category: {category}")
        for func_info in category_funcs:
            print(f"func_info.name: {func_info.name}")
            if func_info.name in ['const_', 'ident_']:
                continue  # Skip as we handle these separately
            print(f"func_info.parity: {func_info.parity}")
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
                            
                    # Add some combinations with random constants if REQUIRES_CONST is False
                    if not REQUIRES_CONST:
                        for var in vars:
                            # Create random constant node
                            const_val = np.random.uniform(-10, 10)
                            const_node = Node("ident", 1, FUNCTIONS['const_'].func,
                                           torch.tensor([float(const_val)], dtype=torch.float32),
                                           requires_grad=False)
                            nodes_list.append(Node("func", 2, func_info.func, 
                                                [get_deepcopy(var), const_node],
                                                requires_grad=requires_grad))
                            if func_info.name in ['pow_', 'div_']:
                                nodes_list.append(Node("func", 2, func_info.func, 
                                                [const_node, get_deepcopy(var)],
                                                requires_grad=requires_grad))
    
    print("nodes_list:", [node.func_info.name for node in nodes_list])
    return nodes_list

# reshape input and evaluate model (model.forward)
def evaluate_model(model: Tree, X_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Evaluate a model on input data using PyTorch tensors"""
    if isinstance(X_data, np.ndarray):
        X_data = to_tensor(X_data, requires_grad=model.requires_grad)
    if X_data.dim() == 1:
        X_data = X_data.reshape(-1, 1)
    # return model.start_node.eval_node(varval=X_data)
    return model.forward(varval=X_data)


# build random tree up to max_depth
def build_random_tree(
    num_vars: int,
    max_depth: int,
    all_functions: List[FunctionInfo],
    requires_grad: bool = False,
    allow_constants: bool = False
) -> Node:
    """
    Recursively builds a random tree up to max_depth.
    - If depth=0 or random stop, returns a variable node
    - If depth>0, randomly selects a function and builds required number of subtrees
    
    Args:
        num_vars: Number of variables available
        max_depth: Maximum depth of the tree
        all_functions: List of available functions
        requires_grad: Whether to enable gradient computation
        allow_constants: Whether to allow creation of constant nodes
    """
    # Higher probability to stop at lower depths for shorter trees
    if max_depth <= 0 or np.random.rand() < 0.2:
        # When REQUIRES_CONST is True or constants are not allowed, only create variable nodes
        if REQUIRES_CONST or not allow_constants:
            var_idx = np.random.randint(0, num_vars)
            return safe_deepcopy(Node("var", 1, FUNCTIONS['ident_'].func, [var_idx], requires_grad=requires_grad))
        else:
            # Choose between variable and constant when constants are allowed
            if np.random.rand() < 0.7:  # Bias towards variables
                var_idx = np.random.randint(0, num_vars)
                return safe_deepcopy(Node("var", 1, FUNCTIONS['ident_'].func, [var_idx], requires_grad=requires_grad))
            else:
                # Constant node
                c_val = np.random.uniform(-10, 10)
                c_tensor = torch.tensor([float(c_val)], dtype=torch.float32)
                return safe_deepcopy(Node("ident", 1, FUNCTIONS['const_'].func, c_tensor, requires_grad=False))
    
    # Build function node
    func_info = np.random.choice(all_functions)
    # Exclude const_ and ident_ from function nodes
    while func_info.name in ['const_', 'ident_']:
        func_info = np.random.choice(all_functions)
    
    if func_info.parity == 1:
        # Unary function
        child = build_random_tree(num_vars, max_depth - 1, all_functions, requires_grad, allow_constants)
        return safe_deepcopy(Node("func", 1, func_info.func, [child], requires_grad=requires_grad))
    else:
        # Binary function
        left = build_random_tree(num_vars, max_depth - 1, all_functions, requires_grad, allow_constants)
        right = build_random_tree(num_vars, max_depth - 1, all_functions, requires_grad, allow_constants)
        return safe_deepcopy(Node("func", 2, func_info.func, [left, right], requires_grad=requires_grad))


# create diverse population of trees
def create_diverse_population(
    data: pd.DataFrame,
    population_size: int = 50,
    max_depth: int = 3,
    requires_grad: bool = False
) -> List[Tree]:
    """
    Generates a list of 'population_size' trees:
    1. Guarantees each function appears at least once as a root
    2. Creates one tree with a constant node
    3. Fills remaining population with random trees (without constants)
    """
    num_vars = data.shape[1] - 1
    # Collect all functions except const_ and ident_
    all_funcs = [f_info for name, f_info in FUNCTIONS.items() 
                if name not in ['const_', 'ident_']]
    
    population = []
    pop_set = set()

    # 1) Guarantee each function appears at least once as root
    for func_info in all_funcs:
        if func_info.parity == 1:
            # Unary function
            child_node = build_random_tree(num_vars, max_depth - 1, all_funcs, requires_grad, allow_constants=False)
            root_node = Node("func", 1, func_info.func, [child_node], requires_grad=requires_grad)
        else:
            # Binary function
            left_node = build_random_tree(num_vars, max_depth - 1, all_funcs, requires_grad, allow_constants=False)
            right_node = build_random_tree(num_vars, max_depth - 1, all_funcs, requires_grad, allow_constants=False)
            root_node = Node("func", 2, func_info.func, [left_node, right_node], requires_grad=requires_grad)
        
        tree = Tree(root_node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        population.append(safe_deepcopy(tree))
        pop_set.add(tree.to_math_expr())
    
    # 2) Create one tree with a constant node if REQUIRES_CONST is False
    if not REQUIRES_CONST:
        # Create a binary function with one constant child
        func_info = np.random.choice([f for f in all_funcs if f.parity == 2])  # Choose a binary function
        var_node = build_random_tree(num_vars, 1, all_funcs, requires_grad, allow_constants=False)  # Create a simple variable node
        const_val = np.random.uniform(-10, 10)
        const_node = Node("ident", 1, FUNCTIONS['const_'].func,
                       torch.tensor([float(const_val)], dtype=torch.float32),
                       requires_grad=False)
        
        # Randomly decide if constant should be left or right child
        if np.random.rand() < 0.5:
            root_node = Node("func", 2, func_info.func, [var_node, const_node], requires_grad=requires_grad)
        else:
            root_node = Node("func", 2, func_info.func, [const_node, var_node], requires_grad=requires_grad)
        
        tree = Tree(root_node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        if tree.to_math_expr() not in pop_set:
            population.append(safe_deepcopy(tree))
            pop_set.add(tree.to_math_expr())
    
    # 3) Fill remaining population with random trees (without constants)
    while len(population) < population_size:
        node = build_random_tree(num_vars, max_depth, all_funcs, requires_grad, allow_constants=False)
        tree = Tree(node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        if tree.to_math_expr() not in pop_set:
            population.append(safe_deepcopy(tree))
            pop_set.add(tree.to_math_expr())
    
    return population

# save model results to files, including plots and expressions
def save_model_results(model: Tree, X_data: np.ndarray, Y_data: np.ndarray, Y_pred: np.ndarray, 
                      target_func: str = None, error_history: np.ndarray = None):
    """
    Save model results to files, including plots and expressions.
    
    Args:
        model: The trained Tree model
        X_data: Input data
        Y_data: True output data
        Y_pred: Predicted output data
        target_func: String representation of the target function (if known)
        error_history: Array of error values during training
    """
    from datetime import datetime
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f'run_{timestamp}')
    os.makedirs(run_dir)
    
    # Save model information to text file with UTF-8 encoding
    with open(os.path.join(run_dir, 'model_info.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Model Information\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Mathematical Expression:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{model.to_math_expr()}\n\n")
        
        if target_func:
            f.write("Target Function:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{target_func}\n\n")
        
        f.write("Model Structure:\n")
        f.write("-" * 30 + "\n")
        # Convert tree structure to plain ASCII
        tree_str = model.print_tree().replace('└', '\\-').replace('├', '|-').replace('─', '-')
        f.write(f"{tree_str}\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Final Error: {model.error:.6f}\n")
        if hasattr(model, 'min_loss'):
            f.write(f"Minimum Loss: {model.min_loss:.6f}\n")
    
    # Save prediction plot
    import matplotlib.pyplot as plt
    
    # Handle different input dimensions
    if len(X_data.shape) == 1:  # 1D input
        plt.figure(figsize=(10, 6))
        plt.scatter(X_data, Y_data, label='True', alpha=0.5)
        plt.scatter(X_data, Y_pred, label='Predicted', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
    elif len(X_data.shape) == 2 and X_data.shape[1] == 2:  # 2D input
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_data[:, 0], X_data[:, 1], Y_data, label='True', alpha=0.5)
        ax.scatter(X_data[:, 0], X_data[:, 1], Y_pred, label='Predicted', alpha=0.5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('y')
    else:  # Higher dimensions - create error vs predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_data, Y_pred, alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        # Add perfect prediction line
        min_val = min(Y_data.min(), Y_pred.min())
        max_val = max(Y_data.max(), Y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Model Predictions vs True Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save error history plot if available
    if error_history is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(error_history, linewidth=2)
        plt.title('Training Error History')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.yscale('log')  # Use log scale for better visualization
        plt.savefig(os.path.join(run_dir, 'error_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

# run evolutionary algorithm 
def run_evolution(X_data: Union[np.ndarray, torch.Tensor], 
                 Y_data: Union[np.ndarray, torch.Tensor], 
                 num_epochs: int = NUM_OF_EPOCHS,
                 population_size: int = NUM_OF_START_MODELS,
                 mutation_prob: float = MUTATION_PROB,
                 max_depth: int = MAX_DEPTH,
                 requires_grad: bool = False,
                 target_func: str = None) -> Tree:
    """Run the evolutionary algorithm with the given parameters"""

    # Convert inputs to PyTorch tensors
    if isinstance(X_data, np.ndarray):
        X_data = to_tensor(X_data, requires_grad=requires_grad)
    if isinstance(Y_data, np.ndarray):
        Y_data = to_tensor(Y_data, requires_grad=False)  # Target doesn't need gradients
    
    # Ensure X_data is 2D
    if X_data.dim() == 1:
        X_data = X_data.reshape(-1, 1)

    data = torch.column_stack((X_data, Y_data))
    
    # Create DataFrame from detached tensors for initialization
    data_np = data.detach().numpy()
    print("Creating diverse population...")
    start_models = create_diverse_population(
        pd.DataFrame(data_np),
        requires_grad=requires_grad,
        population_size=population_size,
        max_depth=DEPTH_OF_START_MODELS  # Initial trees are simpler
    )
    
    # Set evolution parameters
    print("Setting evolution parameters...")
    for model in start_models:
        model.max_depth = max_depth
        model.mutation_prob = mutation_prob
    
    print("Starting evolution...")
    final_model = evolution(num_epochs, start_models, data)
    
    # Evaluate and visualize results
    print("Evaluating and visualizing results...")
    Y_aprox = evaluate_model(final_model, X_data)
    # print("Y_aprox.shape:", Y_aprox.shape)
    
    # Convert tensors to numpy for plotting
    X_plot = X_data.detach().numpy()
    Y_true = Y_data.detach().numpy()
    Y_pred = Y_aprox.detach().numpy().reshape(-1)
    
    # Plot results
    if X_plot.shape[1] == 1:
        X_plot = X_plot.flatten()  # Flatten for 1D plotting
        plot_results(X_plot, Y_true, Y_pred, "Model Comparison for " + target_func + f"\nPredicted: {final_model.to_math_expr()}")
    elif X_plot.shape[1] == 2:
        plot_results(X_plot, Y_true, Y_pred, "Model Comparison for " + target_func + f"\nPredicted: {final_model.to_math_expr()}")
    
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

    # Save results if flag is set
    if SAVE_RESULTS:
        print("Saving results...")
        save_model_results(
            model=final_model,
            X_data=X_plot,
            Y_data=Y_true,
            Y_pred=Y_pred,
            target_func=target_func,
            error_history=final_model.error_history if hasattr(final_model, 'error_history') else None
        )

    return final_model

# main function
if __name__ == "__main__":
    # Example usage with single variable
    print("Single variable example:")
    X_data = np.linspace(-10, 10, 1000)  # Reshape to (100, 1)
    # X_data = np.linspace(-10, 10, 100)
    Y_data = X_data**5  # Example target function: x^2
    # Y_data = np.e ** (-1*(X_data+4)**2)/2  # Example target function: x^2
    # Y_data = np.power(X_data.flatten(), 2)  # Example target function: x^5
    # # Y_data = 5*np.ones(100)  # Example target function: x^5
    # # Y_data = 1/(X_data) # Example target function: x^5
    
    if not FIXED_SEED:
        models = []
        for i in range(10):
            model = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="x^5")
            models.append(model)
        for model in models:
            print(model.to_math_expr())
            print(model.error)
            print(model.forward_loss)
        best_model = min(models, key=lambda x: x.error)
        print("Best model:")
        print(best_model.to_math_expr())
        print(best_model.error)
        print(best_model.forward_loss)
    else:
        model1 = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="x^5")
    
    # Example usage with multiple variables
    # print("\nMultiple variables example:")
    # X_data = np.random.uniform(-5, 5, (500, 2))  # 2 variables
    # Y_data = X_data[:, 0]**2 + X_data[:, 1]**2  # Example target function: x0^2 + x1^2
    # # Y_data = 2*X_data[:, 0] + 5*X_data[:, 1]  # Example target function: x0^2 + x1^2
    # if not FIXED_SEED:
    #     models = []
    #     for i in range(10):
    #         model = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="x0^2 + x1^2")
    #         models.append(model)
    #     for model in models:
    #         print(model.to_math_expr())
    #         print(model.error)
    #         print(model.forward_loss)
    #     best_model = min(models, key=lambda x: x.error)
    #     print("Best model:")
    #     print(best_model.to_math_expr())
    #     print(best_model.error)
    #     print(best_model.forward_loss)    
    # else:
    #     model2 = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="x0^2 + x1^2")
    #     print(model2)

    # # # Load and process dataset examples
    # data_list = load_data()
    # for func_name, data in data_list:
    #     print(f"\nProcessing function: {func_name}")
    #     run_evolution(data.iloc[:, :-1].values, data.iloc[:, -1].values, 
    #                  requires_grad=REQUIRES_GRAD, target_func=func_name)

