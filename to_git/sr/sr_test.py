import pandas as pd
import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Dict, Callable
import time

from nodes import FUNCTIONS, FunctionCategory, get_functions_by_category, get_functions_by_parity, to_tensor
from visual import plot_results
from sr_tree import *
from parameters import *

# Define a class to hold population statistics
class PopulationStats:
    def __init__(self):
        # Initialize dictionaries to store statistics
        self.function_counts = {}
        self.var_count = 0
        self.const_count = 0
        self.total_leaf_nodes = 0
    
    def get_var_const_ratio(self):
        """Get the ratio of variables to all leaf nodes."""
        if self.total_leaf_nodes == 0:
            return 0
        return self.var_count / self.total_leaf_nodes
    
    def get_const_ratio(self):
        """Get the ratio of constants to all leaf nodes."""
        if self.total_leaf_nodes == 0:
            return 0
        return self.const_count / self.total_leaf_nodes
    
    def display_stats(self):
        """Display all collected statistics."""
        print("\n" + "="*50)
        print("POPULATION STATISTICS")
        print("="*50)
        
        # Print function usage table
        print("\nFunction Usage:")
        print("-"*40)
        print(f"{'Function':<15} {'Count':<10} {'Percentage':<15}")
        print("-"*40)
        
        total_funcs = sum(self.function_counts.values())
        for func_name, count in sorted(self.function_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_funcs * 100) if total_funcs > 0 else 0
            print(f"{func_name:<15} {count:<10} {percentage:.2f}%")
        
        # Print leaf node statistics
        print("\nLeaf Node Statistics:")
        print("-"*40)
        print(f"Total leaf nodes: {self.total_leaf_nodes}")
        print(f"Variables: {self.var_count} ({self.get_var_const_ratio()*100:.2f}%)")
        print(f"Constants: {self.const_count} ({self.get_const_ratio()*100:.2f}%)")
        
        # Plot statistics
        self.plot_stats()
    
    def plot_stats(self):
        """Plot statistics as histograms and pie charts."""
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot function usage histogram
        if self.function_counts:
            func_names = list(self.function_counts.keys())
            counts = list(self.function_counts.values())
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            func_names = [func_names[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            ax1.bar(range(len(func_names)), counts, tick_label=func_names)
            ax1.set_title('Function Usage in Population')
            ax1.set_xlabel('Function')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add count values on top of bars
            for i, v in enumerate(counts):
                ax1.text(i, v + 0.5, str(v), ha='center')
        
        # Plot leaf node distribution pie chart
        if self.total_leaf_nodes > 0:
            ax2.pie([self.var_count, self.const_count], 
                   labels=['Variables', 'Constants'],
                   autopct='%1.1f%%',
                   startangle=90,
                   colors=['#66b3ff', '#ff9999'])
            ax2.set_title('Leaf Node Distribution')
        
        plt.tight_layout()
        plt.show()

# Add a function to collect node statistics
def collect_node_stats(node: Node, stats: PopulationStats):
    """
    Recursively collect statistics about nodes in the tree.
    
    Args:
        node: The node to analyze
        stats: The PopulationStats object to update
    """
    if node is None:
        return
    
    # Count function usage
    if node.t_name == "func":
        func_name = node.func.__name__
        if func_name in stats.function_counts:
            stats.function_counts[func_name] += 1
        else:
            stats.function_counts[func_name] = 1
        
        # Recursively process children
        if isinstance(node.data, list):
            for child in node.data:
                collect_node_stats(child, stats)
    
    # Count leaf nodes (variables and constants)
    elif node.t_name == "var":
        stats.var_count += 1
        stats.total_leaf_nodes += 1
    
    elif node.t_name == "ident":
        stats.const_count += 1
        stats.total_leaf_nodes += 1

# load data from csv files for specified function or all functions
def load_data(func_name: Optional[str] = None) -> List[Tuple[str, pd.DataFrame]]:
    """Load data from CSV files for specified function or all functions"""
    # data_folder = "../datasets"
    data_folder = "../new_datasets"
    # data_folder = "../dataset_bodya"

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
                            
                    # Add some combinations with random constants if CONST_OPT is False
                    if not CONST_OPT:
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
    allow_constants: bool = ALLOW_CONSTANTS,
    current_depth: int = 0
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
        current_depth: Current depth in the recursion
    """
    # Higher probability to stop at lower depths for shorter trees
    remaining_depth = max_depth - current_depth
    if remaining_depth <= 0 or np.random.rand() < 0.2:
        # When CONST_OPT is True or constants are not allowed, only create variable nodes
        if CONST_OPT or not allow_constants:
            var_idx = np.random.randint(0, num_vars)
            node = safe_deepcopy(Node("var", 1, FUNCTIONS['ident_'].func, [var_idx], requires_grad=requires_grad))
            node.var_count = 1  # Variable node has var_count of 1
            return node
        else:
            # Choose between variable and constant when constants are allowed
            if np.random.rand() < VARIABLES_OR_CONSTANTS_PROB:  # Bias towards variables
                var_idx = np.random.randint(0, num_vars)
                node = safe_deepcopy(Node("var", 1, FUNCTIONS['ident_'].func, [var_idx], requires_grad=requires_grad))
                node.var_count = 1  # Variable node has var_count of 1
                return node
            else:
            # Constant node
                c_val = np.random.uniform(-10, 10)
                c_tensor = torch.tensor([float(c_val)], dtype=torch.float32)
                node = safe_deepcopy(Node("ident", 1, FUNCTIONS['const_'].func, c_tensor, requires_grad=False))
                node.var_count = 0  # Constant node has var_count of 0
                return node
    
    # Build function node
    func_info = np.random.choice(all_functions)
    # Exclude const_ and ident_ from function nodes
    while func_info.name in ['const_', 'ident_']:
        func_info = np.random.choice(all_functions)
    
    if func_info.parity == 1:
        # Unary function
        child = build_random_tree(num_vars, max_depth, all_functions, requires_grad, allow_constants, current_depth + 1)
        node = safe_deepcopy(Node("func", 1, func_info.func, [child], requires_grad=requires_grad))
        node.var_count = child.var_count  # Unary function inherits child's var_count
        return node
    else:
        # Binary function
        left = build_random_tree(num_vars, max_depth, all_functions, requires_grad, allow_constants, current_depth + 1)
        right = build_random_tree(num_vars, max_depth, all_functions, requires_grad, allow_constants, current_depth + 1)
        node = safe_deepcopy(Node("func", 2, func_info.func, [left, right], requires_grad=requires_grad))
        node.var_count = left.var_count + right.var_count  # Binary function sums children's var_counts
        return node


# create diverse population of trees
def create_diverse_population(
    data: pd.DataFrame,
    population_size: int = 50,
    max_depth: int = 3,
    requires_grad: bool = False
) -> Tuple[List[Tree], PopulationStats]:
    """
    Generates a list of 'population_size' trees and collects statistics:
    1. Guarantees each function appears at least once as a root
    2. Creates one tree with a constant node
    3. Fills remaining population with random trees (without constants)
    
    Returns:
        Tuple containing:
        - List of Tree objects (the population)
        - PopulationStats object with usage statistics
    """
    num_vars = data.shape[1] - 1
    # Collect all functions except const_ and ident_
    all_funcs = [f_info for name, f_info in FUNCTIONS.items() 
                if name not in ['const_', 'ident_']]
    
    population = []
    pop_set = set()
    
    # Initialize statistics object
    stats = PopulationStats()
    
    # 1) Guarantee each function appears at least once as root
    for func_info in all_funcs:
        if func_info.parity == 1:
            # Unary function
            child_node = build_random_tree(num_vars, max_depth - 1, all_funcs, requires_grad, allow_constants=ALLOW_CONSTANTS)
            root_node = Node("func", 1, func_info.func, [child_node], requires_grad=requires_grad)
            root_node.var_count = child_node.var_count  # Set var_count based on child
        else:
            # Binary function
            left_node = build_random_tree(num_vars, max_depth - 1, all_funcs, requires_grad, allow_constants=ALLOW_CONSTANTS)
            right_node = build_random_tree(num_vars, max_depth - 1, all_funcs, requires_grad, allow_constants=ALLOW_CONSTANTS)
            root_node = Node("func", 2, func_info.func, [left_node, right_node], requires_grad=requires_grad)
            root_node.var_count = left_node.var_count + right_node.var_count  # Sum var_counts of children
        
        tree = Tree(root_node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        tree.update_var_counts()  # Update variable counts
        tree.update_depth()       # Update tree depth
        
        # Collect statistics for this tree
        collect_node_stats(tree.start_node, stats)
        
        population.append(safe_deepcopy(tree))
        pop_set.add(tree.math_expr)
    
    # 2) Create one tree with a constant node if CONST_OPT is False
    if not CONST_OPT:
        # Create a binary function with one constant child
        func_info = np.random.choice([f for f in all_funcs if f.parity == 2])  # Choose a binary function
        var_node = build_random_tree(num_vars, 1, all_funcs, requires_grad, allow_constants=ALLOW_CONSTANTS)  # Create a simple variable node
        const_val = np.random.uniform(-10, 10)
        const_node = Node("ident", 1, FUNCTIONS['const_'].func,
                       torch.tensor([float(const_val)], dtype=torch.float32),
                       requires_grad=False)
        const_node.var_count = 0  # Constant has var_count of 0
        
        # Randomly decide if constant should be left or right child
        if np.random.rand() < 0.5:
            root_node = Node("func", 2, func_info.func, [var_node, const_node], requires_grad=requires_grad)
            root_node.var_count = var_node.var_count  # Only var_node contributes to var_count
        else:
            root_node = Node("func", 2, func_info.func, [const_node, var_node], requires_grad=requires_grad)
            root_node.var_count = var_node.var_count  # Only var_node contributes to var_count
        
        tree = Tree(root_node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        tree.update_var_counts()  # Update variable counts
        tree.update_depth()       # Update tree depth
        
        if tree.math_expr not in pop_set:
            # Collect statistics for this tree
            collect_node_stats(tree.start_node, stats)
            
            population.append(safe_deepcopy(tree))
            pop_set.add(tree.math_expr)
    
    # 3) Fill remaining population with random trees (without constants)
    while len(population) < population_size:
        node = build_random_tree(num_vars, max_depth, all_funcs, requires_grad, allow_constants=ALLOW_CONSTANTS)
        tree = Tree(node, requires_grad=requires_grad)
        tree.num_vars = num_vars
        tree.update_var_counts()  # Update variable counts
        tree.update_depth()       # Update tree depth
        
        if tree.math_expr not in pop_set:
            # Collect statistics for this tree
            collect_node_stats(tree.start_node, stats)
            
            population.append(safe_deepcopy(tree))
            pop_set.add(tree.math_expr)
    
    # Display statistics after population is created
    stats.display_stats()
    
    return population, stats

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
        f.write(f"{model.math_expr}\n\n")
        
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

    start_time = time.time()

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
    start_models, stats = create_diverse_population(
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
    end_time = time.time()
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
        plot_results(X_plot, Y_true, Y_pred, "Model Comparison for " + target_func + f"\nPredicted: {final_model.math_expr}")
    elif X_plot.shape[1] == 2:
        plot_results(X_plot, Y_true, Y_pred, "Model Comparison for " + target_func + f"\nPredicted: {final_model.math_expr}")
    
    # Print the final model
    print("\nFinal Model:")
    print(final_model.print_tree())
    print("\nMathematical Expression:")
    print(final_model.math_expr)
    # from simplifying import simplify_expression
    # print("Simplified Expression:", simplify_expression(final_model.math_expr))
    print(f"=============Time taken: {end_time - start_time:.2f} seconds=============")
    
    # if requires_grad:
    #     print("\nGradients:")
    #     gradients = final_model.grad
    #     print("gradients:", gradients)
    #     print("gradients.shape:", gradients.shape)

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
    X_data = np.linspace(-30, 30, 1000)  # Reshape to (100, 1)
    # X_data = np.linspace(-10, 10, 100)
    # Y_data = X_data**5  # Example target function: x^2
    # Y_data = np.exp(-1*(X_data-4)**2)  # Example target function: x^2
    Y_data = np.e ** (-1*(X_data)**2)  # Example target function: e^(-x^2)
    # Y_data = np.e ** (-1*(X_data)**2)  # Example target function: e^(-x^2)
    # Y_data = 2*np.sin(0.5*X_data) + 5*np.cos(2*X_data)  # Example target function: x^2
    # Y_data = np.power(X_data.flatten(), 2)  # Example target function: x^5
    # # Y_data = 5*np.ones(100)  # Example target function: x^5
    # # Y_data = 1/(X_data) # Example target function: x^5
    
    # if not FIXED_SEED:
    #     models = []
    #     for i in range(10):
    #         model = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="e^(-x^2)")
    #         models.append(model)
    #     for model in models:
    #         print(model.math_expr)
    #         print(model.error)
    #         print(model.forward_loss)
    #     best_model = min(models, key=lambda x: x.error)
    #     print("Best model:")
    #     print(best_model.math_expr)
    #     print(best_model.error)
    #     print(best_model.forward_loss)
    # else:
    #     model1 = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="e^(-x^2)")
    
    # Example usage with multiple variables
    print("\nMultiple variables example:")
    X_data = np.random.uniform(-5, 5, (100, 2))  # 2 variables
    Y_data = X_data[:, 0]**2 + X_data[:, 1]**2  # Example target function: x0^2 + x1^2
    # # Y_data = 2*X_data[:, 0] + 5*X_data[:, 1]  # Example target function: x0^2 + x1^2
    # if not FIXED_SEED:
    #     models = []
    #     for i in range(10):
    #         model = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="x0^2 + x1^2")
    #         models.append(model)
    #     for model in models:
    #         print(model.math_expr)
    #         print(model.error)
    #         print(model.forward_loss)
    #     best_model = min(models, key=lambda x: x.error)
    #     print("Best model:")
    #     print(best_model.math_expr)
    #     print(best_model.error)
    #     print(best_model.forward_loss)    
    # else:
    #     model2 = run_evolution(X_data, Y_data, requires_grad=REQUIRES_GRAD, target_func="x0^2 + x1^2")
    #     print(model2)

    # Load and process dataset examples
    data_list = load_data()
    for func_name, data in data_list:
        print(f"\nProcessing function: {func_name}")
        run_evolution(data.iloc[:, :-1].values, data.iloc[:, -1].values, 
                     requires_grad=REQUIRES_GRAD, target_func=func_name)


