import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import os
import time
from typing import Union, Callable, Tuple, List, Optional
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# =============== Model Function for sqrt_log ===============

def sqrt_log_model(x: Union[np.ndarray, float]) -> float:
    """
    Target model function: np.sqrt(x) * np.log(x + 1)
    Based on equation7 from benchmark.py
    """
    # Make sure x is a scalar value, not an array
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = float(x.item())  # Get scalar value from array
        else:
            x = float(x[0])  # Get first element if array has multiple values
    
    # Handle negative input values
    if x < 0:
        return 0.0
        
    return np.sqrt(x) * np.log(x + 1)

# =============== Dataset Loading ===============

def load_dataset(filename: str = "log_plus_pow.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from the new_datasets folder.
    
    Args:
        filename: Name of the CSV file to load
        
    Returns:
        Tuple containing:
        - X_data: Input values array
        - y_data: Output values
    """
    # Path to the dataset
    dataset_path = os.path.join("..", "new_datasets", filename)
    
    # Load data from CSV
    try:
        data = pd.read_csv(dataset_path)
        print(f"Successfully loaded dataset: {filename}")
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        # For 1D data, extract as flat array, not nested arrays
        if 'x' in data.columns and 'y' not in data.columns:
            X_data = data['x'].values  # Flat array for 1D data
        else:
            # For 2D data, use array of [x, y] pairs
            X_data = data[['x', 'y']].values
            
        y_data = data['output'].values
        
        print(f"Features shape: {X_data.shape}")
        print(f"Target shape: {y_data.shape}")
        
        return X_data, y_data
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# =============== Wrapper Functions for Optimization ===============

def wrapped_model_func(x: Union[np.ndarray, float], target_y: float, 
                      model_func: Callable, input_dims: int) -> float:
    """
    Function to minimize for finding x such that model(x) = target_y.
    Works with both 1D and multi-dimensional inputs.
    
    Args:
        x: Input value(s) - will be a 1D array from scipy.optimize
        target_y: Target y value
        model_func: Model function
        input_dims: Input dimensionality (1 or more)
        
    Returns:
        Squared difference between model(x) and target_y
    """
    try:
        if input_dims == 1:
            # For 1D, make sure we're passing a scalar
            x_val = float(x[0]) if hasattr(x, '__getitem__') else float(x)
            y_pred = model_func(x_val)
        else:
            # For multi-dimensional case, pass the whole array
            y_pred = model_func(x)
        
        return (y_pred - target_y)**2
    except Exception as e:
        # Return a high error value if there's any exception
        print(f"Error in wrapped_model_func: {e}")
        return 1.0e10

# =============== Inverse Error Evaluation ===============

def eval_inv_error(X_data: np.ndarray, y_data: np.ndarray, model_func: Callable) -> Tuple[np.ndarray, float, List[float], np.ndarray, np.ndarray]:
    """
    Evaluate inverse error using scipy.optimize.minimize with Nelder-Mead method.
    For each y value, tries to find x such that model_func(x) = y.
    Works with both 1D and multi-dimensional inputs.
    
    Args:
        X_data: Original x values (1D array or 2D array for multi-dimensional inputs)
        y_data: Original y values (potentially with noise)
        model_func: Model function to invert
        
    Returns:
        Tuple containing:
        - x_pred: Predicted x values for each y
        - avg_inv_error: Average inverse error
        - point_errors: List of errors for each point
        - X_data_subset: Subset of X_data used for computation
        - y_data_subset: Subset of y_data used for computation
    """
    # Determine input dimensionality
    input_dims = 1 if X_data.ndim == 1 else X_data.shape[1]
    n_samples = len(y_data)
    
    # For large datasets, use a subset for faster computation
    if n_samples > 100:
        subset_indices = np.random.choice(n_samples, 100, replace=False)
        X_data_subset = X_data[subset_indices]
        y_data_subset = y_data[subset_indices]
        n_samples = 100
    else:
        X_data_subset = X_data
        y_data_subset = y_data
    
    # Initialize arrays for results
    if input_dims == 1:
        x_pred = np.zeros(n_samples)
    else:
        x_pred = np.zeros((n_samples, input_dims))
    
    total_inv_error = 0.0
    point_errors = []
    optimization_times = []
    
    print(f"Starting inverse error evaluation for {n_samples} points using Nelder-Mead method...")
    print(f"Input dimensionality: {input_dims}")
    
    for i in range(n_samples):
        # Get initial guess and target y value
        if input_dims == 1:
            x0 = np.array([float(X_data_subset[i])])  # Ensure it's a simple 1D array with a single float value
        else:
            x0 = X_data_subset[i].copy()
        
        target_y = float(y_data_subset[i])
        
        # Make sure x0 is not near zero for the sqrt_log model to avoid domain issues
        if input_dims == 1 and model_func.__name__ == 'sqrt_log_model' and x0[0] < 0.1:
            x0[0] = 0.1
        
        # Find x that minimizes (f(x) - y)^2 using Nelder-Mead
        try:
            start_time = time.time()
            result = optimize.minimize(
                wrapped_model_func,
                x0=x0,
                args=(target_y, model_func, input_dims),
                method='Nelder-Mead',
                options={'maxiter': 100, 'disp': False}
            )
            opt_time = time.time() - start_time
            optimization_times.append(opt_time)
            
            if result.success:
                # Get the optimized x value(s)
                if input_dims == 1:
                    x_found = result.x[0]
                    x_pred[i] = x_found
                    # Compute Euclidean distance between found x and original x
                    error = float(np.sqrt((x_found - X_data_subset[i]) ** 2))
                else:
                    x_found = result.x
                    x_pred[i] = x_found
                    # Compute Euclidean distance between found x and original x
                    error = float(np.sqrt(np.sum((x_found - X_data_subset[i]) ** 2)))
                
                total_inv_error += error
                point_errors.append(error)
            else:
                # Penalize failed inversions
                print(f"Optimization failed for point {i}: {result.message}")
                if input_dims == 1:
                    x_pred[i] = float(X_data_subset[i])
                else:
                    x_pred[i] = X_data_subset[i].copy()
                error = 10.0
                total_inv_error += error
                point_errors.append(error)
        except Exception as e:
            # Print error but continue with other samples
            print(f"Optimization error for point {i}: {str(e)}")
            if input_dims == 1:
                x_pred[i] = float(X_data_subset[i])
            else:
                x_pred[i] = X_data_subset[i].copy()
            error = 10.0
            total_inv_error += error
            point_errors.append(error)
    
    # Print optimization statistics
    avg_time = np.mean(optimization_times) if optimization_times else 0
    print(f"Average optimization time per point: {avg_time:.4f} seconds")
    print(f"Total inverse error: {total_inv_error:.4f}")
    avg_inv_error = total_inv_error / n_samples
    print(f"Average inverse error: {avg_inv_error:.4f}")
    
    return x_pred, avg_inv_error, point_errors, X_data_subset, y_data_subset

# =============== Visualization Functions ===============

def visualization_plot(X_data: np.ndarray, y_data: np.ndarray, x_pred: np.ndarray, 
                      model_func: Callable):
    """
    Visualize the results of inverse error calculation.
    Automatically handles both 1D and multi-dimensional cases.
    
    Args:
        X_data: Original x values
        y_data: Original y values (potentially with noise)
        x_pred: Predicted x values from inverse error calculation
        model_func: Model function
    """
    # Determine input dimensionality
    input_dims = 1 if X_data.ndim == 1 else X_data.shape[1]
    
    if input_dims == 1:
        # 1D visualization
        visualization_plot_1d(X_data, y_data, x_pred, model_func)
    elif input_dims == 2:
        # 2D visualization
        visualization_plot_2d(X_data, y_data, x_pred, model_func)
    else:
        print(f"Visualization not supported for {input_dims} dimensions")

def visualization_plot_1d(X_data: np.ndarray, y_data: np.ndarray, x_pred: np.ndarray, 
                         model_func: Callable):
    """
    Visualize the results of inverse error calculation for 1D inputs.
    
    Args:
        X_data: Original x values
        y_data: Original y values (potentially with noise)
        x_pred: Predicted x values from inverse error calculation
        model_func: Model function
    """
    # Create a figure with appropriate size
    plt.figure(figsize=(14, 10))
    
    # Generate points for a smooth model curve
    min_x = max(0.01, min(X_data.min(), x_pred.min()) - 1)  # Ensure positive values for sqrt_log
    max_x = max(X_data.max(), x_pred.max()) + 1
    x_curve = np.linspace(min_x, max_x, 1000)
    
    # Calculate y values for each x in x_curve
    y_curve = np.zeros_like(x_curve)
    for i, x in enumerate(x_curve):
        y_curve[i] = model_func(float(x))
    
    # Plot the model function as a blue curve
    plt.plot(x_curve, y_curve, 'b-', linewidth=2, label='Model function')
    
    # Plot original data points
    plt.scatter(X_data, y_data, color='g', s=50, alpha=0.7, label='Original data points')
    
    # Calculate y-values for predicted x points
    y_pred = np.zeros_like(x_pred)
    for i, x in enumerate(x_pred):
        y_pred[i] = model_func(float(x))
    
    # Plot predicted points
    plt.scatter(x_pred, y_pred, color='r', s=50, alpha=0.7, label='Predicted points (model_func(x_pred))')
    
    # Connect each original data point to its corresponding predicted point
    for i in range(len(X_data)):
        # Draw a horizontal line from (X_data[i], y_data[i]) to (x_pred[i], y_data[i])
        plt.plot([X_data[i], x_pred[i]], [y_data[i], y_data[i]], 'r-', alpha=0.5)
        
        # Draw a vertical line from (x_pred[i], y_data[i]) to (x_pred[i], y_pred[i])
        plt.plot([x_pred[i], x_pred[i]], [y_data[i], y_pred[i]], 'r--', alpha=0.3)
    
    # Add labels and legend
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.title("Inverse Error Visualization (1D) using Nelder-Mead for sqrt_log", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def visualization_plot_2d(X_data: np.ndarray, y_data: np.ndarray, x_pred: np.ndarray, 
                         model_func: Callable):
    """
    Visualize the results of inverse error calculation for 2D inputs.
    
    Args:
        X_data: Original x values (2D array)
        y_data: Original y values (potentially with noise)
        x_pred: Predicted x values from inverse error calculation
        model_func: Model function
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate predicted y values
    y_pred = np.array([model_func(x) for x in x_pred])
    
    # Plot original data points
    ax.scatter(X_data[:, 0], X_data[:, 1], y_data, 
               color='g', s=50, alpha=0.7, label='Original data points')
    
    # Plot predicted points
    ax.scatter(x_pred[:, 0], x_pred[:, 1], y_pred, 
               color='r', s=50, alpha=0.7, label='Predicted points')
    
    # Connect original points to predicted points with lines
    for i in range(len(X_data)):
        ax.plot([X_data[i, 0], x_pred[i, 0]], 
                [X_data[i, 1], x_pred[i, 1]], 
                [y_data[i], y_data[i]], 'r-', alpha=0.3)
        
        ax.plot([x_pred[i, 0], x_pred[i, 0]], 
                [x_pred[i, 1], x_pred[i, 1]], 
                [y_data[i], y_pred[i]], 'r--', alpha=0.2)
    
    # Generate a grid for the surface
    x_min, x_max = min(min(X_data[:, 0]), min(x_pred[:, 0])) - 1, max(max(X_data[:, 0]), max(x_pred[:, 0])) + 1
    y_min, y_max = min(min(X_data[:, 1]), min(x_pred[:, 1])) - 1, max(max(X_data[:, 1]), max(x_pred[:, 1])) + 1
    
    # Create a coarser grid if the range is very large
    grid_size = 30
    if max(x_max - x_min, y_max - y_min) > 1000:
        grid_size = 20
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    zz = np.zeros(xx.shape)
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = model_func(np.array([xx[i, j], yy[i, j]]))
    
    # Plot the surface
    surf = ax.plot_surface(xx, yy, zz, alpha=0.3, cmap='viridis', edgecolor='none')
    
    # Set equal aspect ratio for all axes
    max_range = max([
        max(X_data[:, 0].max(), x_pred[:, 0].max()) - min(X_data[:, 0].min(), x_pred[:, 0].min()),
        max(X_data[:, 1].max(), x_pred[:, 1].max()) - min(X_data[:, 1].min(), x_pred[:, 1].min()),
        max(y_data.max(), y_pred.max()) - min(y_data.min(), y_pred.min())
    ])
    
    mid_x = (X_data[:, 0].max() + X_data[:, 0].min()) * 0.5
    mid_y = (X_data[:, 1].max() + X_data[:, 1].min()) * 0.5
    mid_z = (y_data.max() + y_data.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    # Set labels and title
    ax.set_xlabel('X₁', fontsize=14)
    ax.set_ylabel('X₂', fontsize=14)
    ax.set_zlabel('Y', fontsize=14)
    
    ax.set_title("Inverse Error Visualization (2D) using Nelder-Mead for sqrt_log", fontsize=16)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# =============== Main Function ===============

if __name__ == "__main__":
    print("Testing inverse error calculation for sqrt_log dataset")
    
    # Load dataset
    X_data, y_data = load_dataset("log_plus_pow.csv")
    
    # Evaluate inverse error
    x_pred, avg_inv_error, point_errors, X_subset, y_subset = eval_inv_error(X_data, y_data, sqrt_log_model)
    
    # Visualize results - use the subset data
    visualization_plot(X_subset, y_subset, x_pred, sqrt_log_model)
    
    print("\nInverse error evaluation complete. Visualization displayed.") 