import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import torch
from typing import Union, Callable, Tuple, List, Optional
import time
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# =============== Model Functions ===============

# 1D model function (sine with exponential decay)
def model_1d(x: np.ndarray) -> np.ndarray:
    """Target 1D model function: e^(-x²)"""
    return np.exp(-1 * x**2)

def dmodel_1d(x: np.ndarray) -> np.ndarray:
    """Derivative of the 1D target model function"""
    return -2 * x * np.exp(-1 * x**2)

# 2D model function (paraboloid)
def model_2d(x: np.ndarray) -> np.ndarray:
    """Target 2D model function: x₁² + x₂²"""
    return x[0]**2 + x[1]**2

# =============== Dataset Creation ===============

def create_dataset(n_points: int = 100, input_dims: int = 1, 
                  x_min: float = -5, x_max: float = 5, 
                  noise_scale: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a dataset with noise based on the selected model function.
    
    Args:
        n_points: Number of data points
        input_dims: Input dimensionality (1 or more)
        x_min: Minimum x value for all dimensions
        x_max: Maximum x value for all dimensions
        noise_scale: Scale of noise to add
        
    Returns:
        Tuple containing:
        - X_data: Input values array (shape depends on input_dims)
        - y_data: Clean y-values
        - y_noise_data: Noisy y-values
    """
    if input_dims == 1:
        # 1D case: generate evenly spaced points
        X_data = np.linspace(x_min, x_max, n_points)
        y_data = model_1d(X_data)
    else:
        # Multi-dimensional case: generate random points in the specified range
        X_data = np.random.uniform(x_min, x_max, (n_points, input_dims))
        
        # Apply appropriate model function based on dimension
        if input_dims == 2:
            y_data = np.array([model_2d(x) for x in X_data])
        else:
            # For higher dimensions, use a generalized version (sum of squares)
            y_data = np.array([np.sum(x**2) for x in X_data])
    
    # Add noise to y values
    noise = np.random.normal(0, noise_scale, n_points)
    y_noise_data = y_data + noise
    
    print(f"Created dataset with {n_points} points, {input_dims} input dimension(s)")
    print(f"X shape: {X_data.shape}, y shape: {y_noise_data.shape}")
    
    return X_data, y_data, y_noise_data

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
    if input_dims == 1:
        # Extract scalar value for 1D case
        x_val = x[0] if hasattr(x, '__getitem__') else x
        y_pred = model_func(x_val)
    else:
        # For multi-dimensional case, pass the whole array
        y_pred = model_func(x)
    
    return (y_pred - target_y)**2

# =============== Inverse Error Evaluation ===============

def eval_inv_error(X_data: np.ndarray, y_data: np.ndarray, model_func: Callable) -> Tuple[np.ndarray, float, List[float]]:
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
    """
    # Determine input dimensionality
    input_dims = 1 if X_data.ndim == 1 else X_data.shape[1]
    n_samples = len(y_data)
    
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
            x0 = np.array([X_data[i]])
        else:
            x0 = X_data[i].copy()
        
        target_y = float(y_data[i])
        
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
                    # Compute squared error between found x and original x
                    error = float(np.sqrt((x_found - X_data[i]) ** 2))
                else:
                    x_found = result.x
                    x_pred[i] = x_found
                    # Compute squared error between found x and original x (Euclidean distance)
                    error = float(np.sqrt(np.sum((x_found - X_data[i]) ** 2)))
                
                total_inv_error += error
                point_errors.append(error)
            else:
                # Penalize failed inversions
                print(f"Optimization failed for point {i}: {result.message}")
                if input_dims == 1:
                    x_pred[i] = X_data[i]
                else:
                    x_pred[i] = X_data[i].copy()
                error = 10.0
                total_inv_error += error
                point_errors.append(error)
        except Exception as e:
            # Print error but continue with other samples
            print(f"Optimization error for point {i}: {str(e)}")
            if input_dims == 1:
                x_pred[i] = X_data[i]
            else:
                x_pred[i] = X_data[i].copy()
            error = 10.0
            total_inv_error += error
            point_errors.append(error)
    
    # Print optimization statistics
    avg_time = np.mean(optimization_times) if optimization_times else 0
    print(f"Average optimization time per point: {avg_time:.4f} seconds")
    print(f"Total inverse error: {total_inv_error:.4f}")
    avg_inv_error = total_inv_error / n_samples
    print(f"Average inverse error: {avg_inv_error:.4f}")
    
    return x_pred, avg_inv_error, point_errors

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
    x_curve = np.linspace(min(X_data) - 1, max(X_data) + 1, 1000)
    y_curve = model_func(x_curve)
    
    # Plot the model function as a blue curve
    plt.plot(x_curve, y_curve, 'b-', linewidth=2, label='Model function')
    
    # Plot original data points
    plt.scatter(X_data, y_data, color='g', s=50, alpha=0.7, label='Original data points')
    
    # Calculate y-values for predicted x points
    y_pred = model_func(x_pred)
    
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
    plt.title("Inverse Error Visualization (1D) using Nelder-Mead", fontsize=16)
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
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
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
    
    ax.set_title("Inverse Error Visualization (2D) using Nelder-Mead", fontsize=16)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# =============== Main Function ===============

if __name__ == "__main__":
    print("Testing inverse error calculation with different dimensionalities using Nelder-Mead")
    
    # Test with 1D data
    print("\n==== 1D Test ====")
    X_1d, _, y_1d_noise = create_dataset(n_points=50, input_dims=1, noise_scale=0.2)
    x_1d_pred, inv_error_1d, _ = eval_inv_error(X_1d, y_1d_noise, model_1d)
    visualization_plot(X_1d, y_1d_noise, x_1d_pred, model_1d)
    
    # Test with 2D data
    print("\n==== 2D Test ====")
    X_2d, _, y_2d_noise = create_dataset(n_points=50, input_dims=2, noise_scale=0.5)
    x_2d_pred, inv_error_2d, _ = eval_inv_error(X_2d, y_2d_noise, model_2d)
    visualization_plot(X_2d, y_2d_noise, x_2d_pred, model_2d)
    
    print("\nInverse error evaluation complete. Visualizations displayed.")
