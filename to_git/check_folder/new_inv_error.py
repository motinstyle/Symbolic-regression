import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import torch
from typing import Union, Callable, Tuple, List
import time

# Set random seed for reproducibility
np.random.seed(42)

# True model function (sine)
def model(x: np.ndarray) -> np.ndarray:
    """Target model function (sine)"""
    return np.exp(-1*x**2)

def dmodel(x: np.ndarray) -> np.ndarray:
    """Derivative of the target model function"""
    return -2*x*np.exp(-1*x**2)

# Create a dataset with noise
def create_dataset(n_points: int = 100, x_min: float = -5, x_max: float = 5, 
                  noise_scale: float = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a dataset with noise based on the model function.
    
    Args:
        n_points: Number of data points
        x_min: Minimum x value
        x_max: Maximum x value
        noise_scale: Scale of noise to add
        
    Returns:
        Tuple containing:
        - X_data: x-values array
        - y_data: Clean y-values
        - y_noise_data: Noisy y-values
    """
    X_data = np.linspace(x_min, x_max, n_points)
    y_data = model(X_data)
    y_noise = np.random.uniform(0, noise_scale, n_points)
    print(y_noise)
    y_noise_data = y_data + y_noise
    return X_data, y_data, y_noise_data

# Wrapper for optimization function
def wrapped_model_func(x: np.ndarray, target_y: float, model_func: Callable) -> float:
    """
    Function to minimize for finding x such that model(x) = target_y.
    
    Args:
        x: Input x value (will be a 1D array from scipy.optimize)
        target_y: Target y value
        model_func: Model function
        
    Returns:
        Squared difference between model(x) and target_y
    """
    # Extract the value since optimize.minimize passes x as an array
    x_val = x[0] if hasattr(x, '__getitem__') else x
    y_pred = model_func(x_val)
    return (y_pred - target_y)**2

# Evaluate inverse error using scipy.optimize.minimize
def eval_inv_error(X_data: np.ndarray, y_data: np.ndarray, model_func: Callable, 
                  method: str = 'BFGS') -> Tuple[np.ndarray, float]:
    """
    Evaluate inverse error using scipy.optimize.minimize.
    For each y value, tries to find x such that model_func(x) = y.
    
    Args:
        X_data: Original x values
        y_data: Original y values (potentially with noise)
        model_func: Model function to invert
        method: Optimization method for scipy.optimize.minimize
        
    Returns:
        Tuple containing:
        - x_pred: Predicted x values for each y
        - inv_error: Average inverse error
    """
    n_samples = len(y_data)
    x_pred = np.zeros(n_samples)
    total_inv_error = 0.0
    optimization_times = []
    
    print(f"Starting inverse error evaluation for {n_samples} points using {method} method...")
    
    for i in range(n_samples):
        # Use current X as initial guess
        x0 = np.array([X_data[i]])
        target_y = float(y_data[i])
        
        # Find x that minimizes (f(x) - y)^2
        try:
            start_time = time.time()
            result = optimize.minimize(
                wrapped_model_func,
                x0=x0,
                args=(target_y, model_func),
                method=method,
                options={'maxiter': 100, 'disp': False}
            )
            opt_time = time.time() - start_time
            optimization_times.append(opt_time)
            
            if result.success:
                # Get the optimized x value
                x_found = result.x[0]
                # Store the predicted x
                x_pred[i] = x_found
                # Compute squared error between found x and original x
                error = float((x_found - X_data[i]) ** 2)
                total_inv_error += error
            else:
                # Penalize failed inversions
                print(f"Optimization failed for point {i}: {result.message}")
                x_pred[i] = X_data[i]  # Use original x as fallback
                total_inv_error += 10.0
        except Exception as e:
            # Print error but continue with other samples
            print(f"Optimization error for point {i}: {str(e)}")
            x_pred[i] = X_data[i]  # Use original x as fallback
            total_inv_error += 10.0
    
    # Print optimization statistics
    avg_time = np.mean(optimization_times)
    print(f"Average optimization time per point: {avg_time:.4f} seconds")
    print(f"Total inverse error: {total_inv_error:.4f}")
    avg_inv_error = total_inv_error / n_samples
    print(f"Average inverse error: {avg_inv_error:.4f}")
    
    return x_pred, avg_inv_error

# Visualization function to show results
def visualization_plot(X_data: np.ndarray, y_data: np.ndarray, x_pred: np.ndarray, 
                      model_func: Callable, method_name: str = ""):
    """
    Visualize the results of inverse error calculation.
    
    Args:
        X_data: Original x values
        y_data: Original y values (potentially with noise)
        x_pred: Predicted x values from inverse error calculation
        model_func: Model function
        method_name: Name of the optimization method used (for title)
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
    # with horizontal lines (same y, different x)
    for i in range(len(X_data)):
        # Draw a horizontal line from (X_data[i], y_data[i]) to (x_pred[i], y_data[i])
        plt.plot([X_data[i], x_pred[i]], [y_data[i], y_data[i]], 'r-', alpha=0.5)
        
        # Draw a vertical line from (x_pred[i], y_data[i]) to (x_pred[i], y_pred[i])
        plt.plot([x_pred[i], x_pred[i]], [y_data[i], y_pred[i]], 'r--', alpha=0.3)
    
    # Add labels and legend
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    title = "Inverse Error Visualization"
    if method_name:
        title += f" using {method_name} method"
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def compare_optimization_methods(X_data: np.ndarray, y_data: np.ndarray, model_func: Callable):
    """
    Compare different optimization methods for inverse error calculation.
    
    Args:
        X_data: Original x values
        y_data: Original y values
        model_func: Model function
    """
    # List of optimization methods to compare
    methods = ['Nelder-Mead', 'BFGS', 'Powell', 'CG', 'L-BFGS-B']
    
    for method in methods:
        print(f"\n{'-'*50}")
        print(f"Testing optimization method: {method}")
        print(f"{'-'*50}")
        
        x_pred, inv_error = eval_inv_error(X_data, y_data, model_func, method=method)
        
        # Visualize results for this method
        visualization_plot(X_data, y_data, x_pred, model_func, method_name=method)
        
        print(f"Inverse error with {method}: {inv_error:.6f}")
        
        # Calculate residuals in y (how well model(x_pred) matches y_data)
        y_pred = model_func(x_pred)
        y_residuals = y_data - y_pred
        y_rmse = np.sqrt(np.mean(y_residuals**2))
        print(f"Y-RMSE with {method}: {y_rmse:.6f}")

# Main function
if __name__ == "__main__":
    # Generate dataset
    X_data, _, y_noise_data = create_dataset(n_points=50)
    
    # Create a combined dataset for reference
    data = np.column_stack((X_data, y_noise_data))
    
    print("Dataset created with shape:", data.shape)
    
    # Compare different optimization methods
    compare_optimization_methods(X_data, y_noise_data, model)
    
    # Default method: BFGS
    x_pred, inv_error = eval_inv_error(X_data, y_noise_data, model)
    
    # Visualize results
    visualization_plot(X_data, y_noise_data, x_pred, model, method_name="BFGS")
    
    print(f"Final average inverse error: {inv_error:.6f}")
    print("Optimization complete. Visualization displayed.") 