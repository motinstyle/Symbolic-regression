import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_results(X, y_true, y_pred, title="Model Comparison", return_plot: bool = False):
    """Plot actual vs predicted results with error handling for invalid values.
    
    Args:
        X: Input data
        y_true: True values
        y_pred: Predicted values
        title: Title for the plot
        save_plot: Whether to save the plot immediately (deprecated, use return_fig=True instead)
    
    Returns:
        fig: Matplotlib figure object if return_fig=True, otherwise None
    """
    # Check for infinite or NaN values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if isinstance(X, np.ndarray) and len(X.shape) == 2 and X.shape[1] == 2:
        valid_mask &= np.all(np.isfinite(X), axis=1)  # Make sure X is also valid

    # Filter only valid data
    X = X[valid_mask]
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0 or len(y_pred) == 0:
        print("No valid data points available for plotting.")
        return None

    if isinstance(X, np.ndarray) and len(X.shape) == 2 and X.shape[1] == 2:
        # 3D plots for two input variables
        fig = plt.figure(figsize=(20, 6))
        
        # First subplot: Combined true and predicted values
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1_true = ax1.scatter(X[:, 0], X[:, 1], y_true, c='blue', label='True', alpha=0.6)
        scatter1_pred = ax1.scatter(X[:, 0], X[:, 1], y_pred, c='red', label='Predicted', alpha=0.6)
        ax1.set_title(f"{title}\nTrue vs Predicted")
        ax1.set_xlabel("X₁")
        ax1.set_ylabel("X₂")
        ax1.set_zlabel("Y")
        ax1.legend()
        
        # Second subplot: True values
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], y_true, c=y_true, cmap='viridis', alpha=0.6)
        ax2.set_title(f"{title}\nTrue Values")
        ax2.set_xlabel("X₁")
        ax2.set_ylabel("X₂")
        ax2.set_zlabel("Y")
        plt.colorbar(scatter2, ax=ax2)
        
        # Third subplot: Predicted values
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(X[:, 0], X[:, 1], y_pred, c=y_pred, cmap='viridis', alpha=0.6)
        ax3.set_title(f"{title}\nPredicted Values")
        ax3.set_xlabel("X₁")
        ax3.set_ylabel("X₂")
        ax3.set_zlabel("Y")
        plt.colorbar(scatter3, ax=ax3)
        
        # Set same scale for all plots
        z_min = min(y_true.min(), y_pred.min())
        z_max = max(y_true.max(), y_pred.max())
        ax1.set_zlim(z_min, z_max)
        ax2.set_zlim(z_min, z_max)
        ax3.set_zlim(z_min, z_max)
        
        # Adjust layout
        plt.tight_layout()
        
    else:
        # 2D plot for single input variable
        fig = plt.figure(figsize=(10, 6))
        
        # Sort points by X value for better line plotting
        sort_idx = np.argsort(X)
        X = X[sort_idx]
        y_true = y_true[sort_idx]
        y_pred = y_pred[sort_idx]
        
        plt.plot(X, y_true, label="Actual", color='blue', alpha=0.7)
        plt.plot(X, y_pred, label="Predicted", color='red', linestyle='--', alpha=0.7)
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    if return_plot:
        return fig
    else:
        plt.show()


def plot_rmse_history(mean_rmse: np.ndarray, median_rmse: np.ndarray, best_rmse: np.ndarray, title: str = "Training Error History", return_plot: bool = False):
    """
    Plot RMSE statistics history in three subplots.
    
    Args:
        mean_rmse: Array of mean RMSE values over epochs
        median_rmse: Array of median RMSE values over epochs
        best_rmse: Array of best RMSE values over epochs
        title: Title for the overall figure
        
    Returns:
        fig: Matplotlib figure object
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Mean RMSE subplot
    plt.subplot(131)
    plt.plot(mean_rmse, 'b-', label='Mean RMSE')
    plt.title('Mean RMSE History')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    
    # Median RMSE subplot
    plt.subplot(132)
    plt.plot(median_rmse, 'g-', label='Median RMSE')
    plt.title('Median RMSE History')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    
    # Best RMSE subplot
    plt.subplot(133)
    plt.plot(best_rmse, 'r-', label='Best RMSE')
    plt.title('Best RMSE History')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    if return_plot:
        return fig
    else:
        plt.show()


def plot_2d_scatter(X, y, title: str = "Scatter Plot"):
    """
    Plot 2D scatter plot of input data (X) and target values (y).
    
    Args:   
        X: Input data (numpy array)
        y: Target values (numpy array)
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title(title)
    plt.show()

def plot_3d_scatter(X, y, title: str = "Scatter Plot"):
    """
    Plot 3D scatter plot of input data (X) and target values (y).
    
    Args:
        X: Input data (numpy array)
        y: Target values (numpy array)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  

    # Create scatter plot
    scatter = ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis')
    
    # Add labels and colorbar
    ax.set_xlabel('X1') 
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.colorbar(scatter)
    plt.axis('equal')
    plt.title(title)
    plt.show()

def plot_scatter(X, y, title: str = "Scatter Plot"):
    """
    Plot scatter plot of input data (X) and target values (y).
    
    Args:
        X: Input data (numpy array)
        y: Target values (numpy array)
        title: Title for the plot
    """
    if X.shape[-1] == 1:
        plot_2d_scatter(X, y, title)
    elif X.shape[-1] == 2:
        plot_3d_scatter(X, y, title)
    else:
        raise ValueError("Invalid input data dimensions")



