import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_results(X, y_true, y_pred, title="Model Comparison"):
    """Plot actual vs predicted results with error handling for invalid values."""
    # Проверка на бесконечные или NaN значения

    print("y_true.shape:", y_true.shape)
    print("y_pred.shape:", y_pred.shape)
    print("np.isfinite(y_true).shape:", np.isfinite(y_true).shape)
    print("np.isfinite(y_pred).shape:", np.isfinite(y_pred).shape)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    print("valid_mask.shape:", valid_mask.shape)
    if isinstance(X, np.ndarray) and len(X.shape) == 2 and X.shape[1] == 2:
        valid_mask &= np.all(np.isfinite(X), axis=1)  # Убедимся, что X тоже корректен

    # Отфильтруем только корректные данные
    print("X.shape:", X.shape)
    # print("valid_mask.shape:", valid_mask.shape)
    X = X[valid_mask]
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0 or len(y_pred) == 0:
        print("No valid data points available for plotting.")
        return

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
        plt.figure(figsize=(10, 6))
        
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
    
    plt.show()