import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a folder for saving data
output_folder = "new_datasets"
os.makedirs(output_folder, exist_ok=True)

# Set of equations (approximately from Feynman equations)
def equation1(x, y):
    return x**2 + y**2  # Square of sum

def equation2(x, y):
    return x * y + np.sin(x)  # Sum of product and sine

def equation3(x, y, z):
    return x * y * z / (x + y + z)  # Equation with fraction

def equation4(x):
    return np.log(x + 1) + x**2  # Logarithmic function

def equation5(x, y):
    return np.exp(x) / (y + 1)  # Exponential division

def equation6(x, y, z):
    return np.cos(x) + np.sin(y) + np.tan(z)  # Trigonometric sum

def equation7(x):
    return np.sqrt(x) * np.log(x + 1)  # Square root and logarithm

def equation8(x, y):
    return x**3 - y**3  # Difference of cubes

def equation9(x, y, z):
    return x * y**2 - z * x**2  # Mixed quadratic function

def equation10(x, y):
    return np.arctan(x / (y + 1))  # Arctangent

# Functions and their parameters
functions = [
    ("sum_of_2pow", equation1, ["x", "y"]),
    ("mult_plus_sin", equation2, ["x", "y"]),
    ("mult_div_sum", equation3, ["x", "y", "z"]),
    ("log_plus_pow", equation4, ["x"]),
    ("exp_div_plus", equation5, ["x", "y"]),
    ("cos_sin_tan", equation6, ["x", "y", "z"]),
    ("sqrt_log", equation7, ["x"]),
    ("pow_minus_pow", equation8, ["x", "y"]),
    ("mult_pow_minus", equation9, ["x", "y", "z"]),
    ("atan_div_plus", equation10, ["x", "y"]),
]

# Data generation and visualization
for func_name, func, vars in functions:
    num_samples = 1000

    # Generate input data with consolidation
    inputs = {}
    means = [-500, -250, 0, 250, 500]
    std = 60
    num_means = len(means)
    
    # Calculate samples per grid point based on dimensionality
    samples_per_point = num_samples // (num_means ** len(vars))
    
    # Generate all combinations of means for the grid
    mean_combinations = np.array(np.meshgrid(*[means for _ in vars])).T.reshape(-1, len(vars))
    
    # Initialize arrays for each variable
    for var in vars:
        inputs[var] = np.zeros(num_samples)
    
    # Fill the arrays with samples around each grid point
    for i, point_means in enumerate(mean_combinations):
        start_idx = i * samples_per_point
        end_idx = (i + 1) * samples_per_point
        
        # Generate samples for each variable at this grid point
        for j, var in enumerate(vars):
            mean = point_means[j]
            # Generate normally distributed samples around the grid point
            samples = np.random.normal(mean, std, size=samples_per_point)
            # Add some noise to avoid perfect grid alignment
            samples = samples * (1 + 0.1 * np.random.randn(samples_per_point))
            inputs[var][start_idx:end_idx] = samples

    # Create DataFrame for input data
    data = pd.DataFrame(inputs)

    # Calculate output data
    try:
        data["output"] = func(**{var: data[var] for var in vars})
    except Exception as e:
        print(f"Error calculating output for {func_name}: {e}")
        continue

    # Add noise
    # noise = 0.05 * np.random.randn(num_samples)
    noise = 0.05 * (np.max(data["output"]) - np.min(data["output"])) * np.random.randn(num_samples)
    data["output"] += noise

    # Save to CSV file
    output_path = os.path.join(output_folder, f"{func_name}.csv")
    data.to_csv(output_path, index=False)
    print(f"Generated dataset for {func_name}: {output_path}")

    # Visualization
    if len(vars) == 1:  # For single variable functions
        plt.figure()
        plt.scatter(data[vars[0]], data["output"], s=5, alpha=0.7, label="Data points")
        plt.xlabel(vars[0])
        plt.ylabel("output")
        plt.title(f"Visualization of {func_name}")
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"{func_name}_plot.png"))
        plt.show()  # Show plot
        plt.close()
    elif len(vars) == 2:  # For two variable functions
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[vars[0]], data[vars[1]], data["output"], s=5, alpha=0.7, label="Data points")
        ax.set_xlabel(vars[0])
        ax.set_ylabel(vars[1])
        ax.set_zlabel("output")
        ax.set_title(f"Visualization of {func_name}")
        plt.legend()
        plt.savefig(os.path.join(output_folder, f"{func_name}_plot.png"))
        plt.show()  # Show plot
        plt.close()
    else:
        print(f"Visualization skipped for {func_name} with {len(vars)} variables.")

print("All datasets and visualizations generated.")
