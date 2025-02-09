# Symbolic Regression Implementation

A Python implementation of symbolic regression that combines genetic programming and neural network approaches. The program is designed to find mathematical expressions that best fit given input-output data pairs, with support for both single and multiple variable functions.

## Program Structure

The implementation consists of several Python modules:

### Main Module (`sr_test.py`)
The main module orchestrates the symbolic regression process. Key features include:
- Data loading and preprocessing
- Initial population generation
- Evolution process control
- Model evaluation and visualization

Main functions:
- `create_diverse_population()`: Generates initial population with guaranteed function coverage
- `build_random_tree()`: Creates random expression trees
- `run_evolution()`: Executes the evolutionary algorithm

### Tree Implementation (`sr_tree.py`)
Implements the core tree structure for representing mathematical expressions:
- `Tree` class: Represents complete mathematical expressions
- `Node` class: Individual nodes in the expression tree
- Evolution operators (mutation, crossover)
- Forward and inverse evaluation methods
- Domain penalty calculation

### Function Definitions (`nodes.py`)
Contains definitions of available mathematical functions:
- Basic arithmetic operations (addition, multiplication, division)
- Trigonometric functions (sin, cos, tan)
- Exponential and logarithmic functions
- Function categorization and properties
- Domain and range constraints
- constant node (defined as an identity of a some predefined constant, cant be changed)

### Visualization (`visual.py`)
Provides visualization capabilities:
- 2D plotting for single-variable functions
- 3D plotting for two-variable functions
- Error history visualization
- Comparison plots of true vs predicted values

## Installation

Required dependencies:
```bash
pip install numpy torch pandas matplotlib scipy
```

## Usage Guide

### Basic Usage
To run symbolic regression on your data:

```python
from sr_test import run_evolution
import numpy as np

# Single variable example
X = np.linspace(-10, 10, 100)
Y = np.power(X, 2)  # Target function: x^2

model = run_evolution(
    X_data=X,
    Y_data=Y,
    num_epochs=100,
    population_size=70,
    mutation_prob=0.15,
    max_depth=10,
    requires_grad=False
)

# Print the resulting expression
print(model.to_math_expr())
```

### Data Format
The program accepts data in the following formats:
- NumPy arrays
- PyTorch tensors
- CSV files (through the `load_data()` function)

For CSV files, the format should be:
- Input variables in columns 1 to n-1
- Target variable in the last column

### Configuration Parameters
Key parameters that can be adjusted:
- `population_size`: Size of the population (default: 70)
- `num_epochs`: Number of evolution generations (default: 100)
- `mutation_prob`: Probability of mutation (default: 0.15)
- `max_depth`: Maximum depth of expression trees (default: 10)
- `requires_grad`: Whether to compute gradients (default: False)

## Advanced Features

### Custom Functions
New functions can be added by modifying `nodes.py`:
```python
FUNCTIONS['new_func_'] = FunctionInfo(
    name='new_func_',
    func=new_func_implementation,
    parity=1,  # 1 for unary, 2 for binary
    category=FunctionCategory.ARITHMETIC,
    display_name='new_func',
    domain_min=-np.inf,
    domain_max=np.inf
)
```

### Domain Constraints
The program supports domain constraints for functions:
- Automatic domain penalty calculation
- Range validation for mathematical operations
- Adaptive penalty weighting in the fitness function
Domain constraints are defined in the `FUNCTIONS` dictionary in `nodes.py`. 

### Visualization Options
Different visualization modes available:
- Single variable plots (2D)
- Two variable surface plots (3D)
- Error evolution plots
- Custom plot configurations

## Example Applications

### Single Variable Regression
```python
# Finding a polynomial function
X = np.linspace(-5, 5, 100)
Y = X**3 - 2*X + 1
model = run_evolution(X, Y)
```

### Multiple Variable Regression
```python
# Finding a 2D function
X = np.random.uniform(-5, 5, (100, 2))
Y = X[:, 0]**2 + X[:, 1]**2
model = run_evolution(X, Y)
```

## Implementation Details

### Genetic Operators

#### Mutation
The mutation operator randomly modifies nodes in the expression tree while preserving the tree structure:
- Each node has a mutation probability of 0.15 (configurable)
- For function nodes:
  - Unary functions can be replaced with other unary functions
  - Binary functions can be replaced with other binary functions
  - Function type (unary/binary) is preserved to maintain tree structure
- Variable and constant nodes are not mutated
- Domain constraints are checked after mutation

#### Crossover
The crossover operator combines two parent trees to create offspring:
1. Randomly selects a node from parent1
2. Randomly selects a node from parent2
3. Creates a copy of parent1
4. Replaces the selected node in the copy with the subtree from parent2
5. Validates the resulting tree:
   - Checks maximum depth constraint
   - Verifies variable indices
   - Ensures domain constraints
6. If validation fails, tries different nodes (up to 10 attempts)
7. Returns original parent1 copy if no valid crossover is found

### Inverse Function Approximation

The program uses optimization-based inverse function approximation:
```python
def eval_inv_error(self, data):
    """
    Evaluates inverse error using scipy.optimize.minimize.
    For each point in the dataset, tries to find x such that f(x) = y.
    """
    # Normalize data for better optimization
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    
    # For each sample point
    for i in range(n_samples):
        # Use current X as initial guess
        result = opt.minimize(
            self.wrapped_tree_func,
            x0=X[i],
            args=(y[i],),
            method='BFGS',
            options={'maxiter': 10}
        )
```

Key features:
- Uses BFGS optimization method
- Normalizes input data
- Limited to 10 iterations for efficiency
- Handles optimization failures gracefully
- Returns squared error between found and original inputs

### Model Selection Criteria

The final model is selected based on a composite fitness function:
```python
fitness = mse_loss + 10*tree_size + inverse_error + domain_penalty
```

Where:
- `mse_loss`: Mean squared error on forward predictions
- `tree_size`: Number of nodes in the tree (complexity penalty)
- `inverse_error`: Error from inverse function approximation
- `domain_penalty`: Penalty for violating function domain constraints

Selection process:
1. Population is sorted by fitness (lower is better)
2. Top 10 models are preserved for next generation
3. These models are used to generate new population through crossover
4. Process continues for specified number of epochs
5. Best model from final population is returned 

## Performance Tips
- Use smaller population sizes for initial testing
- Increase population size and epochs for better results
- Enable gradients only when necessary
- Normalize input data for better convergence
- Use appropriate max_depth for your problem complexity

## Technical Details

### Tree Evaluation Process

#### Value Propagation
The computation tree processes input values in a bottom-up manner:
1. Input values are converted to PyTorch tensors
2. Variable nodes receive their corresponding values from the input tensor
3. Constant nodes maintain their predefined values
4. Function nodes evaluate their children first, then apply their operation
5. Results are cached at each node for efficiency
6. The root node's output represents the final computation

Example of value propagation:
```python
def eval_node(self, varval=None, cache=True):
    """Evaluate node with PyTorch tensors and optional gradient computation"""
    if varval is not None:
        # Convert input to tensor if needed
        if isinstance(varval, np.ndarray):
            varval = to_tensor(varval, requires_grad=self.requires_grad)
        if varval.dim() == 1:
            varval = varval.reshape(-1, 1)
        self.num_vars = varval.shape[1]
    
    # Handle different node types
    if self.t_name == "ident":
        # Constant node
        result = to_tensor(self.data[0], requires_grad=False)
        result = result.expand(varval.shape[0])
    elif self.t_name == "var":
        # Variable node
        var_idx = self.data[0]
        result = varval[:, var_idx].clone()
        if self.requires_grad:
            result.requires_grad_(True)
    else:
        # Function node
        if self.parity == 1:
            child_result = self.data[0].eval_node(varval=varval, cache=cache)
            result = self.func(child_result)
        else:
            in1 = self.data[0].eval_node(varval=varval, cache=cache)
            in2 = self.data[1].eval_node(varval=varval, cache=cache)
            result = self.func(in1, in2)
```

#### Domain Constraints Handling
Domain constraints are enforced at each node during evaluation:

1. Each function has defined domain constraints:
```python
FUNCTIONS['log_'] = FunctionInfo(
    name='log_',
    func=log_,
    parity=1,
    category=FunctionCategory.LOGARITHMIC,
    domain_min=0,  # log(x) defined only for x > 0
    range_min=-MAX_VALUE,
    range_max=MAX_VALUE
)
```

2. Domain violations are tracked during evaluation:
```python
# Check domain constraints for unary functions
if isinstance(child_result, torch.Tensor) and self.func_info is not None:
    mask_min = child_result < self.func_info.domain_min
    mask_max = child_result > self.func_info.domain_max
    if mask_min.any() or mask_max.any():
        self.domain_penalty += torch.sum(torch.abs(
            torch.where(mask_min, 
                       self.func_info.domain_min - child_result, 
                       0)
        )) + torch.sum(torch.abs(
            torch.where(mask_max, 
                       child_result - self.func_info.domain_max, 
                       0)
        ))
```

3. Domain penalties are accumulated:
- Each violation adds to the node's domain_penalty
- Penalties are propagated up the tree
- Total penalty affects the model's fitness score

4. Handling invalid inputs:
- Small epsilon values are added to prevent division by zero
- Values are clamped to valid ranges
- NaN results are penalized heavily in the fitness function

### Gradient Computation (requires_grad=True)

When gradient computation is enabled:

1. Input Handling:
```python
# Variable nodes create gradient-enabled tensors
if self.t_name == "var" and self.requires_grad:
    result = varval[:, var_idx].clone()
    result.requires_grad_(True)
```

2. Forward Pass:
- All operations use PyTorch's autograd system
- Intermediate results maintain gradient information
- Results are cached for backward pass

3. Gradient Calculation:
```python
def forward_with_gradients(self, varval):
    """
    Performs forward pass and computes gradients for each input vector.
    Returns:
        outputs: Tensor (N,) containing scalar outputs
        gradients: Tensor (N, num_vars) containing gradients
    """
    outputs = self.forward(varval)
    gradients = torch.autograd.functional.vjp(
        self.forward,
        varval,
        v=torch.ones_like(outputs),
        create_graph=self.requires_grad
    )[1]
    return outputs, gradients
```

4. Gradient Usage:
- Gradients are used in inverse function approximation
- Help guide the optimization process
- Enable sensitivity analysis of the model
- Can be used for additional regularization

### Error Calculation

The total error of a model combines multiple components:

```python
def eval_tree_error(self, data):
    # Forward pass error (MSE)
    y_pred = self.forward(X)
    mse_loss = calculate_mse(y_pred, y)
    
    # Domain penalty
    domain_penalty = self.start_node.get_total_domain_penalty()
    domain_penalty /= len(y)  # Normalize by sample size
    
    # Inverse error (if enabled)
    if self.requires_grad:
        inv_loss = self.eval_inv_error(data)
    else:
        inv_loss = 0
    
    # Tree complexity penalty
    complexity_penalty = 10 * self.max_num_of_node
    
    # Total error
    self.error = mse_loss + complexity_penalty + inv_loss + domain_penalty
```

This comprehensive error calculation ensures that:
- The model fits the data well (MSE)
- Respects domain constraints
- Maintains reasonable complexity
- Has good inverse function behavior (if enabled)