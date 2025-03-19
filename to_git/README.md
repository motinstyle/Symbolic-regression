# Symbolic Regression Implementation

A Python implementation of symbolic regression that combines genetic programming and neural network approaches. The program is designed to find mathematical expressions that best fit given input-output data pairs, with support for both single and multiple variable functions.

## Program Structure

### PSEUDO CODE

```
SYMBOLIC REGRESSION ALGORITHM
============================

1. INITIALIZATION
----------------
1.1. Load configuration parameters from parameters.py
1.2. Create diverse initial population:
    - Ensure each function appears at least once as root
    - Create one tree with a constant node if CONST_OPT is False
    - Fill remaining population with unique random trees
    - Initialize variable counts (var_count) for all nodes
1.3. Initialize set of unique expressions

2. EVOLUTION LOOP (for num_epochs iterations)
------------------------------------------
2.1. Track unique expressions in current population
2.2. For each model in population:
    - Create copy for evaluation

2.3. If not first epoch:
    a) CROSSOVER PHASE
       For each model:
       - Try up to 10 times to create unique offspring
       - Select parent2 using tournament selection between two random models
       - Perform crossover with selected parent
       - Update variable counts and tree depth
       - Add to evaluation list if expression is unique
    
    b) EVALUATION PHASE (after crossover)
       Evaluate all models from crossover phase:
       - Calculate forward loss (RMSE)
       - Add domain penalty if applicable
       - Calculate inverse error if REQUIRES_GRAD is true
    
    c) SELECTION PHASE (intermediate)
       Select best models after crossover for mutation using specified method
       If SELECTION_METHOD is "NSGA-II":
       - Perform non-dominated sorting into fronts
       - Calculate crowding distance within fronts
       - Select models based on front rank and crowding distance
       If SELECTION_METHOD is "top_k":
       - Sort by total error
       - Select top k models
    
    d) MUTATION PHASE
       For each selected model:
       - Try up to 10 times to create unique mutant
       - Apply mutation with probability MUTATION_PROB
       - Apply constant mutation if CONST_OPT is False with probability CONST_MUTATION_PROB
       - Update variable counts and tree depth
       - Add to population if expression is unique

2.4. If CONST_OPT and epoch % CONST_OPT_FREQUENCY == 0:
     - Optimize constants in population

2.5. Evaluate all models in final population

2.6. Track and update best model:
     - Update if current model has lower error
     - Store error history
     - Check for early stopping if min_loss < 0.0001

3. MODEL EVALUATION
-----------------
3.1. Calculate error components:
    - Forward loss (RMSE between predictions and targets)
    - Domain penalty (for function domain violations)
    - Inverse error (if REQUIRES_GRAD is true)

3.2. Total error calculation:
    error = forward_loss + domain_penalty + inv_loss

4. TREE OPERATIONS
----------------
4.1. Build Random Tree:
    - Input: num_vars, max_depth, functions, requires_grad, allow_constants
    - Recursively build tree up to max_depth
    - Calculate var_count (number of variables) for each node
    - Return variable node or constant node at leaves
    - Ensure unique expressions in population

4.2. Mutation:
    - Randomly select node
    - Replace with compatible function (same parity)
    - Maintain tree structure
    - Update variable counts for affected nodes
    - Try multiple times for unique expression

4.3. Crossover:
    - Use tournament selection to choose second parent based on multiple criteria
    - Select random nodes from two parents
    - Exchange subtrees
    - Update variable counts and recalculate tree depth
    - Validate resulting tree
    - Ensure unique expression

4.4. Constant Optimization:
    - Periodically optimize constant values
    - Only if CONST_OPT is true
    - Use gradient-based optimization

4.5. Tree Depth Calculation:
    - Nodes with zero variables (var_count = 0) are treated as leaf nodes
    - Subtrees containing only constants do not contribute to tree depth
    - Depth calculation uses var_count to determine effective tree structure

5. TOURNAMENT SELECTION
---------------------
5.1. Compare two models based on multiple criteria:
    - Error value (RMSE)
    - Tree complexity (number of nodes)
    - Domain loss penalties
    - Forward loss and inverse loss
5.2. If one model dominates on all criteria, select it
5.3. Otherwise, select model with higher crowding distance (more unique)

6. OUTPUT
--------
6.1. Return best model found during evolution
6.2. Plot error history
6.3. Display final statistics:
    - Best error achieved
    - Number of unique expressions found
    - Final mathematical expression
```

The implementation consists of several Python modules:

### Main Module (`sr_test.py`)
The main module orchestrates the symbolic regression process. Key features include:
- Data loading and preprocessing
- Initial population generation with unique expression tracking
- Evolution process control with NSGA-II selection
- Model evaluation and visualization

Main functions:
- `create_diverse_population()`: Generates initial population with guaranteed function coverage and unique expressions
- `build_random_tree()`: Creates random expression trees with optional constant nodes and tracking variable counts
- `run_evolution()`: Executes the evolutionary algorithm with unique expression tracking

### Tree Implementation (`sr_tree.py`)
Implements the core tree structure for representing mathematical expressions:
- `Tree` class: Represents complete mathematical expressions
- `Node` class: Individual nodes in the expression tree with variable counting
- Evolution operators (mutation, crossover) with tournament selection
- Advanced tree depth calculation based on variable counts
- Forward and inverse evaluation methods
- Domain penalty calculation
- NSGA-II based model selection

### Function Definitions (`nodes.py`)
Contains definitions of available mathematical functions:
- Basic arithmetic operations (addition, multiplication, division)
- Trigonometric functions (sin, cos, tan)
- Exponential and logarithmic functions
- Function categorization and properties
- Domain and range constraints
- Constant nodes with optimization support (if CONST_OPT is True, else they can be mutated by adding random normal noise to the constant value)

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
Key parameters that can be adjusted in `parameters.py`:
- `POPULATION_SIZE`: Size of the population (default: 100)
- `NUM_OF_EPOCHS`: Number of evolution generations (default: 100)
- `MUTATION_PROB`: Probability of mutation (default: 0.25)
- `MAX_DEPTH`: Maximum depth of expression trees (default: 10)
- `REQUIRES_GRAD`: Whether to compute gradients (default: False)
- `CONST_OPT`: Whether to use constant optimization (default: False)
- `CONST_MUTATION_PROB`: Probability of mutating constants when CONST_OPT is False (default: 0.25)
- `SELECTION_METHOD`: Model selection method ("NSGA-II" or "top_k")
- `FIXED_SEED`: Random seed for reproducibility (default: 3)

## Implementation Details

### Genetic Operators

#### Mutation
The mutation operator randomly modifies nodes in the expression tree while preserving the tree structure:
- Each node has a mutation probability of 0.25 (configurable)
- For function nodes:
  - Unary functions can be replaced with other unary functions
  - Binary functions can be replaced with other binary functions
  - Function type (unary/binary) is preserved to maintain tree structure
- Constant nodes can be mutated with probability CONST_MUTATION_PROB
- Variable counts are updated after mutation
- Domain constraints are checked after mutation

#### Crossover
The crossover operator combines two parent trees to create offspring:
1. Uses tournament selection to choose the second parent
   - Compares two random models based on error, complexity, and other metrics
   - Selects the model that dominates on all criteria or has higher crowding distance
2. Randomly selects a node from parent1
3. Randomly selects a node from parent2
4. Creates a copy of parent1
5. Replaces the selected node in the copy with the subtree from parent2
6. Updates variable counts and recalculates tree depth
7. Validates the resulting tree:
   - Checks maximum depth constraint
   - Verifies variable indices
   - Ensures domain constraints
8. Tries up to 10 times to create a unique expression
9. Returns original parent1 copy if no valid crossover is found

### Variable Counting and Tree Depth Calculation
The implementation uses a novel approach to tree depth calculation:
1. Each node maintains a `var_count` property indicating how many variable nodes exist in its subtree
2. When building or modifying trees:
   - Variable nodes have `var_count = 1`
   - Constant nodes have `var_count = 0`
   - Unary function nodes inherit the `var_count` of their child
   - Binary function nodes sum the `var_count` of their children
3. In depth calculation:
   - Nodes with `var_count = 0` (subtrees with only constants) are treated as leaf nodes
   - This ensures that expressions like `x + (1 + 2)` have effective depth of 1
   - The actual tree structure may be deeper, but the functional complexity is measured by variable-dependent depth

### Modified Evolutionary Strategy
The implementation uses an improved evolution strategy:
1. Crossover phase:
   - Selects parents using tournament selection
   - Creates offspring with updated variable counts
2. Intermediate evaluation and selection:
   - Evaluates all crossover offspring
   - Selects best models using NSGA-II or top-k
3. Mutation phase:
   - Applies mutation only to the best selected models
   - Updates variable counts after mutation
4. This strategy focuses computational effort on promising models and maintains higher diversity.

### Inverse Function Approximation (if REQUIRES_GRAD is True, currently not used in experiments)
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
- Uses BFGS optimization method (can be changed to other optimization methods)
- Normalizes input data
- Limited to 10 iterations for efficiency
- Handles optimization failures gracefully
- Returns squared error between found and original inputs

### Model Selection

The implementation supports two selection methods:

#### NSGA-II Selection
Multi-objective optimization considering:
- Forward loss (RMSE)
- Inverse error (if enabled)
- Domain penalty
- Model complexity

Process:
1. Non-dominated sorting into Pareto fronts
2. Crowding distance calculation within fronts
3. Selection based on front rank and crowding distance

#### Top-k Selection
Simple selection based on total error score:
```python
error = mse_loss + complexity_penalty + inv_loss + domain_penalty
```

Where:
- `mse_loss`: Mean squared error on forward predictions
- `tree_size`: Number of nodes in the tree (complexity penalty)
- `inverse_error`: Error from inverse function approximation
- `domain_penalty`: Penalty for violating function domain constraints

#### Unique Expression Tracking
The evolution process maintains diversity by:
1. Tracking unique expressions using a set
2. Attempting multiple crossovers/mutations to generate new expressions
3. Only adding models with unique mathematical expressions to the population

### Error Calculation

The total error of a model combines multiple components:
```python
error = forward_loss + domain_penalty + inv_loss(if REQUIRES_GRAD is True)
```

Where:
- `forward_loss`: Mean squared error on predictions
- `domain_penalty`: Penalty for violating function domain constraints
- `inv_loss`: Error from inverse function approximation (if enabled)

## Performance Tips
- Use smaller population sizes for initial testing
- Enable REQUIRES_GRAD only when inverse error is needed
- Use NSGA-II selection for better diversity
- Adjust mutation and crossover probabilities based on problem complexity
- Use appropriate max_depth for your problem
- Consider enabling constant optimization for problems requiring precise constants