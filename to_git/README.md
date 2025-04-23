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
- Command-line argument parsing using argparse
- Data loading from specified directories
- Initial population generation with unique expression tracking
- Evolution process control with NSGA-II selection
- Model evaluation and visualization
- Support for multiple independent runs and best model selection

Main functions:
- `parse_arguments()`: Parses command-line arguments for customizing program execution
- `load_data()`: Loads data from CSV files in a specified directory
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
Or use `requirements.txt` file for enviromental setting 

## Usage Guide

### Command-Line Usage
The program now supports command-line arguments for easier execution:

```bash
# Run with a specific data directory
python sr_test.py --data_dir "../datasets"

# Run for a specific function in the data directory
python sr_test.py --data_dir "../datasets" --function "sin_x"

# Customize algorithm parameters
python sr_test.py --data_dir "../datasets" --epochs 50 --population 100 --max_depth 5

# Run multiple independent trials and select the best result
python sr_test.py --data_dir "../datasets" --runs 5

# Enable gradient computation for inverse error evaluation
python sr_test.py --data_dir "../datasets" --requires_grad
```

Available command-line arguments:
- `--data_dir`: Directory containing dataset CSV files
- `--function`: Specific function to test (optional)
- `--epochs`: Number of evolution epochs
- `--start_population_size`: Initial population size 
- `--population`: Population size
- `--mutation_prob`: Mutation probability
- `--max_depth`: Maximum tree depth
- `--requires_grad`: Enable gradient computation
- `--runs`: Number of independent runs to perform
- `--requires_forward_error`: Enable forward error computation
- `--requires_inv_error`: Enable inverse error computation
- `--requires_abs_inv_error`: Enable absolute inverse error computation
- `--requires_spatial_abs_inv_error`: Enable spatial absolute inverse error computation
- `--use_functions`: Comma-separated list of function names to use (e.g., "sum_,mult_,sin_,cos_"). If not specified, all functions will be used
- `--list_functions`: List all available functions and exit
- `--save_results`: Save results instead of displaying them

### Programmatic Usage
To run symbolic regression programmatically:

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
print(model.math_expr)
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
- `SAVE_RESULTS`: Save results instead of displaying
- `NUM_OF_RUNS`: How many times will be one expression evaluated 
- `REQUIRES_FORWARD_ERROR`: Use RMSE fitness function as a metric
- `REQUIRES_INV_ERROR`: Use RMSE of inverse prediction fitness function as a metric
- `REQUIRES_ABS_ERROR`: Use RMSE of orthogonal points predictions fitness function as a metric
- `REQUIRES_SPATIAL_ABS_ERROR`: Use RMSE of inputs from orthogonal points predictions fitness function as a metric
- `INV_ERROR_EVAL_FREQ`: Frequency of inverse loss usage  
- `ABS_INV_ERROR_EVAL_FREQ`: Frequency of orthogonal loss usage 
- `SPATIAL_INV_ERROR_EVAL_FREQ`: Frequency of orthogonal spatial loss usage
- `NUM_OF_POINTS_FOR_CONST_NODE_CHECK`: Sampling of the input space for constant node detection
- `CONST_NODE_CHECK_MAGNITUDE`: What deviation from mean should be still considered as a constant
- `ALLOW_CONSTANTS`: Enable usage of constants in expressions
- `VARIABLES_OR_CONSTANTS_PROB`: Probability of using variable as a leaf node instead of a constant
- `NUM_OF_START_MODELS`: Size of the initial population (default: 1000)
- `POPULATION_SIZE`: Size of the population (default: 100)
- `NUM_OF_EPOCHS`: Number of evolution generations (default: 100)
- `MUTATION_PROB`: Probability of mutation (default: 0.25)
- `MAX_DEPTH`: Maximum depth of expression trees (default: 10)
- `REQUIRES_GRAD`: Whether to compute gradients (default: False)
- `CONST_OPT`: Whether to use constant optimization (default: False)
- `CONST_MUTATION_PROB`: Probability of mutating constants when CONST_OPT is False (default: 0.25)
- `SELECTION_METHOD`: Model selection method ("NSGA-II" or "top_k")
- `FIXED_SEED`: Random seed for reproducibility (default: 3)
Whn programm is launched without input arguments, these parameters will be used 

#### Error Evaluation Parameters
The program supports various error metrics that can be enabled in `parameters.py`:
- `REQUIRES_INV_ERROR`: Enable inverse error evaluation (default: False)
  - When enabled, evaluates how well the inverse of the model approximates the original inputs
  - Uses PyTorch-based optimization via `eval_inv_error_torch` function
  - Controlled by `INV_ERROR_EVAL_FREQ` to set evaluation frequency

- `REQUIRES_ABS_ERROR`: Enable absolute inverse error evaluation (default: True)
  - Evaluates RMSE between original and predicted inputs using optimization
  - Uses `eval_abs_inv_error_torch` which applies batch gradient descent
  - Controlled by `ABS_INV_ERROR_EVAL_FREQ` to set evaluation frequency

- `REQUIRES_SPATIAL_ABS_ERROR`: Enable spatial absolute error evaluation (default: False)
  - Evaluates only the spatial component of the absolute inverse error
  - Measures Euclidean distance between original and optimized inputs

- `NUM_OF_POINTS_FOR_CONST_NODE_CHECK`: Number of points to check for constant nodes (default: 1000)
  - Used to detect when a subtree evaluates to a constant across input points

- `CONST_NODE_CHECK_MAGNITUDE`: Range magnitude for constant node detection (default: 30)
  - Defines the range of inputs used to verify if nodes behave as constants

These error metrics provide different ways to assess model accuracy and can be combined to guide the evolutionary process.

### Error Calculation

The total error of a model combines multiple components based on the enabled error metrics:
```python
error = forward_loss
```

When additional error metrics are enabled:
```python
if REQUIRES_INV_ERROR:
    error += inv_loss

if REQUIRES_ABS_ERROR:
    error += abs_loss
    
if REQUIRES_SPATIAL_ABS_ERROR:
    error += spatial_loss
```

Where:
- `forward_loss`: Mean squared error on predictions
- `inv_loss`: Error from inverse function approximation using PyTorch optimization
- `abs_loss`: Absolute inverse error (optimization-based with batch gradient descent)
- `spatial_loss`: Spatial component of the absolute error (Euclidean distance in input space)
- `domain_penalty`: Penalty for violating function domain constraints (not available in this version)

The program adapts the error calculation based on which metrics are enabled, focusing computational resources on the relevant error components.

### Inverse Function Approximation and Absolute Inverse Error

The program uses PyTorch-based optimization for inverse function approximation and absolute inverse error evaluation.

Key features of Inverse Function Approximation:
- Uses PyTorch's Adam optimizer instead of scipy's methods
- Performs batch gradient descent for efficiency
- Applies normalization to improve optimization
- Includes gradient clipping to prevent numerical instability
- Filters points that are outside the model's output range
- Calculates error as the Euclidean distance in the input space

Key features of Absolute Inverse Error:
- Combines both spatial and output errors in a unified metric
- Uses PyTorch's automatic differentiation
- Performs batch optimization for efficiency
- Calculates RMSE incorporating both input and output space differences
- Handles numerical instabilities by checking for NaN/Inf values

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

## Performance Tips
- Use smaller population sizes for initial testing
- Enable REQUIRES_GRAD only when inverse error is needed
- Use NSGA-II selection for better diversity
- Adjust mutation and crossover probabilities based on problem complexity
- Use appropriate max_depth for your problem
- Consider enabling constant optimization for problems requiring precise constants
- For large datasets or multiple functions, use the command-line interface with the `--data_dir` argument
- When exploring different parameter settings, try multiple runs with `--runs N` to find the best model