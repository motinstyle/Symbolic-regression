# save results
SAVE_RESULTS = False

# requires gradient (also for enabling usage of inverse error)
REQUIRES_GRAD = True

# requires inverse error
REQUIRES_INV_ERROR = True  

# requires absolute error
REQUIRES_ABS_ERROR = False

# number of points for constant node check
NUM_OF_POINTS_FOR_CONST_NODE_CHECK = 1000

# constant node check magnitude
CONST_NODE_CHECK_MAGNITUDE = 30

# inv error evaluation frequency
INV_ERROR_EVAL_FREQ = 1 # 5

# inv error evaluation frequency
ABS_INV_ERROR_EVAL_FREQ = 5

# requires constants
CONST_OPT = False

# allow constants in the initial population
ALLOW_CONSTANTS = True

# constant mutation probability
CONST_MUTATION_PROB = 0.25

# initial population size
NUM_OF_START_MODELS = 1000 #1000

# depth of initial models (in create_diverse_population)
DEPTH_OF_START_MODELS = 3

# population size
POPULATION_SIZE = 100 #100

# number of models to select from the population
NUM_OF_MODELS_TO_SELECT = 10

# selection method (NSGA-II or top_k)
SELECTION_METHOD = "NSGA-II" # "top_k"

# number of epochs
NUM_OF_EPOCHS = 100

# probability of variables will be generated instead of constants
VARIABLES_OR_CONSTANTS_PROB = 0.9

# frequency of constant optimization
CONST_OPT_FREQUENCY = 10

# mutation probability
MUTATION_PROB = 0.25

# inverse error coefficient
INV_ERROR_COEF = 0.7

# max depth of a tree
MAX_DEPTH = 4

# fixed seed
FIXED_SEED = False # 3
if FIXED_SEED is not None:
    import numpy as np
    import torch
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)

