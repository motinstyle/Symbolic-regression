
# save results
SAVE_RESULTS = False

# requires gradient (also for enabling usage of inverse error)
REQUIRES_GRAD = False

# inv error evaluation frequency
INV_ERROR_EVAL_FREQ = 5 

# requires constants
REQUIRES_CONST = False

# constant mutation probability
CONST_MUTATION_PROB = 0.25

# initial population size
NUM_OF_START_MODELS = 100

# depth of initial models (in create_diverse_population)
DEPTH_OF_START_MODELS = 5

# population size
POPULATION_SIZE = 100

# number of models to select from the population
NUM_OF_MODELS_TO_SELECT = 10

# selection method (NSGA-II or top_k)
SELECTION_METHOD = "NSGA-II" # "top_k"

# number of epochs
NUM_OF_EPOCHS = 100

# frequency of constant optimization
CONST_OPT_FREQUENCY = 10

# mutation probability
MUTATION_PROB = 0.25

# inverse error coefficient
INV_ERROR_COEF = 0.7

# max depth of a tree
MAX_DEPTH = 5

# fixed seed
FIXED_SEED = 3 # 3
if FIXED_SEED is not None:
    import numpy as np
    import torch
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)

