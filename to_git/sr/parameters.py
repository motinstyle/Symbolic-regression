
# save results
SAVE_RESULTS = False

# requires gradient (also for enabling usage of inverse error)
REQUIRES_GRAD = False

# initial population size
NUM_OF_START_MODELS = 100
DEPTH_OF_START_MODELS = 3

# number of epochs
NUM_OF_EPOCHS = 200

# population size
POPULATION_SIZE = 10

# mutation probability
MUTATION_PROB = 0.15

# inverse error coefficient
INV_ERROR_COEF = 0.7

# max depth of a tree
MAX_DEPTH = 10

# fixed seed
FIXED_SEED = 2
if FIXED_SEED is not None:
    import numpy as np
    np.random.seed(FIXED_SEED)

