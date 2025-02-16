import types
from typing import List
# from sr_tree import Tree

def non_dominated_sort(models):
    """Perform non-dominated sorting to classify models into Pareto fronts."""
    fronts = [[]]

    domination_counts = {model: 0 for model in models}  # How many models dominate this one
    dominated_models = {model: [] for model in models}  # Models that this model dominates

    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                continue

            # Compare models based on evaluation criteria
            if dominates(model_a, model_b):
                dominated_models[model_a].append(model_b)
            elif dominates(model_b, model_a):
                domination_counts[model_a] += 1

        # If model is non-dominated, place it in the first front
        if domination_counts[model_a] == 0:
            fronts[0].append(model_a)

    # Generate subsequent fronts
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for model in fronts[current_front]:
            for dominated_model in dominated_models[model]:
                domination_counts[dominated_model] -= 1
                if domination_counts[dominated_model] == 0:
                    next_front.append(dominated_model)
        if next_front:
            fronts.append(next_front)
        current_front += 1

    return fronts


def dominates(model_a, model_b) -> bool:
    """Check if model_a dominates model_b."""
    # Model A dominates Model B if it is better or equal in all criteria and strictly better in at least one
    criteria_a = [model_a.forward_loss, model_a.inv_loss, model_a.domain_loss]
    criteria_b = [model_b.forward_loss, model_b.inv_loss, model_b.domain_loss]

    better_or_equal = all(a <= b for a, b in zip(criteria_a, criteria_b))
    strictly_better = any(a < b for a, b in zip(criteria_a, criteria_b))

    return better_or_equal and strictly_better

def calculate_crowding_distance(front):
    """Calculate crowding distance for models in a Pareto front."""
    num_models = len(front)
    if num_models == 0:
        return

    # Initialize crowding distance to 0 for all models
    for model in front:
        model.crowding_distance = 0

    # For each criterion, sort models and calculate distances
    criteria = ["forward_loss", "inv_loss", "domain_loss"]
    for criterion in criteria:
        front.sort(key=lambda model: getattr(model, criterion))

        # Assign infinite distance to boundary points
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')

        # Calculate normalized distance for internal points
        min_val = getattr(front[0], criterion)
        max_val = getattr(front[-1], criterion)
        if max_val > min_val:  # Avoid division by zero
            for i in range(1, num_models - 1):
                prev_val = getattr(front[i - 1], criterion)
                next_val = getattr(front[i + 1], criterion)
                front[i].crowding_distance += (next_val - prev_val) / (max_val - min_val)
