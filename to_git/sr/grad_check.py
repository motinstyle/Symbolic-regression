import torch
# import sr_tree as sr_tree
import nodes
from sr_tree import Tree, Node, safe_deepcopy
import numpy as np
from nodes import FUNCTIONS

# Tree class
# def __init__(self, start_node: Node, max_depth: int = 10, mutation_prob: float = 0.15,
#                  crossover_prob: float = 0.5, requires_grad: bool = False):
#         self.start_node = start_node
#         self.max_depth = max_depth
#         self.mutation_prob = mutation_prob
#         self.crossover_prob = crossover_prob
#         self.max_num_of_node = 0
#         self.error = 0
#         self.num_vars = None  # Will be set during evaluation
#         self.requires_grad = requires_grad
#         self._set_requires_grad(requires_grad)
#         self.enumerate_tree()

# Node class 
# def __init__(self, t_name: str, parity: int, func: callable, data: Union[List, torch.Tensor, int],
#                  requires_grad: bool = False):
#         self.t_name = t_name   # type
#         self.parity = parity   # parity of a func or an operator
#         self.func = func       # func or operator
#         self.data = data       # [child node/s] for Node, [value] for idents (consts), [var index] for variables
#         self.node_num = -1
#         self.num_vars = None   # Will be set when evaluating
#         self.requires_grad = requires_grad
#         self._cached_output = None  # Cache for forward pass
#         self._cached_grad = None    # Cache for backward pass

# pi_half = torch.tensor([np.pi / 2])

# ident_func = nodes.FUNCTIONS['ident_'].func
# sin_func = nodes.FUNCTIONS['sin_'].func
# cos_func = nodes.FUNCTIONS['cos_'].func
# sum_func = nodes.FUNCTIONS['sum_'].func
# var0 = Node("var", 1, ident_func, [0], requires_grad=True)
# var1 = Node("var", 1, ident_func, [1], requires_grad=True)
# sin_ = Node("sin", 1, sin_func, [var0], requires_grad=True)
# cos_ = Node("cos", 1, cos_func, [var1], requires_grad=True)
# sum_ = Node("sum", 2, sum_func, [sin_, cos_], requires_grad=True)

# tree = Tree(sum_, requires_grad=True)


# input_val = torch.tensor([[pi_half, pi_half]], requires_grad=True)
# res = tree.start_node.eval_node(varval=input_val, cache=True)
# print(res)

# res.backward()
# res.retain_grad()
# print("res.grad:", res.grad)
# print("input_val.grad:", input_val.grad)

# tree.get_gradients()
# print("tree.start_node._cached_grad:", tree.start_node._cached_grad)
# print("var0._cached_grad:", var0._cached_grad)
# print("var1._cached_grad:", var1._cached_grad)

# tensor1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# tensor2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# t_sum = torch.sum(tensor1 * tensor2)
# t_sum.retain_grad()

# t_sum.backward()
# print("t_sum:", t_sum)
# print("t_sum.grad:", t_sum.grad)


# print("tensor1.grad:", tensor1.grad)  # Градиент сохраняется
# print("tensor2.grad:", tensor2.grad)  # Градиент сохраняется

# Create test data similar to sr_test.py
X_data = np.linspace(1, 10, 100)  # Create 100 points between 1 and 10
Y_data = X_data**2  # Target function: x^2

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_data.reshape(-1, 1), dtype=torch.float32)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32)
data = torch.column_stack((X_tensor, Y_tensor))

# Create a tree representing x^2
# First, create variable node (x)
var_node = Node("var", 1, FUNCTIONS['ident_'].func, [0], requires_grad=True)

# Create multiplication node (x * x)
mult_node = Node("func", 2, FUNCTIONS['mult_'].func, [safe_deepcopy(var_node), safe_deepcopy(var_node)], requires_grad=True)

# Create the tree
tree = Tree(mult_node, requires_grad=True)
tree.num_vars = 1  # Set number of variables

# Evaluate tree error
tree.eval_tree_error(data)

print(f"Tree expression: {tree.to_math_expr()}")
print(f"Tree error: {tree.error}")
print(f"Forward loss: {tree.forward_loss}")

# Test forward pass
Y_pred = tree.forward(X_tensor)
print(f"\nFirst 5 predictions vs actual values:")
for i in range(5):
    print(f"X: {X_data[i]:.2f}, Predicted: {Y_pred[i].item():.2f}, Actual: {Y_data[i]:.2f}")



