import torch
# import sr_tree as sr_tree
import nodes
from sr_tree import Tree, Node
import numpy as np

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

pi_half = torch.tensor([np.pi / 2])

ident_func = nodes.FUNCTIONS['ident_'].func
sin_func = nodes.FUNCTIONS['sin_'].func
cos_func = nodes.FUNCTIONS['cos_'].func
sum_func = nodes.FUNCTIONS['sum_'].func
var0 = Node("var", 1, ident_func, [0], requires_grad=True)
var1 = Node("var", 1, ident_func, [1], requires_grad=True)
sin_ = Node("sin", 1, sin_func, [var0], requires_grad=True)
cos_ = Node("cos", 1, cos_func, [var1], requires_grad=True)
sum_ = Node("sum", 2, sum_func, [sin_, cos_], requires_grad=True)

tree = Tree(sum_, requires_grad=True)


input_val = torch.tensor([[pi_half, pi_half]], requires_grad=True)
res = tree.start_node.eval_node(varval=input_val, cache=True)
print(res)

res.backward()
res.retain_grad()
print("res.grad:", res.grad)
print("input_val.grad:", input_val.grad)

tree.get_gradients()
print("tree.start_node._cached_grad:", tree.start_node._cached_grad)
print("var0._cached_grad:", var0._cached_grad)
print("var1._cached_grad:", var1._cached_grad)

# tensor1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# tensor2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# t_sum = torch.sum(tensor1 * tensor2)
# t_sum.retain_grad()

# t_sum.backward()
# print("t_sum:", t_sum)
# print("t_sum.grad:", t_sum.grad)


# print("tensor1.grad:", tensor1.grad)  # Градиент сохраняется
# print("tensor2.grad:", tensor2.grad)  # Градиент сохраняется



