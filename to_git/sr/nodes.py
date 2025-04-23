import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

MAX_VALUE = 1e9
EPSILON = 1e-10

class FunctionCategory(Enum):
    ARITHMETIC = "arithmetic"
    TRIGONOMETRIC = "trigonometric"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    IDENTITY = "identity"
    CONSTANT = "constant"

@dataclass
class FunctionInfo:
    name: str
    func: Callable
    parity: int  # 1 for unary, 2 for binary
    category: FunctionCategory
    display_name: str
    is_commutative: bool = False
    is_associative: bool = False
    domain_min: float = -MAX_VALUE
    domain_max: float = MAX_VALUE
    range_min: float = -MAX_VALUE
    range_max: float = MAX_VALUE
    description: str = ""
    requires_grad: bool = True  # Whether this function supports gradient computation

def const_(const):
    """Convert input to tensor if needed"""
    if not torch.is_tensor(const):
        const = torch.tensor(const, dtype=torch.float32)
    return torch.clamp(const, -MAX_VALUE, MAX_VALUE)

def ident_(x):
    """Identity function that ensures tensor output"""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.clamp(x, -MAX_VALUE, MAX_VALUE)

def neg_(x):
    """Vectorized negation with value capping"""
    return torch.clamp(-x, -MAX_VALUE, MAX_VALUE)

def sum_(x, y):
    """Vectorized addition with value capping"""
    result = x + y
    return torch.clamp(result, -MAX_VALUE, MAX_VALUE)

def mult_(x, y):
    """Vectorized multiplication with value capping"""
    result = x * y
    return torch.clamp(result, -MAX_VALUE, MAX_VALUE)

def div_(x, y):
    """Vectorized division with handling of division by zero"""
    # Add small epsilon to denominator to avoid division by zero
    safe_y = y + torch.sign(y) * EPSILON
    safe_y = torch.where(y == 0, torch.ones_like(y) * EPSILON, safe_y)
    result = x / safe_y
    return torch.clamp(result, -MAX_VALUE, MAX_VALUE)

def exp_(x):
    """Vectorized exponential with value capping"""
    # Clamp input to avoid overflow
    safe_x = torch.clamp(x, -100, 100)  # exp(100) is already huge
    result = torch.exp(safe_x)
    return torch.clamp(result, -MAX_VALUE, MAX_VALUE)

def sin_(x):
    """Vectorized sine"""
    return torch.sin(x)

def cos_(x):
    """Vectorized cosine"""
    return torch.cos(x)

# def pow_(x, power):
#     """Vectorized power with handling of negative powers"""
#     # Handle negative/zero base for fractional powers
#     power = 2
#     safe_x = torch.where(x < 0, torch.zeros_like(x), x)
    
#     # For negative powers, compute positive power first then reciprocal
#     # abs_power = torch.abs(power)
#     # result = torch.pow(safe_x + EPSILON, abs_power)
#     # result = torch.where(power < 0, 1.0 / (result + EPSILON), result)
    
#     return torch.clamp(safe_x**power, -MAX_VALUE, MAX_VALUE)

def pow2_(x):
    """Vectorized power with handling of negative powers"""
    return torch.clamp(x**2, -MAX_VALUE, MAX_VALUE)

def pow3_(x):
    """Vectorized power with handling of negative powers"""
    return torch.clamp(x**3, -MAX_VALUE, MAX_VALUE)

def abs_(x):
    """Vectorized absolute value"""
    return torch.clamp(torch.abs(x), -MAX_VALUE, MAX_VALUE)

def log_(x):
    """Vectorized natural logarithm with handling of negative/zero inputs"""
    safe_x = x + EPSILON
    safe_x = torch.where(x <= 0, torch.ones_like(x) * EPSILON, safe_x)
    result = torch.log(safe_x)
    return torch.clamp(result, -MAX_VALUE, MAX_VALUE)

def sqrt_(x):
    """Vectorized square root with handling of negative inputs"""
    safe_x = torch.where(x < 0, torch.zeros_like(x), x)
    return torch.sqrt(safe_x + EPSILON)

def tan_(x):
    """Vectorized tangent with value capping"""
    return torch.clamp(torch.tan(x), -MAX_VALUE, MAX_VALUE)

def atan_(x):
    """Vectorized arctangent"""
    return torch.atan(x)

# Define all available functions with their properties
FUNCTIONS = {
    'const_': FunctionInfo(
        name='const_',
        func=const_,
        parity=1,
        category=FunctionCategory.CONSTANT,
        display_name='c',
        description="Constant function",
        requires_grad=False  # Constants don't need gradients
    ),
    'ident_': FunctionInfo(
        name='ident_',
        func=ident_,
        parity=1,
        category=FunctionCategory.IDENTITY,
        display_name='x',
        description="Identity function"
    ),
    'neg_': FunctionInfo(
        name='neg_',
        func=neg_,
        parity=1,
        category=FunctionCategory.ARITHMETIC,
        display_name='-',
        description="Negation of a number"
    ),
    'sum_': FunctionInfo(
        name='sum_',
        func=sum_,
        parity=2,
        category=FunctionCategory.ARITHMETIC,
        display_name='+',
        is_commutative=True,
        is_associative=True,
        description="Addition of two numbers"
    ),
    'mult_': FunctionInfo(
        name='mult_',
        func=mult_,
        parity=2,
        category=FunctionCategory.ARITHMETIC,
        display_name='*',
        is_commutative=True,
        is_associative=True,
        description="Multiplication of two numbers"
    ),
    'div_': FunctionInfo(
        name='div_',
        func=div_,
        parity=2,
        category=FunctionCategory.ARITHMETIC,
        display_name='/',
        domain_min=-MAX_VALUE,
        description="Division of two numbers"
    ),
    'exp_': FunctionInfo(
        name='exp_',
        func=exp_,
        parity=1,
        category=FunctionCategory.EXPONENTIAL,
        display_name='exp',
        domain_min=-100,
        range_min=0,
        description="Exponential function"
    ),
    'sin_': FunctionInfo(
        name='sin_',
        func=sin_,
        parity=1,
        category=FunctionCategory.TRIGONOMETRIC,
        display_name='sin',
        range_min=-1,
        range_max=1,
        description="Sine function"
    ),
    'cos_': FunctionInfo(
        name='cos_',
        func=cos_,
        parity=1,
        category=FunctionCategory.TRIGONOMETRIC,
        display_name='cos',
        range_min=-1,
        range_max=1,
        description="Cosine function"
    ),
    'pow2_': FunctionInfo(
        name='pow2_',
        func=pow2_,
        parity=1,
        category=FunctionCategory.EXPONENTIAL,
        display_name='^2',
        description="Power function"
    ),
    'pow3_': FunctionInfo(
        name='pow3_',
        func=pow3_,
        parity=1,
        category=FunctionCategory.EXPONENTIAL,
        display_name='^3',
        description="Power function"
    ),
    'abs_': FunctionInfo(
        name='abs_',
        func=abs_,
        parity=1,
        category=FunctionCategory.ARITHMETIC,
        display_name='abs',
        range_min=0,
        description="Absolute value function"
    ),
    'log_': FunctionInfo(
        name='log_',
        func=log_,
        parity=1,
        category=FunctionCategory.LOGARITHMIC,
        display_name='log',
        domain_min=0,
        description="Natural logarithm"
    ),
    'sqrt_': FunctionInfo(
        name='sqrt_',
        func=sqrt_,
        parity=1,
        category=FunctionCategory.ARITHMETIC,
        display_name='√',
        domain_min=0,
        description="Square root function"
    ),
    'tan_': FunctionInfo(
        name='tan_',
        func=tan_,
        parity=1,
        category=FunctionCategory.TRIGONOMETRIC,
        display_name='tan',
        description="Tangent function"
    ),
    'atan_': FunctionInfo(
        name='atan_',
        func=atan_,
        parity=1,
        category=FunctionCategory.TRIGONOMETRIC,
        display_name='arctan',
        range_min=-np.pi/2,
        range_max=np.pi/2,
        description="Arctangent function"
    )
}

# List of function names for backward compatibility
func_names = list(FUNCTIONS.keys())

def get_function_info(func_name: str) -> Optional[FunctionInfo]:
    """Get detailed information about a function by its name"""
    return FUNCTIONS.get(func_name)

def get_functions_by_category(category: FunctionCategory) -> List[FunctionInfo]:
    """Get all functions of a specific category"""
    return [f for f in FUNCTIONS.values() if f.category == category]

def get_functions_by_parity(parity: int) -> List[FunctionInfo]:
    """Get all functions with a specific parity (1 for unary, 2 for binary)"""
    return [f for f in FUNCTIONS.values() if f.parity == parity]

def to_tensor(x: Union[np.ndarray, torch.Tensor, float, int], requires_grad: bool = False) -> torch.Tensor:
    """Convert input to PyTorch tensor with proper settings"""
    if torch.is_tensor(x):
        x = x.detach().clone()
        x.requires_grad = requires_grad
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    else:
        x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = requires_grad
    return x