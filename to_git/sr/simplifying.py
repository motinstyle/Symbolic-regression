import sympy as sp
from sympy import sin, cos, exp, log, sqrt, tan, atan
import re

def simplify_expression(expr_str):
    """
    Simplifies a mathematical expression string using SymPy.
    
    Args:
        expr_str (str): A string representation of a mathematical expression,
                        as obtained from the to_math_expr() method in sr_tree.py
    
    Returns:
        str: A simplified string representation of the mathematical expression
    
    Example:
        >>> simplify_expression("sin(x0) + sin(x0)")
        "2*sin(x0)"
        >>> simplify_expression("x0 * 0 + 5")
        "5"
    """
    # If the expression is empty or None, return as is
    if not expr_str:
        return expr_str
    
    try:
        # Replace function names to match SymPy syntax
        # First, handle special cases like power functions
        expr_str = re.sub(r'\(([^)]+)\)\^2', r'pow2(\1)', expr_str)
        expr_str = re.sub(r'\(([^)]+)\)\^3', r'pow3(\1)', expr_str)
        
        # Create a dictionary to map variable names to sympy symbols
        var_pattern = re.compile(r'x(\d+)')
        var_matches = var_pattern.findall(expr_str)
        
        symbols = {}
        for var_idx in var_matches:
            symbol_name = f'x{var_idx}'
            symbols[symbol_name] = sp.Symbol(symbol_name)
        
        # Define custom functions for those not directly available in SymPy
        x = sp.Symbol('x')
        
        # Define pow2 and pow3 functions
        pow2_func = sp.Lambda(x, x**2)
        pow3_func = sp.Lambda(x, x**3)
        
        # Define abs_ function
        abs_func = sp.Lambda(x, sp.Abs(x))
        
        # Create a dictionary of local functions to use during evaluation
        local_dict = {
            'sin': sin,
            'cos': cos,
            'exp': exp,
            'log': log,
            'sqrt': sqrt,
            'tan': tan,
            'atan': atan,
            'abs': sp.Abs,
            'pow2': pow2_func,
            'pow3': pow3_func,
            **symbols
        }
        
        # Parse the expression using sympy
        expr = sp.sympify(expr_str, locals=local_dict)
        
        # Simplify the expression
        simplified_expr = sp.simplify(expr)
        
        # Convert back to string
        result = str(simplified_expr)
        
        # Replace power notation back to caret notation if needed
        result = result.replace('**2', '^2').replace('**3', '^3')
        
        # Replace any remaining ** with ^ for consistency
        result = result.replace('**', '^')
        
        return result
        
    except Exception as e:
        # If simplification fails, return the original expression
        print(f"Simplification error: {e}")
        return expr_str

def batch_simplify_expressions(expr_list):
    """
    Simplifies a list of mathematical expression strings using SymPy.
    
    Args:
        expr_list (list): A list of string representations of mathematical expressions
    
    Returns:
        list: A list of simplified string representations of the mathematical expressions
    """
    return [simplify_expression(expr) for expr in expr_list]

def map_functions_sympy(expr_str):
    """
    Maps function names from nodes.py format to SymPy format.
    
    Args:
        expr_str (str): A string representation of a mathematical expression
    
    Returns:
        str: A string with function names mapped to SymPy format
    """
    # Map of function names from nodes.py to SymPy
    function_map = {
        'pow2': lambda x: f"({x})^2",  # Custom power function
        'pow3': lambda x: f"({x})^3",  # Custom power function
        'ident': lambda x: x,          # Identity function
    }
    
    # Regular expression to find function calls
    func_pattern = re.compile(r'(\w+)\(([^)]+)\)')
    
    # Replace function calls according to the mapping
    def replace_func(match):
        func_name = match.group(1)
        arg = match.group(2)
        
        if func_name in function_map:
            return function_map[func_name](arg)
        return f"{func_name}({arg})"
    
    return func_pattern.sub(replace_func, expr_str)

def test_simplification():
    """
    Test function to demonstrate expression simplification with various examples.
    """
    test_expressions = [
        "x0 + x0",                   # Basic arithmetic
        "sin(x0) + sin(x0)",         # Trigonometric functions
        "x0 * 0 + 5",                # Zero multiplication
        "log(exp(x0))",              # Logarithm and exponential
        "(x0^2)^(1/2)",             # Powers
        "sin(x0)^2 + cos(x0)^2",     # Pythagorean identity
        "abs(x0) + abs(-x0)",        # Absolute value
        "x0 + 0",                    # Identity element
        "x0 * 1",                    # Identity element
        "pow2(x0) + 3*pow2(x0)",     # Custom power function
        "(x0 + x1) * (x0 - x1)",     # Difference of squares
        "sin(x0 + 0)"                # Function with identity element
    ]
    
    print("=== Testing Expression Simplification ===")
    for expr in test_expressions:
        simplified = simplify_expression(expr)
        print(f"Original: {expr}")
        print(f"Simplified: {simplified}")
        print("---")

if __name__ == "__main__":
    test_simplification()
