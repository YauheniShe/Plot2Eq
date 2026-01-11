import sympy as sp
import random

OPERATORS = {
    'Add': {'symbol': sp.Add, 'arity': 2, 'weight': 5.0},
    'Sub': {'symbol': lambda x, y: x - y, 'arity': 2, 'weight': 5.0},
    'Mul': {'symbol': sp.Mul, 'arity': 2, 'weight': 5.0},
    'Div': {'symbol': lambda x, y: x / y, 'arity': 2, 'weight': 2.0},
    'IntPow': {'symbol': sp.Pow, 'arity': 2, 'weight': 1.0},
    'sin': {'symbol': sp.sin, 'arity': 1, 'weight': 3.0},
    'cos': {'symbol': sp.cos, 'arity': 1, 'weight': 3.0},
    'exp': {'symbol': sp.exp, 'arity': 1, 'weight': 1.0},
    'log': {'symbol': sp.log, 'arity': 1, 'weight': 2.0},
    'sqrt': {'symbol': sp.sqrt, 'arity': 1, 'weight': 2.0},
    'sqr':  {'symbol': lambda x: x**2, 'arity': 1, 'weight': 3.0},
    'abs': {'symbol': sp.Abs, 'arity': 1, 'weight': 4.0},
    'Neg': {'symbol': lambda x: -x, 'arity': 1.0, 'weight': 2.0}
}

VARIABLES = ['x']
CONSTANT = ['pi', 'e', '1', 'C']


class Constant(sp.Symbol):
    pass

class ExpressionGenerator:
    def __init__(self, max_depth: int, const_prob: float, leaf_prob: float) -> None:
        self.max_depth  = max_depth
        self.const_prob = const_prob
        self.leaf_prob = leaf_prob

    def generate_expr(self, depth=None):
        if depth is None:
            depth = self.max_depth
        while True:
            expr = self.generate_recursive()
            if expr is None:
                continue
            expr = expr.simplify()
            if expr.has(sp.I) or expr.has(sp.nan) or expr.has(sp.zoo) or expr.is_constant():
                continue
            expr = self.round_floats(expr, 2)
            return expr

    def generate_recursive(self, depth=None):
        if depth is None:
            depth = self.max_depth
        is_root = (depth == self.max_depth)
        if not is_root and (random.random() < self.const_prob or depth == 0):
            return self.generate_leaf()
        weights = [OPERATORS[name]['weight'] for name in OPERATORS.keys()]
        op_name = random.choices(list(OPERATORS.keys()), weights=weights, k=1)[0]
        op_info = OPERATORS[op_name]
        arity = op_info['arity']
        symbol = op_info['symbol']
        if arity == 1:
            args = self.generate_recursive(depth - 1)
            return symbol(args)
        if arity == 2:
            if op_name == 'IntPow':
                first_arg = self.generate_recursive(depth - 1)
                second_arg = sp.Integer(random.randint(-5, 5))
                args = (first_arg, second_arg)
                return symbol(*args)
            first_arg = self.generate_recursive(depth - 1)
            second_arg = self.generate_recursive(depth - 1)
            args = (first_arg, second_arg)
            return symbol(*args)

    def round_floats(self, expr, precision=2):
        return expr.xreplace({
            n: round(n, precision) for n in expr.atoms(sp.Float)
        })

    def generate_leaf(self):
        if random.random() < self.leaf_prob:
            c_type = random.choice(CONSTANT)
            if c_type == 'C':
                val = round(random.uniform(-5, 5), 2)
                return sp.Float(val)
            if c_type == 'pi':
                return sp.pi
            if c_type == 'e':
                return sp.E
            else:
                return sp.Integer(1)
        else:
            var = random.choice(VARIABLES)
            return sp.Symbol(var, real=True)
        
generator = ExpressionGenerator(max_depth=4, const_prob=0.1, leaf_prob=0.2) 

exprs = set()
attempts = 0

while len(exprs) < 100 and attempts < 1000:
    attempts += 1
    expr = generator.generate_expr()
    if expr is None:
        continue
    if expr.is_constant():
        continue
    if str(expr) in ['x', '-x'] and len(exprs) > 5:
        continue
    expr_str = str(expr)
    if expr_str not in exprs:
        exprs.add(expr_str)
        print(f"{len(exprs)}. {expr_str}")

print(f"\nВсего сгенерировано уникальных неконстантных функций: {len(exprs)}")