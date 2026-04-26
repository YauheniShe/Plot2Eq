import sympy as sp
import random
import signal
import logging

from contextlib import contextmanager

class TimeoutException(Exception): 
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

OPERATORS = {
    'Add': {'symbol': sp.Add, 'arity': 2, 'weight': 5.0},
    'Sub': {'symbol': lambda x, y: x - y, 'arity': 2, 'weight': 5.0},
    'Mul': {'symbol': sp.Mul, 'arity': 2, 'weight': 5.0},
    'Div': {'symbol': lambda x, y: x / y, 'arity': 2, 'weight': 2.0},
    'IntPow': {'symbol': sp.Pow, 'arity': 2, 'weight': 1.0},
    'sin': {'symbol': sp.sin, 'arity': 1, 'weight': 1.0},
    'cos': {'symbol': sp.cos, 'arity': 1, 'weight': 1.0},
    'exp': {'symbol': sp.exp, 'arity': 1, 'weight': 1.0},
    'log': {'symbol': sp.log, 'arity': 1, 'weight': 2.0},
    'sqrt': {'symbol': sp.sqrt, 'arity': 1, 'weight': 2.0},
    'sqr':  {'symbol': lambda x: x**2, 'arity': 1, 'weight': 3.0},
    'abs': {'symbol': sp.Abs, 'arity': 1, 'weight': 4.0},
    'Neg': {'symbol': lambda x: -x, 'arity': 1, 'weight': 2.0},
    'GenExp': {'symbol': sp.Pow, 'arity': 2, 'weight': 1.0}
}

VARIABLES = ['x']
CONSTANT = ['C']

class ExpressionGenerator:
    def __init__(self, max_depth: int, const_prob: float, leaf_prob: float, timeout: int) -> None:
        self.max_depth  = max_depth
        self.const_prob = const_prob
        self.leaf_prob = leaf_prob
        self.timeout = timeout 

    def _smart_clean(self, expr):
        expr = expr.evalf()
        expr = expr.subs(sp.E, sp.E.evalf())
        expr = expr.subs(sp.pi, sp.pi.evalf())
        def replace_trivial_floats(node):
            if node.is_Float:
                if node == 1.0:
                    return sp.Integer(1)
                if node == -1.0:
                    return sp.Integer(-1)
                if node == 0.0:
                    return sp.Integer(0)
            return node
        return expr.replace(lambda x: x.is_Float, replace_trivial_floats)
    
    def generate_expr(self, depth=None):
        if depth is None:
            depth = self.max_depth
        while True:
            stage = 'INIT'
            expr = None
            try:
                with time_limit(self.timeout):
                    stage = 'GENERATE'
                    expr = self._generate_recursive(depth)
                    if expr is None:
                        continue
                    stage = 'SIMPLIFY'
                    expr = expr.simplify()
                    if expr.has(sp.I) or expr.has(sp.nan) or expr.has(sp.zoo) or expr.is_constant():
                        continue
                    stage = 'CLEAN'
                    expr = self._smart_clean(expr)
                    stage = 'ROUND'
                    expr = self._round_floats(expr)
                    return expr
            except TimeoutException:
                try:
                    bad_expr = str(expr) if expr is not None else 'None'
                except:
                    bad_expr = 'Неизвестное выражение (стадия генерации)'
                logging.error(f"Timeout[{stage}] {self.timeout} секунд. Пропущено выражение: {bad_expr}")
                continue

            except Exception as e:
                logging.exception("Ошибка при генерации выражения.")
                continue

    def _generate_recursive(self, depth=None):
        if depth is None:
            depth = self.max_depth
        is_root = (depth == self.max_depth)
        if not is_root and (random.random() < self.const_prob or depth == 0):
            return self._generate_leaf()
        weights = [OPERATORS[name]['weight'] for name in OPERATORS.keys()]
        op_name = random.choices(list(OPERATORS.keys()), weights=weights, k=1)[0]
        op_info = OPERATORS[op_name]
        arity = op_info['arity']
        symbol = op_info['symbol']
        if arity == 1:
            args = self._generate_recursive(depth - 1)
            return symbol(args)
        if arity == 2:
            if op_name == 'IntPow':
                first_arg = self._generate_recursive(depth - 1)
                exponents = [-3, -2, -1, 2, 3]
                second_arg = sp.Integer(random.choice(exponents))
                args = (first_arg, second_arg)
                return symbol(*args)
            if op_name == 'GenExp':
                val = round(random.uniform(0, 5), 3)
                first_arg = sp.Float(val)
                second_arg = self._generate_recursive(depth - 1)
                args = (first_arg, second_arg)
                return symbol(*args)
            first_arg = self._generate_recursive(depth - 1)
            second_arg = self._generate_recursive(depth - 1)
            args = (first_arg, second_arg)
            return symbol(*args)

    def _round_floats(self, expr, precision=3):
        return expr.xreplace({
            n: round(n, precision) for n in expr.atoms(sp.Float)
        })

    def _generate_leaf(self):
        if random.random() < self.leaf_prob:
            c_type = random.choice(CONSTANT)
            if c_type == 'C':
                val = round(random.uniform(-5, 5), 3)
                return sp.Float(val)
        else:
            var = random.choice(VARIABLES)
            return sp.Symbol(var, real=True)
