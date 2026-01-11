import pandas as pd
import numpy as np
import sympy as sp
import warnings
import time

warnings.filterwarnings('ignore', category=RuntimeWarning)

from expression import ExpressionGenerator

class Tokenizer:
    def __init__(self) -> None:
        self.token_map = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>' : 2,
            'Add': 3, 'Mul': 4, 'Pow': 5,
            'sin': 6, 'cos': 7, 'exp': 8, 'log': 9, 'sqrt': 10, 'Abs': 11,
            'x': 12, 'pi': 13, 'E': 14, 'CONST': 15
        }

    def expr_to_token_seq(self, expr):
        pass

    def token_seq_to_expr(self, tokens):
        pass

    

class DataGenerator:
    def __init__(self, max_depth: int, step: float, const_prob: float, leaf_prob: float,
                 min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        self.max_depth = max_depth
        self.step = step
        self.const_prob = const_prob
        self.generator = ExpressionGenerator(max_depth, const_prob, leaf_prob)
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def generate_points(self, expr):
        steps = int((self.max_x - self.min_x) / self.step)
        f = sp.lambdify(sp.Symbol('x'), expr, modules='numpy')
        x_values = np.linspace(num=steps, start=self.min_x, stop = self.max_x)
        try:
            y_values = f(x_values)
            if np.isscalar(y_values) or np.ndim(y_values) == 0:
                y_values = np.full_like(x_values, y_values)
            if np.iscomplexobj(y_values):
                return None
        except:
            return None
        mask = np.isfinite(y_values) & (y_values < self.max_y) & (y_values > self.min_y)
        if np.sum(mask) / np.size(mask) <= 0.3:
            return None
        valid_y = y_values[mask]
        dy = np.diff(valid_y)
        mean_change = np.mean(np.abs(dy))
        if mean_change > 0.5:
            return None
        sign_changes = np.sum(np.diff(np.sign(dy)) != 0)
        if sign_changes > len(valid_y) * 0.3:
            return None
        y_values[~mask] = 0.0
        return np.vstack((x_values, y_values, mask.astype(float))).T
        
    
    def expr_to_tokens_id(self, expr):
        pass #TODO

    def generate_data(self):
        pass


data_gen  = DataGenerator(max_depth=4, step=0.02, const_prob=0.1, leaf_prob=0.2, min_x=-10, max_x=10, min_y=-10, max_y=10)
exprs = set()
attempts = 0
start = time.time()
while len(exprs) < 100 and attempts < 1000:
    attempts += 1
    expr = data_gen.generator.generate_expr()
    if expr is None:
        continue
    if expr.is_constant():
        continue
    if str(expr) in ['x', '-x'] and len(exprs) > 5:
        continue
    if data_gen.generate_points(expr) is None:
        continue
    expr_str = str(expr)
    if expr_str not in exprs:
        exprs.add(expr_str)
        print(f"{len(exprs)}. {expr_str}")

print(f"\nВсего сгенерировано уникальных неконстантных функций: {len(exprs)}")
print(f"Всего затрачено {time.time() - start}")