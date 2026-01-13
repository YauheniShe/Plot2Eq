import typing as tp
import sympy as sp
import numpy as np
import numpy.typing as npt

class Tokenizer:
    def __init__(self) -> None:
        self.token_map = {
            '<PAD>': 0, '<SOS>': 1, '<EOS>' : 2,
            'Add': 3, 'Mul': 4, 'Pow': 5,
            'sin': 6, 'cos': 7, 'exp': 8, 'log': 9, 'sqrt': 10, 'Abs': 11,
            'x': 12, 'CONST': 13,
            '-1': 14, '-2': 15, '-3': 16, '2': 17, '3': 18
        }

        self.id_map = ['<PAD>', 'SOS', '<EOD>',
                       sp.Add, sp.Mul, sp.Pow,
                       sp.sin, sp.cos, sp.exp, sp.log, sp.sqrt, sp.Abs,
                        sp.Symbol('x'), 'CONST', -1, -2, -3, 2, 3]
        
        self.arity_map = {
            'Add': 2,
            'Mul': 2,
            'Pow': 2,
            'sin': 1,
            'cos': 1,
            'exp': 1,
            'log': 1,
            'sqrt': 1,
            'Abs': 1
        }


    def _expr_to_token_seq_recursive(self, expr: sp.Expr) -> npt.NDArray[np.int8]:
        # Переменная
        if expr.is_Symbol:
            token_id = self.token_map['x']
            return np.array([token_id], dtype=np.int8)
        
        # Константы
        if expr.is_Number:
            token_id = self.token_map['CONST']
            return np.array([token_id], dtype=np.int8)

        # Операторы
        name = expr.func.__name__
        
        if name not in self.token_map:
            raise RuntimeError(f'Неизвестный оператор: {name}')
            
        token_id = self.token_map[name]
        token_id_seq = np.array([token_id], dtype=np.int8)

        # Pow
        if name == 'Pow':
            base, exponent = expr.args
            base = tp.cast(sp.Expr, base)
            exponent = tp.cast(sp.Expr, exponent)
            # IntPow
            if exponent.is_Integer:
                first_arg = self._expr_to_token_seq_recursive(base)
                expr_str = str(exponent)
                if expr_str in self.token_map:
                    second_arg_id = self.token_map[expr_str]
                else:
                    second_arg_id = self.token_map['CONST']
                
                second_arg = np.array([second_arg_id], dtype=np.int8)
                
                # Seq: [Pow, Base, ExponentToken]
                return np.concatenate((token_id_seq, first_arg, second_arg))
            
            # a^f(x)
            if not exponent.is_Number:
                # Проверка на exp(x)
                if base.is_Number:
                    base = tp.cast(sp.Number, base)
                    if np.abs(float(base) - np.e) <= 1e-2:
                        op = np.array([self.token_map['exp']], dtype=np.int8)
                        arg = self._expr_to_token_seq_recursive(exponent)
                        return np.concatenate((op, arg))
                
                first_arg = np.array([self.token_map['CONST']], dtype=np.int8)
                second_arg = self._expr_to_token_seq_recursive(exponent)
                
                # Seq: [Pow, CONST, ExponentExpr]
                return np.concatenate((token_id_seq, first_arg, second_arg))

            # sqrt:
            if np.abs(float(exponent) - 0.5) <= 1e-6:
                op_sqrt = np.array([self.token_map['sqrt']], dtype=np.int8)
                arg = self._expr_to_token_seq_recursive(base)
                return np.concatenate((op_sqrt, arg))
            
            # Остальные степени
            first_arg = self._expr_to_token_seq_recursive(base)
            second_arg = np.array([self.token_map['CONST']], dtype=np.int8)
            return np.concatenate((token_id_seq, first_arg, second_arg))

        # Остальные операторы
        args = expr.args
        if len(args) == 1:
            arg = self._expr_to_token_seq_recursive(tp.cast(sp.Expr, args[0]))
            return np.concatenate((token_id_seq, arg))
            
        if len(args) >= 2:
            op_tokens = np.tile(token_id_seq, len(args) - 1)
            arg_token_list = [self._expr_to_token_seq_recursive(tp.cast(sp.Expr, arg)) for arg in args]
            return np.concatenate((op_tokens, *arg_token_list))
        
        raise RuntimeError("Что-то пошло не так")
        

    
    def expr_to_token_seq(self, expr: sp.Expr) -> npt.NDArray[np.int8]:
        seq = self._expr_to_token_seq_recursive(expr)
        start = np.array([self.token_map['<SOS>']], dtype=np.int8)
        end = np.array([self.token_map['<EOS>']], dtype=np.int8)
        return np.concatenate((start, seq, end))
    
    def token_seq_to_expr(self, tokens: npt.NDArray[np.int8]) -> sp.Expr:
        stack = []
        for id in tokens[::-1]:
            if id == 0 or id == 1 or id == 2:
                continue
            token = self.id_map[id]
            if type(token) is int:
                token = sp.Integer(token)
                stack.append(token)
                continue
            if token == 'CONST':
                token = sp.Symbol('CONST')
                stack.append(token)
                continue
            if token == sp.Symbol('x'):
                stack.append(token)
                continue
            arity = self.arity_map[token.__name__]
            if arity == 1:
                arg = stack[-1]
                stack.pop()
                token = token(arg, evaluate=False)
                stack.append(token)
                continue
            if arity == 2:
                first_arg = stack[-1]
                stack.pop()
                second_arg = stack[-1]
                stack.pop()
                args = (first_arg, second_arg)
                token = token(*args, evaluate=False)
                stack.append(token)
                continue
            raise RuntimeError('Что-то пошло не так')
        return stack[-1]