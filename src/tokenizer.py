import typing as tp

import numpy as np
import numpy.typing as npt
import sympy as sp

x = sp.Symbol("x", real=True)


class TokenizerError(Exception):
    """Базовый класс для всех ошибок токенизатора."""

    pass


class InvalidExpressionError(TokenizerError):
    def __init__(self, invalid_expr: sp.Expr, reason: str = "") -> None:
        self.invalid_expr = invalid_expr
        self.reason = reason

        message = f"Неподдерживаемое выражение: {invalid_expr}"
        if reason:
            message += f"Причина: {reason}"

        super().__init__(message)


class UnknownTokenError(TokenizerError):
    def __init__(self, expr: sp.Expr, token_name: str) -> None:
        self.token_name = token_name
        self.expr = expr
        super().__init__(
            f"Неизвестный оператор или токен: '{token_name}' в выражении {expr}"
        )


class TokenDecodingError(TokenizerError):
    pass


class Tokenizer:
    def __init__(self) -> None:

        self.tokens = [
            ("<pad>", 0),
            ("<sos>", 1),
            ("<eos>", 2),
            ("x", sp.Symbol("x", real=True)),
            ("C", sp.Symbol("C", real=True)),
            ("Add", sp.Add),
            ("cos", sp.cos),
            ("exp", sp.exp),
            ("log", sp.log),
            ("Mul", sp.Mul),
            ("Pow", sp.Pow),
            ("sin", sp.sin),
            ("sqrt", sp.sqrt),
            ("tan", sp.tan),
            ("Abs", sp.Abs),
            ("-5", sp.Integer(-5)),
            ("-4", sp.Integer(-4)),
            ("-3", sp.Integer(-3)),
            ("-2", sp.Integer(-2)),
            ("-1", sp.Integer(-1)),
            ("1", sp.Integer(1)),
            ("2", sp.Integer(2)),
            ("3", sp.Integer(3)),
            ("4", sp.Integer(4)),
            ("5", sp.Integer(5)),
        ]

        self.token_map = {name: idx for idx, (name, _) in enumerate(self.tokens)}

        self.id_to_str = {idx: name for idx, (name, _) in enumerate(self.tokens)}

        self.id_map = [
            name if name in ("<pad>", "<sos>", "<eos>") else obj
            for name, obj in self.tokens
        ]

        self.arity_map = {
            "Add": 2,
            "cos": 1,
            "exp": 1,
            "log": 1,
            "Mul": 2,
            "Pow": 2,
            "sin": 1,
            "sqrt": 1,
            "tan": 1,
            "Abs": 1,
        }

    def _expr_to_token_seq_recursive(
        self, current_expr: sp.Expr, start_expr: sp.Expr
    ) -> list:
        # Переменная
        if current_expr == x:
            token_id = self.token_map["x"]
            return [token_id]

        # Константы
        if current_expr.is_Symbol and str(current_expr).startswith("C_"):
            token_id = self.token_map["C"]
            return [token_id]

        # Целые числа-степени, котоырые являются аргументом Pow
        if current_expr.is_Number and current_expr.is_Integer:
            if str(current_expr) not in self.token_map:
                raise InvalidExpressionError(
                    start_expr, "Степень в Pow выходит за рамки допустимых значений"
                )
            value = int(current_expr)
            token_id = self.token_map[str(value)]
            return [token_id]

        if current_expr.is_Rational and not current_expr.is_Integer:
            current_expr = tp.cast(sp.Rational, current_expr)
            p = current_expr.p
            q = current_expr.q
            if str(p) not in self.token_map or str(q) not in self.token_map:
                raise InvalidExpressionError(
                    start_expr,
                    f"Отсутствуют токены для числителя или знаменателя рационального числа {current_expr}",
                )

            p_token_id = self.token_map[str(p)]
            q_token_id = self.token_map[str(q)]

            return [
                self.token_map["Mul"],
                p_token_id,
                self.token_map["Pow"],
                q_token_id,
                self.token_map["-1"],
            ]

        # Операторы
        name = current_expr.func.__name__

        if name not in self.token_map:
            raise UnknownTokenError(start_expr, name)

        token_id = self.token_map[name]
        token_id_seq = [token_id]

        # Pow
        if name == "Pow":
            base, exponent = current_expr.args

            if exponent == sp.S.Half:
                token_id = self.token_map["sqrt"]
                arg = self._expr_to_token_seq_recursive(
                    tp.cast(sp.Expr, base), start_expr
                )
                return [token_id] + arg

            if exponent == sp.Rational(-1, 2):
                arg = self._expr_to_token_seq_recursive(
                    tp.cast(sp.Expr, base), start_expr
                )
                return [token_id, self.token_map["sqrt"]] + arg + [self.token_map["-1"]]

            if base.has(x) and exponent.has(x):
                raise InvalidExpressionError(
                    start_expr, "symplify сгенерировал выражение вида f(x)^g(x)"
                )

            base = tp.cast(sp.Expr, base)
            exponent = tp.cast(sp.Expr, exponent)

            first_arg = self._expr_to_token_seq_recursive(base, start_expr)
            second_arg = self._expr_to_token_seq_recursive(exponent, start_expr)
            return token_id_seq + first_arg + second_arg

        # Остальные операторы
        args = current_expr.args
        if len(args) == 1:
            arg = self._expr_to_token_seq_recursive(
                tp.cast(sp.Expr, args[0]), start_expr
            )
            return token_id_seq + arg

        if len(args) >= 2:
            op_tokens = token_id_seq * (len(args) - 1)
            arg_tokens = []
            for arg in args:
                arg_tokens.extend(
                    self._expr_to_token_seq_recursive(tp.cast(sp.Expr, arg), start_expr)
                )
            return op_tokens + arg_tokens

        raise InvalidExpressionError(start_expr, "не удалось распарсить выражение")

    def expr_to_token_seq(self, expr: sp.Expr) -> npt.NDArray:
        try:
            seq = np.array(self._expr_to_token_seq_recursive(expr, start_expr=expr))
            start = np.array([self.token_map["<sos>"]])
            end = np.array([self.token_map["<eos>"]])
            return np.concatenate((start, seq, end))
        except TokenizerError:
            raise
        except Exception as e:
            raise TokenizerError(
                f"Внутренняя ошибка при токенизации выражения {expr}"
            ) from e

    def token_seq_to_expr(self, tokens: npt.NDArray) -> sp.Expr:
        stack = []

        for id in tokens[::-1]:
            if id == 0 or id == 1 or id == 2:
                continue

            if id not in self.id_to_str:
                raise TokenDecodingError(f"Неизвестный ID токена: {id}")

            token = self.id_map[id]
            name = self.id_to_str[id]

            if name not in self.arity_map:
                stack.append(token)
                continue

            arity = self.arity_map[token.__name__]

            try:
                if arity == 1:
                    arg = stack[-1]
                    stack.pop()
                    token = token(arg, evaluate=False)
                    stack.append(token)
                    continue
                elif arity == 2:
                    first_arg = stack[-1]
                    stack.pop()
                    second_arg = stack[-1]
                    stack.pop()
                    args = (first_arg, second_arg)
                    token = token(*args, evaluate=False)
                    stack.append(token)
                    continue
                else:
                    raise TokenDecodingError(
                        f"Неподдерживаемая арность: {arity} для токена {name}"
                    )

            except IndexError as e:
                raise TokenDecodingError(
                    f"Недостаточно аргументов для оператора '{name}'. Последовательность повреждена."
                ) from e

        return stack[-1]
