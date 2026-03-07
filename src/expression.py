import _thread
import logging
import random
import threading
from contextlib import contextmanager

import numpy as np
import sympy as sp


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    timer = threading.Timer(seconds, _thread.interrupt_main)
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out!")
    finally:
        timer.cancel()


x = sp.Symbol("x", real=True)

UNARY_OPS = {
    "sin": {"symbol": sp.sin, "weight": 4},
    "cos": {"symbol": sp.cos, "weight": 4},
    "exp": {"symbol": sp.exp, "weight": 4},
    "ln": {"symbol": sp.log, "weight": 4},
    "sqrt": {"symbol": sp.sqrt, "weight": 4},
    "abs": {"symbol": sp.Abs, "weight": 4},
    "tan": {"symbol": sp.tan, "weight": 4},
    "arcsin": {"symbol": sp.asin, "weight": 2},
    "pow2": {"symbol": lambda arg: sp.Pow(arg, sp.Integer(2)), "weight": 4},
    "pow3": {"symbol": lambda arg: sp.Pow(arg, sp.Integer(3)), "weight": 2},
    "pow4": {"symbol": lambda arg: sp.Pow(arg, sp.Integer(4)), "weight": 1},
    "pow5": {"symbol": lambda arg: sp.Pow(arg, sp.Integer(5)), "weight": 1},
}

BINARY_OPS = {
    "Add": {"symbol": sp.Add, "weight": 10},
    "Sub": {"symbol": lambda a, b: a - b, "weight": 5},
    "Mul": {"symbol": sp.Mul, "weight": 10},
    "Div": {"symbol": lambda a, b: a / b, "weight": 5},
}

CONSTANTS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


class ExpressionGenerator:
    def __init__(
        self,
        max_ops: int,
        timeout: int,
        unary_ops: dict = UNARY_OPS,
        binary_ops: dict = BINARY_OPS,
        constants: list = CONSTANTS,
    ) -> None:
        """
        :param max_ops: Количество операторов (внутренних узлов) в выражении (n в статье).
        :param unary_ops: Словарь унарных операторов.
        :param binary_ops: Словарь бинарных операторов.
        :param leaves: Список всех возможных листьев (переменные + константы), равномерное распределение L.
        :param timeout: Таймаут для генерации и упрощения выражения.
        """
        self.max_ops = max_ops
        self.timeout = timeout

        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.constants = constants

        self.unary_ops_keys = list(unary_ops.keys())
        self.unary_ops_weights = [op["weight"] for op in unary_ops.values()]
        self.p1 = sum(self.unary_ops_weights)

        self.binary_ops_keys = list(binary_ops.keys())
        self.binary_ops_weights = [op["weight"] for op in binary_ops.values()]
        self.p2 = sum(self.binary_ops_weights)

        self.L = 1
        self.D = [[0] * (max_ops + 1) for _ in range(max_ops + 2)]

        for e in range(max_ops + 2):
            self.D[e][0] = self.L**e

        for n in range(1, max_ops + 1):
            for e in range(1, max_ops - n + 2):
                self.D[e][n] = self.L * self.D[e - 1][n] + self.p1 * self.D[e][n - 1]
                if e + 1 < max_ops + 2:
                    self.D[e][n] += self.p2 * self.D[e + 1][n - 1]

    def _sample_leaf(self):
        if random.random() < 0.8:
            return x
        else:
            return sp.Integer(random.choice(self.constants))

    def _skeletonize(self, expr, c_list):
        def get_c():
            c = sp.Symbol(f"C_{len(c_list)}", real=True)
            c_list.append(c)
            return c

        if expr == x:
            return sp.Add(sp.Mul(get_c(), x, evaluate=False), get_c(), evaluate=False)

        if expr.is_Number:
            return get_c()

        if expr.is_Add:
            return sp.Add(
                *[self._skeletonize(arg, c_list) for arg in expr.args], evaluate=False
            )

        if expr.is_Mul:
            return sp.Mul(
                *[self._skeletonize(arg, c_list) for arg in expr.args], evaluate=False
            )

        if expr.is_Pow:
            base = self._skeletonize(expr.args[0], c_list)
            if expr.args[1].has(x):
                expo = self._skeletonize(expr.args[1], c_list)
            else:
                expo = expr.args[1]
            pow_expr = sp.Pow(base, expo, evaluate=False)
            return sp.Mul(get_c(), pow_expr, evaluate=False)

        if expr.is_Function:
            arg = self._skeletonize(expr.args[0], c_list)
            return sp.Mul(get_c(), expr.func(arg), evaluate=False)

        return expr

    def _clean_skeleton(self, expr):
        C_tok = sp.Symbol("C_tok", real=True)

        # 1. Раскрываем скобки
        try:
            expr = sp.expand(expr)
        except Exception as e:
            logging.warning(f"Ошибка при раскрытии скобок sp.expand: {e}")

        # 2. Умные правила поглощения (Constant Folding)
        def _fold(node):
            if node == x:
                return x
            if isinstance(node, sp.Symbol) and str(node).startswith("C_"):
                return C_tok
            if isinstance(node, sp.Number):
                return node  # Базовые числа (2, -1, 1/2) пока сохраняем

            # Рекурсивно собираем узлы снизу вверх
            args = [_fold(arg) for arg in node.args]
            new_expr = node.func(*args) if args else node

            # ПРАВИЛО 1: В поддереве ВООБЩЕ НЕТ переменной x
            if not getattr(new_expr, "has", lambda var: False)(x):
                # Если там уже затесался C_tok (например, C_tok + 2) -> всё съедается в C_tok
                if getattr(new_expr, "has", lambda var: False)(C_tok):
                    return C_tok
                # Если это чистое структурное число (например 2 или 1/2) -> оставляем
                if isinstance(new_expr, sp.Number):
                    return new_expr
                # ВАШ СЛУЧАЙ: всякие tan(4), exp(2) попадают сюда и беспощадно превращаются в C_tok!
                return C_tok

            # ПРАВИЛО 2: Поддерево содержит x (схлопываем множители и слагаемые)
            if new_expr.is_Mul:
                has_x = [
                    a for a in new_expr.args if getattr(a, "has", lambda var: False)(x)
                ]
                no_x = [
                    a
                    for a in new_expr.args
                    if not getattr(a, "has", lambda var: False)(x)
                ]

                if any(
                    a == C_tok or getattr(a, "has", lambda var: False)(C_tok)
                    for a in no_x
                ):
                    return sp.Mul(C_tok, *has_x)
                return new_expr

            if new_expr.is_Add:
                has_x = []
                no_x = []
                for a in new_expr.args:
                    if getattr(a, "has", lambda var: False)(x):
                        has_x.append(a)
                    else:
                        no_x.append(a)

                if any(getattr(a, "has", lambda var: False)(C_tok) for a in no_x):
                    collapsed_no_x = C_tok
                elif no_x:
                    collapsed_no_x = sp.Add(*no_x)
                else:
                    collapsed_no_x = sp.Integer(0)

                x_terms = {}
                for term in has_x:
                    if term.is_Mul:
                        c_parts = [
                            a
                            for a in term.args
                            if not getattr(a, "has", lambda var: False)(x)
                        ]
                        x_parts = [
                            a
                            for a in term.args
                            if getattr(a, "has", lambda var: False)(x)
                        ]
                        x_key = sp.Mul(*x_parts)
                        c_key = sp.Mul(*c_parts) if c_parts else sp.Integer(1)
                    else:
                        x_key = term
                        c_key = sp.Integer(1)

                    if x_key in x_terms:
                        x_terms[x_key] = x_terms[x_key] + c_key
                    else:
                        x_terms[x_key] = c_key

                res_terms = []
                if collapsed_no_x != sp.Integer(0):
                    res_terms.append(collapsed_no_x)

                for x_key, c_val in x_terms.items():
                    if getattr(c_val, "has", lambda var: False)(C_tok):
                        res_terms.append(C_tok * x_key)
                    else:
                        res_terms.append(c_val * x_key)

                if not res_terms:
                    return sp.Integer(0)
                if len(res_terms) == 1:
                    return res_terms[0]
                return sp.Add(*res_terms)

            return new_expr

        folded = expr
        while True:
            new_folded = _fold(folded)
            if new_folded == folded:
                break
            folded = new_folded

        new_c_list = []

        def _reindex(node):
            if node == C_tok:
                c = sp.Symbol(f"C_{len(new_c_list)}", real=True)
                new_c_list.append(c)
                return c
            if not getattr(node, "args", None):
                return node
            new_args = [_reindex(arg) for arg in node.args]
            return node.func(*new_args)

        clean_expr = _reindex(folded)
        return clean_expr, new_c_list

    def _generate_raw(self, fixed_n: int):
        """
        Генерирует AST дерево в префиксной нотации (Алгоритм 2 из статьи),
        и собирает из него SymPy выражение.
        """
        e = 1
        n = fixed_n
        tree_nodes = []

        while n > 0:
            probs = []
            choices = []
            for k in range(e):
                if self.p1 > 0:
                    w1 = (self.L**k) * self.p1 * self.D[e - k][n - 1]
                    if w1 > 0:
                        probs.append(w1)
                        choices.append((k, 1))

                if self.p2 > 0 and (e - k + 1) < len(self.D):
                    w2 = (self.L**k) * self.p2 * self.D[e - k + 1][n - 1]
                    if w2 > 0:
                        probs.append(w2)
                        choices.append((k, 2))

            if sum(probs) == 0:
                return None

            k, arity = random.choices(choices, weights=probs, k=1)[0]

            for _ in range(k):
                tree_nodes.append(("leaf", None))

            if arity == 1:
                op_name = random.choices(
                    self.unary_ops_keys, weights=self.unary_ops_weights, k=1
                )[0]
                tree_nodes.append(("unary", op_name))
                e = e - k
            else:
                op_name = random.choices(
                    self.binary_ops_keys, weights=self.binary_ops_weights, k=1
                )[0]
                tree_nodes.append(("binary", op_name))
                e = e - k + 1

            n -= 1

        for _ in range(e):
            tree_nodes.append(("leaf", None))

        iterator = iter(tree_nodes)

        def build_ast():
            try:
                token_type, value = next(iterator)
            except StopIteration:
                return None

            if token_type == "leaf":
                return self._sample_leaf()

            elif token_type == "unary":
                arg = build_ast()
                if arg is None:
                    return None
                func = self.unary_ops[value]["symbol"]
                return func(arg)
            elif token_type == "binary":
                arg1 = build_ast()
                arg2 = build_ast()
                if arg1 is None or arg2 is None:
                    return None
                func = self.binary_ops[value]["symbol"]
                return func(arg1, arg2)

        return build_ast()

    def generate_expr(self):
        n = random.randint(1, self.max_ops)
        while True:
            stage = "INIT"
            expr = None
            try:
                with time_limit(self.timeout):
                    stage = "GENERATE"
                    expr = self._generate_raw(n)
                    if expr is None:
                        continue

                    stage = "SIMPLIFY"
                    expr = sp.simplify(expr)

                    if (
                        expr.has(sp.I, sp.nan, sp.zoo, sp.oo, -sp.oo)
                        or expr.is_constant()
                    ):
                        continue

                    stage = "SKELETONIZE"
                    c_list = []
                    raw_skeleton = self._skeletonize(expr, c_list)

                    stage = "CLEAN_SKELETON"
                    skeleton, c_list = self._clean_skeleton(raw_skeleton)

                    if not skeleton.has(x):
                        continue

                    max_c = min(3, len(c_list))
                    num_random_c = random.randint(0, max_c)

                    chosen_cs = set(random.sample(c_list, num_random_c))
                    inactive_c_values = {
                        c: sp.Integer(1) for c in c_list if c not in chosen_cs
                    }

                    expr_instantiated = skeleton.subs(inactive_c_values)

                    c_values = {}

                    for c in chosen_cs:
                        val = random.uniform(-5, 5)
                        if abs(val) < 0.5:
                            val += np.sign(val)
                        c_values[c] = val

                    expr_instantiated = expr_instantiated.subs(c_values)

                    return skeleton, expr, expr_instantiated

            except TimeoutException:
                try:
                    bad_expr = str(expr) if expr is not None else "None"
                except Exception:
                    bad_expr = "Неизвестное выражение (стадия генерации)"
                logging.error(
                    f"Timeout[{stage}] {self.timeout} секунд. Пропущено выражение: {bad_expr}"
                )
                continue
            except Exception:
                logging.exception("Ошибка при генерации выражения.")
                continue
