import warnings

import numpy as np
import sympy as sp
from scipy.optimize import differential_evolution


def assign_unique_constants(expr):
    c_sym = sp.Symbol("C", real=True)
    c_list = []
    c_counter = 0

    def _walk(node):
        nonlocal c_counter

        if node == c_sym:
            new_c = sp.Symbol(f"C_{c_counter}", real=True)
            c_list.append(new_c)
            c_counter += 1
            return new_c

        if node.args:
            new_args = [_walk(arg) for arg in node.args]
            return node.func(*new_args, evaluate=False)

        return node

    new_expr = _walk(expr)
    return new_expr, c_list


def calculate_robust_mse(preds, y_data):

    if np.isscalar(preds):
        preds = np.full_like(y_data, preds, dtype=float)

    if np.iscomplexobj(preds):
        valid_complex = np.abs(np.imag(preds)) < 1e-5
        preds = np.real(preds)
        preds[~valid_complex] = np.nan

    valid_mask = np.isfinite(preds)
    valid_count = np.sum(valid_mask)
    total_count = len(y_data)

    if valid_count < 0.7 * total_count:
        return 1e9

    valid_preds = preds[valid_mask]
    valid_y = y_data[valid_mask]

    mse = np.mean((valid_preds - valid_y) ** 2)
    invalid_ratio = (total_count - valid_count) / total_count
    penalty = invalid_ratio * 1.0

    return mse + penalty


def fit_constants(points_tensor, expr, bounds_range=(-10.0, 10.0), max_iter=200):
    parameterized_expr, params = assign_unique_constants(expr)
    x_sym = sp.Symbol("x", real=True)

    y_vals = points_tensor[0].cpu().numpy()
    mask = points_tensor[1].cpu().numpy().astype(bool)
    x_vals = np.linspace(0, 1, len(y_vals))

    x_data = x_vals[mask]
    y_data = y_vals[mask]

    if len(params) == 0:
        f_expr = sp.lambdify(x_sym, parameterized_expr, modules=["numpy"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = f_expr(x_data)
                mse = calculate_robust_mse(preds, y_data)

            return parameterized_expr, [], (mse if mse < 1e8 else float("inf"))

        except Exception:
            return parameterized_expr, [], float("inf")

    f_expr = sp.lambdify((x_sym, *params), parameterized_expr, modules=["numpy"])

    def objective_func(p):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = f_expr(x_data, *p)
                return calculate_robust_mse(preds, y_data)

        except Exception:
            return 1e9

    bounds = [bounds_range for _ in params]

    try:
        result = differential_evolution(
            objective_func,
            bounds=bounds,  # type: ignore
            strategy="best1bin",
            maxiter=max_iter,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-5,
            polish=True,
            updating="deferred",
            workers=-1,
        )

        if not result.success and result.fun >= 1e8:
            return parameterized_expr, [], float("inf")

        best_popt = result.x
        best_mse = result.fun

        subs_dict = {p: round(val, 3) for p, val in zip(params, best_popt)}
        final_expr = parameterized_expr.subs(subs_dict)

        return final_expr, best_popt, best_mse

    except Exception:
        return parameterized_expr, [], float("inf")
