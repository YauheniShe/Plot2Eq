import torch

from plot2eq.inference.fit_constants import fit_constants


def predict_best_equation(
    model, points_tensor, tokenizer, beam_size=5, length_penalty=0.01
):

    model.eval()
    sos_idx = tokenizer.token_map["<sos>"]
    eos_idx = tokenizer.token_map["<eos>"]

    candidates = model.beam_search(points_tensor, sos_idx, eos_idx, beam_size=beam_size)
    candidates = candidates[0]

    best_expr = None
    best_score = float("inf")
    best_mse = float("inf")
    best_params = None

    seen_exprs = set()

    for i in range(beam_size):
        tokens = candidates[i]
        seq_length = (
            (~torch.isin(tokens, torch.tensor([0, 1, 2], device=tokens.device)))
            .sum()
            .item()
        )
        try:
            expr = tokenizer.token_seq_to_expr(tokens)
            expr_str = str(expr)

            if expr_str in seen_exprs:
                continue

            seen_exprs.add(expr_str)

            final_expr, popt, mse = fit_constants(points_tensor, expr)

            if mse == float("inf"):
                continue

            score = mse + length_penalty * seq_length

            if score < best_score:
                best_score = score
                best_expr = final_expr
                best_mse = mse
                best_params = popt

        except Exception:
            continue

    return {
        "best_expr": best_expr,
        "score": best_score,
        "mse": best_mse,
        "params": best_params,
    }
