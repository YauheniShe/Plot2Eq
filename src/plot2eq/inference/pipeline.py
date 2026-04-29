import torch

from plot2eq.inference.fit_constants import fit_constants


def predict_top_k_equations(
    model,
    points_tensor,
    tokenizer,
    x_data,
    y_data,
    beam_size=5,
    top_k=1,
    length_penalty=0.01,
    fast_mode=False,
    opt_max_iter=200,
    opt_popsize=15,
):
    model.eval()

    pad_idx = tokenizer.token_map["<pad>"]
    sos_idx = tokenizer.token_map["<sos>"]
    eos_idx = tokenizer.token_map["<eos>"]

    candidates = model.beam_search(points_tensor, sos_idx, eos_idx, beam_size=beam_size)
    candidates = candidates[0]

    results = []
    seen_exprs = set()

    for i in range(beam_size):
        tokens = candidates[i]
        special_tokens = torch.tensor([pad_idx, sos_idx, eos_idx], device=tokens.device)

        seq_length = (~torch.isin(tokens, special_tokens)).sum().item()
        try:
            expr = tokenizer.token_seq_to_expr(tokens)
            expr_str = str(expr)

            if expr_str in seen_exprs:
                continue

            seen_exprs.add(expr_str)

            final_expr, popt, mse = fit_constants(
                expr,
                x_data,
                y_data,
                max_iter=opt_max_iter,
                popsize=opt_popsize,
                fast_mode=fast_mode,
            )

            if mse == float("inf"):
                continue

            score = mse + length_penalty * seq_length

            results.append(
                {
                    "skeleton": str(expr),
                    "best_expr": final_expr,
                    "score": score,
                    "mse": mse,
                    "params": popt,
                }
            )

        except Exception:
            continue

    results.sort(key=lambda x: x["score"])

    return results[:top_k]
