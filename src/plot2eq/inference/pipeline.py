import os
from concurrent.futures import ProcessPoolExecutor

import torch

from plot2eq.inference.fit_constants import fit_constants


def _worker_process(args):
    (
        expr,
        seq_length,
        x_data,
        y_data,
        opt_max_iter,
        opt_popsize,
        fast_mode,
        length_penalty,
    ) = args
    try:
        final_expr, popt, mse = fit_constants(
            expr,
            x_data,
            y_data,
            max_iter=opt_max_iter,
            popsize=opt_popsize,
            fast_mode=fast_mode,
        )

        if mse == float("inf"):
            return None

        score = 100.0 / (1.0 + mse + length_penalty * seq_length)

        return {
            "skeleton": str(expr),
            "best_expr": final_expr,
            "score": score,
            "mse": float(mse),
            "params": popt.tolist() if hasattr(popt, "tolist") else popt,  # type: ignore
        }
    except Exception:
        return None


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

    candidates = model.beam_search(
        points_tensor, sos_idx, eos_idx, beam_size=beam_size
    )[0]

    unique_candidates = {}
    for i in range(beam_size):
        tokens = candidates[i]
        special_tokens = torch.tensor([pad_idx, sos_idx, eos_idx], device=tokens.device)
        seq_length = (~torch.isin(tokens, special_tokens)).sum().item()
        try:
            expr = tokenizer.token_seq_to_expr(tokens)
            expr_str = str(expr)
            if expr_str not in unique_candidates:
                unique_candidates[expr_str] = (expr, seq_length)
        except Exception:
            continue

    tasks = [
        (
            expr,
            seq_length,
            x_data,
            y_data,
            opt_max_iter,
            opt_popsize,
            fast_mode,
            length_penalty,
        )
        for expr_str, (expr, seq_length) in unique_candidates.items()
    ]

    results = []
    n_cores = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        for res in executor.map(_worker_process, tasks):
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]
