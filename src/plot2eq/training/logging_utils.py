import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
import wandb

from plot2eq.inference.pipeline import predict_top_k_equations


@torch.no_grad()
def create_val_predictions_table(model, points, true_tokens, tokenizer, num_examples=5):

    num_examples = min(num_examples, points.size(0))
    sample_points = points[:num_examples]
    sample_true_tokens = true_tokens[:num_examples]

    val_table = wandb.Table(
        columns=["Plot", "True Equation", "Predicted Equation", "MSE", "Score"]
    )

    for i in range(num_examples):
        y_vals = sample_points[i, 0].cpu().numpy()
        mask = sample_points[i, 1].cpu().numpy().astype(bool)
        x_vals = np.linspace(0, 1, len(y_vals))

        x_data_fit = x_vals[mask]
        y_data_fit = y_vals[mask]

        results = predict_top_k_equations(
            model,
            points_tensor=sample_points[i : i + 1],
            tokenizer=tokenizer,
            x_data=x_data_fit,
            y_data=y_data_fit,
            beam_size=5,
            top_k=1,
            length_penalty=0.01,
        )

        if results:
            res = results[0]
            pred_expr = res["best_expr"]
            mse = res["mse"]
            score = res["score"]
        else:
            pred_expr = None
            mse = float("inf")
            score = 0.0

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x_data_fit, y_data_fit, color="black", linewidth=3, label="Target")
        pred_str = "Parse Error / Fit Failed"

        if pred_expr is not None:
            pred_str = f"$${sp.latex(pred_expr)}$$"
            x_sym = sp.Symbol("x", real=True)
            f_expr = sp.lambdify(x_sym, pred_expr, modules=["numpy"])
            try:
                preds_y = f_expr(x_data_fit)
                if np.isscalar(preds_y):
                    preds_y = np.full_like(x_data_fit, preds_y)
                ax.plot(
                    x_data_fit,
                    preds_y,
                    color="red",
                    linestyle="--",
                    linewidth=2.5,
                    label="Predicted",
                )
            except Exception:
                pass

        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        plt.tight_layout()

        true_toks = [
            t.item() for t in sample_true_tokens[i] if t.item() not in (0, 1, 2)
        ]
        try:
            true_expr = tokenizer.token_seq_to_expr(torch.tensor(true_toks))
            true_str = f"$${sp.latex(true_expr)}$$"
        except Exception:
            true_str = "Parse Error"

        val_table.add_data(
            wandb.Image(fig), true_str, pred_str, f"{mse:.5f}", f"{score:.5f}"
        )
        plt.close(fig)

    return val_table
