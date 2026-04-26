import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.amp.autocast_mode import autocast
from tqdm.auto import tqdm

from .datamodule import build_dataloaders
from .model import Plot2EqModel
from .tokenizer import Tokenizer


@dataclass
class TrainConfig:
    data_dir: str = "../data"
    resume_from_checkpoint: str | None = None
    project_name: str = "Plot2Eq"

    batch_size: int = 256
    epochs: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    d_model: int = 512
    nhead: int = 8
    num_enc_layers: int = 4
    num_dec_layers: int = 4
    dropout: float = 0.1
    max_seq_len = 128
    label_smoothing: float = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def create_val_predictions_table(model, points, true_tokens, tokenizer, num_examples=5):

    sos_idx = tokenizer.token_map["<sos>"]
    eos_idx = tokenizer.token_map["<eos>"]

    num_examples = min(num_examples, points.size(0))
    sample_points = points[:num_examples]
    sample_true_tokens = true_tokens[:num_examples]

    generated_tokens = model.generate(
        sample_points, sos_idx=sos_idx, eos_idx=eos_idx, max_len=128
    )

    val_table = wandb.Table(columns=["Plot", "True Equation", "Predicted Equation"])

    for i in range(num_examples):
        y_vals = sample_points[i, 0].cpu().numpy()
        mask = sample_points[i, 1].cpu().numpy()
        x_vals = [j / len(y_vals) for j in range(len(y_vals))]

        fig, ax = plt.subplots(figsize=(4, 3))
        y_plot = y_vals.copy()
        y_plot[mask == 0.0] = float("nan")
        ax.plot(x_vals, y_plot, color="blue", linewidth=2)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()

        true_toks = [
            t.item() for t in sample_true_tokens[i] if t.item() not in (0, 1, 2)
        ]
        try:
            true_expr = tokenizer.token_seq_to_expr(torch.tensor(true_toks))
            true_str = f"${sp.latex(true_expr)}$"
        except Exception:
            true_str = "Parse Error"

        pred_toks = [t.item() for t in generated_tokens[i] if t.item() not in (0, 1, 2)]
        try:
            pred_expr = tokenizer.token_seq_to_expr(torch.tensor(pred_toks))
            pred_str = f"${sp.latex(pred_expr)}$"
        except Exception:
            pred_str = "Invalid Syntax"

        val_table.add_data(wandb.Image(fig), true_str, pred_str)
        plt.close(fig)

    return val_table


def train_loop(cfg: TrainConfig):

    wandb.init(project=cfg.project_name, config=asdict(cfg))

    train_loader, val_loader, vocab_size, pad_idx = build_dataloaders(
        data_dir=cfg.data_dir, batch_size=cfg.batch_size
    )

    tokenizer = Tokenizer()

    model = Plot2EqModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_enc_layers=cfg.num_enc_layers,
        num_dec_layers=cfg.num_dec_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    model: nn.Module = torch.compile(model)  # type: ignore

    param_dict = {param_name: param for param_name, param in model.named_parameters()}

    decay = set()
    no_decay = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "gamma" in name:
            no_decay.add(name)
        else:
            decay.add(name)

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optim_groups, lr=cfg.lr, betas=(0.9, 0.95))

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx, label_smoothing=cfg.label_smoothing
    )

    start_epoch = 1
    best_val_loss = float("inf")
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    if cfg.resume_from_checkpoint and Path(cfg.resume_from_checkpoint).exists():
        print(f"Начинаем обучение с чекпоинта: {cfg.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location=cfg.device)

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(
            f"Возобновлено с эпохи {checkpoint['epoch']}. Начинаем следующую эпоху: {start_epoch}"
        )
    else:
        print("Начинаем обучение с нуля.")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]")

        for batch_idx, (points, tokens) in enumerate(pbar):
            points, tokens = points.to(cfg.device), tokens.to(cfg.device)

            tgt_input = tokens[:, :-1]
            tgt_expected = tokens[:, 1:].contiguous().view(-1)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(points, tgt_input)
                logits = logits.view(-1, vocab_size)

                loss = criterion(logits, tgt_expected)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_tokens = 0
        total_tokens = 0

        correct_sequences = 0
        total_sequences = 0
        val_table = None

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Val]")

            val_table = wandb.Table(
                columns=["Plot", "True Equation", "Predicted Equation"]
            )

            for batch_idx, (points, tokens) in enumerate(pbar_val):
                points, tokens = points.to(cfg.device), tokens.to(cfg.device)

                tgt_input = tokens[:, :-1]
                tgt_expected_seq = tokens[:, 1:].contiguous()

                with autocast("cuda", dtype=torch.bfloat16):
                    logits = model(points, tgt_input)
                    preds_seq = logits.argmax(dim=-1)
                    mask_seq = tgt_expected_seq != pad_idx
                    correct_sequences += (
                        ((preds_seq == tgt_expected_seq) | ~mask_seq)
                        .all(dim=-1)
                        .sum()
                        .item()
                    )
                    total_sequences += points.size(0)

                    logits_flat = logits.view(-1, vocab_size)
                    tgt_expected_flat = tgt_expected_seq.view(-1)

                    loss = criterion(logits_flat, tgt_expected_flat)

                val_loss += loss.item()

                preds_flat = preds_seq.view(-1)
                mask_flat = tgt_expected_flat != pad_idx
                correct_tokens += (
                    (preds_flat[mask_flat] == tgt_expected_flat[mask_flat]).sum().item()
                )
                total_tokens += mask_flat.sum().item()

                if batch_idx == 0:
                    val_table = create_val_predictions_table(
                        model=model,
                        points=points,
                        true_tokens=tokens,
                        tokenizer=tokenizer,
                    )

        avg_val_loss = val_loss / len(val_loader)
        val_tok_acc = correct_tokens / total_tokens
        val_seq_acc = correct_sequences / total_sequences

        print(
            f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Token Acc: {val_tok_acc:.4f} | "
            f"Seq Acc (Exact Match): {val_seq_acc:.4f}\n"
        )

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_tok_acc": val_tok_acc,
                "val_seq_acc": val_seq_acc,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "Predictions": val_table,
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),  # type: ignore
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, "checkpoints/best_model.pth")
