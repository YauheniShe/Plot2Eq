from dataclasses import dataclass

import torch


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
    max_seq_len: int = 128
    label_smoothing: float = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
