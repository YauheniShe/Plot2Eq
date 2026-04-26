from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .tokenizer import Tokenizer


class HandDrawnAugmentation:
    def __init__(self, max_drift_scale=0.005, max_wobble_scale=0.015, p=0.8):
        """
        :param max_drift_scale: Максимальное плавное искажение
        :param max_wobble_scale: Максимальные мелкие неровности
        :param p: Вероятность применения аугментации (иногда полезно видеть идеальные данные)
        """
        self.max_drift_scale = max_drift_scale
        self.max_wobble_scale = max_wobble_scale
        self.p = p

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return points

        y = points[0:1, :].clone()
        mask = points[1:2, :].clone()
        seq_len = y.size(1)
        device = y.device

        min_strength = 0.5
        drift_strength = (
            torch.empty(1, device=device).uniform_(min_strength, 1.0).item()
        )
        wobble_strength = (
            torch.empty(1, device=device).uniform_(min_strength, 1.0).item()
        )

        current_drift = self.max_drift_scale * drift_strength
        current_wobble = self.max_wobble_scale * wobble_strength

        drift_freq = torch.randint(2, 6, (1,)).item()
        wobble_freq = torch.randint(10, 30, (1,)).item()

        low_freq_noise = torch.randn(1, 1, 1, drift_freq, device=device)  # type: ignore
        drift = F.interpolate(
            low_freq_noise, size=(1, seq_len), mode="bicubic", align_corners=True
        ).view(1, seq_len)

        mid_freq_noise = torch.randn(1, 1, 1, wobble_freq, device=device)  # type: ignore
        wobble = F.interpolate(
            mid_freq_noise, size=(1, seq_len), mode="bicubic", align_corners=True
        ).view(1, seq_len)

        total_distortion = (drift * current_drift) + (wobble * current_wobble)
        y = y + total_distortion * mask

        x_freq = torch.randint(5, 15, (1,)).item()
        x_noise = torch.randn(1, 1, 1, x_freq, device=device)  # type: ignore
        x_drift = F.interpolate(
            x_noise, size=(1, seq_len), mode="bicubic", align_corners=True
        ).view(seq_len)

        envelope = F.interpolate(
            torch.rand(1, 1, 1, 3, device=device),
            size=(1, seq_len),
            mode="bicubic",
            align_corners=True,
        ).view(seq_len)
        x_drift = x_drift * envelope

        x_shift_amplitude = torch.rand(1, device=device).item() * 3.0 + 1.0
        new_indices = (
            torch.arange(seq_len, device=device).float() + x_drift * x_shift_amplitude
        )
        new_indices = torch.clamp(new_indices, 0, seq_len - 1)

        idx_floor = new_indices.long()
        idx_ceil = torch.clamp(idx_floor + 1, 0, seq_len - 1)
        weight = new_indices - idx_floor.float()

        mask_floor = mask[:, idx_floor]
        mask_ceil = mask[:, idx_ceil]

        valid_transition = mask_floor * mask_ceil

        idx_nearest = torch.round(new_indices).long()
        y_nearest = y[:, idx_nearest]

        y_interp = y[:, idx_floor] * (1.0 - weight) + y[:, idx_ceil] * weight

        y = torch.where(valid_transition == 1.0, y_interp, y_nearest)

        mask = mask[:, idx_nearest]

        if torch.rand(1).item() > 0.5:
            shear_factor = (torch.rand(1, device=device) - 0.5) * 0.1
            tiling = torch.linspace(-1, 1, seq_len, device=device) * shear_factor
            y = y + tiling * mask

        y_valid_mask = mask.bool().squeeze(0)

        if y_valid_mask.any():
            y_valid_points = y[0, y_valid_mask]
            min_val = torch.min(y_valid_points)
            max_val = torch.max(y_valid_points)

            if max_val > min_val:
                y[0, y_valid_mask] = (y[0, y_valid_mask] - min_val) / (
                    max_val - min_val
                )
            else:
                y[0, y_valid_mask] = 0.5

        return torch.cat((y, mask), dim=0)


class SymbolicDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        map_location: str | None = None,
        drawn_augmentation: bool = False,
        max_drift_scale=0.005,
        max_wobble_scale=0.015,
        p=0.8,
    ) -> None:

        self.drawn_augmentation = drawn_augmentation
        self.max_drift_scale = 0.005
        self.max_wobble_scale = 0.015
        self.p = 0.8

        self.transform = HandDrawnAugmentation(max_drift_scale, max_wobble_scale, p)

        self.tokenizer = Tokenizer()

        self.data_dir = Path(data_dir)
        self.file_names = sorted(
            [f.name for f in self.data_dir.iterdir() if f.name.endswith(".pt")]
        )

        self.map_location = map_location

        if map_location is None:
            self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        elif map_location not in ("cuda", "cpu"):
            raise ValueError(f"Invalid map_location: {map_location}!")

        self.points = []
        self.tokens = []

        if not self.file_names:
            raise RuntimeError(f"Не найдено ни одного .pt файла в папке {data_dir}")

        print(f"Загружаем {len(self.file_names)} файлов в память")

        for f in tqdm(self.file_names):
            file_path = self.data_dir / f
            data = torch.load(
                file_path, map_location=self.map_location, weights_only=False
            )
            self.points.append(data["points"])
            self.tokens.append(data["tokens"])

        self.points = torch.cat(self.points, dim=0)
        self.tokens = torch.cat(self.tokens, dim=0)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        points, tokens = self.points[index], self.tokens[index]

        if self.drawn_augmentation:
            points = self.transform(points)

        return points, tokens

    def visualize(
        self, index: int, apply_transform: bool = False, title_prefix: str = ""
    ):

        points = self.points[index]
        tokens = self.tokens[index]

        if apply_transform and self.transform is not None:
            points = self.transform(points)

        y_vals = points[0].cpu().numpy()
        mask = points[1].cpu().numpy()

        x_vals = np.linspace(0, 1, len(y_vals))

        clean_tokens = [t.item() for t in tokens if t.item() not in (0, 1, 2)]
        try:
            expr = self.tokenizer.token_seq_to_expr(torch.tensor(clean_tokens))
            expr_str = str(expr)
        except Exception as e:
            expr_str = f"Ошибка декодирования: {e}"

        plt.figure(figsize=(8, 5))

        plt.fill_between(
            x_vals,
            -0.05,
            1.05,
            where=(mask == 0.0),
            color="red",
            alpha=0.1,
            label="Маска = 0 (Разрыв/Асимптота)",
            transform=plt.gca().get_xaxis_transform(),
        )

        y_plot = y_vals.copy()
        y_plot[mask == 0.0] = np.nan

        plt.plot(
            x_vals,
            y_plot,
            linewidth=2.5,
            color="blue",
            alpha=0.8,
            label="Линия графика",
        )

        plt.title(f"{title_prefix}Скелет: {expr_str}", fontsize=14, pad=15)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def visualize_batch(self, batch_size=9, indices=None, apply_transform=True):
        """
        Отрисовывает сетку из batch_size графиков.
        :param batch_size: Количество графиков
        :param indices: Конкретные индексы для отрисовки (опционально)
        :param apply_transform: Применять ли аугментацию
        """
        import math

        import matplotlib.pyplot as plt

        if indices is None:
            indices = np.random.choice(len(self.points), size=batch_size, replace=False)
        else:
            batch_size = len(indices)

        grid_cols = math.ceil(math.sqrt(batch_size))
        grid_rows = math.ceil(batch_size / grid_cols)

        fig, axes = plt.subplots(
            grid_rows, grid_cols, figsize=(5 * grid_cols, 4 * grid_rows)
        )

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, ax in zip(indices, axes):
            points = self.points[idx]
            tokens = self.tokens[idx]

            if apply_transform and self.transform is not None:
                points = self.transform(points)

            y_vals = points[0].cpu().numpy()
            mask = points[1].cpu().numpy()
            x_vals = np.linspace(0, 1, len(y_vals))

            clean_tokens = [t.item() for t in tokens if t.item() not in (0, 1, 2)]
            try:
                expr = self.tokenizer.token_seq_to_expr(torch.tensor(clean_tokens))
                expr_str = str(expr)
            except Exception as e:
                expr_str = f"Ошибка: {e}"

            ax.fill_between(
                x_vals,
                -0.05,
                1.05,
                where=(mask == 0.0),
                color="red",
                alpha=0.1,
                transform=ax.get_xaxis_transform(),
            )

            y_plot = y_vals.copy()
            y_plot[mask == 0.0] = np.nan

            ax.plot(x_vals, y_plot, linewidth=2.5, color="blue", alpha=0.8)
            ax.set_title(f"Idx: {idx}\n{expr_str}", fontsize=10, pad=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", alpha=0.6)

        for i in range(batch_size, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
