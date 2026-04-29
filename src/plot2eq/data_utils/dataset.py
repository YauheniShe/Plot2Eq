import io
import math
import tarfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from plot2eq.core.tokenizer import Tokenizer
from plot2eq.data.augmentation import HandDrawnAugmentation


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
        self.max_drift_scale = max_drift_scale
        self.max_wobble_scale = max_wobble_scale
        self.p = p

        self.transform = HandDrawnAugmentation(max_drift_scale, max_wobble_scale, p)

        self.tokenizer = Tokenizer()

        self.data_dir = Path(data_dir)
        self.map_location = map_location

        if map_location is None:
            self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        elif not map_location.startswith(("cuda", "cpu")):
            raise ValueError(f"Invalid map_location: {map_location}!")

        self.points = []
        self.tokens = []
        self.file_names = []

        if self.data_dir.is_dir():
            self.file_names = sorted(
                [f.name for f in self.data_dir.iterdir() if f.name.endswith(".pt")]
            )
            if not self.file_names:
                raise RuntimeError(f"Не найдено ни одного .pt файла в папке {data_dir}")

            print(f"Загружаем {len(self.file_names)} файлов в память (из директории)")
            for f in tqdm(self.file_names):
                file_path = self.data_dir / f
                data = torch.load(
                    file_path, map_location=self.map_location, weights_only=True
                )
                self.points.append(data["points"])
                self.tokens.append(data["tokens"])

        elif self.data_dir.is_file() and self.data_dir.suffix.lower() == ".zip":
            with zipfile.ZipFile(self.data_dir, "r") as z:
                self.file_names = sorted([f for f in z.namelist() if f.endswith(".pt")])
                if not self.file_names:
                    raise RuntimeError(
                        f"Не найдено ни одного .pt файла в архиве {data_dir}"
                    )

                print(
                    f"Загружаем {len(self.file_names)} файлов в память (из ZIP-архива)"
                )
                for f_name in tqdm(self.file_names):
                    with z.open(f_name) as f:
                        buffer = io.BytesIO(f.read())
                        data = torch.load(
                            buffer, map_location=self.map_location, weights_only=True
                        )
                        self.points.append(data["points"])
                        self.tokens.append(data["tokens"])

        elif self.data_dir.is_file() and self.data_dir.name.endswith(
            (".tar", ".tar.gz", ".tgz")
        ):
            mode = "r:gz" if self.data_dir.name.endswith(("gz", "tgz")) else "r"
            with tarfile.open(self.data_dir, mode) as tar:
                members = [m for m in tar.getmembers() if m.name.endswith(".pt")]
                members.sort(key=lambda m: m.name)
                self.file_names = [m.name for m in members]

                if not self.file_names:
                    raise RuntimeError(
                        f"Не найдено ни одного .pt файла в архиве {data_dir}"
                    )

                print(
                    f"Загружаем {len(self.file_names)} файлов в память (из TAR-архива)"
                )
                for member in tqdm(members):
                    f = tar.extractfile(member)
                    if f is not None:
                        buffer = io.BytesIO(f.read())
                        data = torch.load(
                            buffer, map_location=self.map_location, weights_only=True
                        )
                        self.points.append(data["points"])
                        self.tokens.append(data["tokens"])
        else:
            raise ValueError(
                f"Путь {self.data_dir} не существует, не является папкой или поддерживаемым архивом!"
            )

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
            latex_expr = sp.latex(expr)
            title_str = f"${latex_expr}$"
        except Exception as e:
            title_str = f"Ошибка декодирования: {e}"

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

        plt.title(f"{title_prefix}\n{title_str}", fontsize=16, pad=15)
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
