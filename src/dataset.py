from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SymbolicDataset(Dataset):
    def __init__(self, data_dir: Path | str, map_location: str | None = None) -> None:
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
        return self.points[index], self.tokens[index]
