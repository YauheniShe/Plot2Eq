import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def process_and_save_chunk(filepath, output_dir, max_seq_len, min_mask_ratio):
    with gzip.open(filepath, "rb") as f:
        data = pickle.load(f)  # type: ignore

    x_list, y_list, meta_rows = [], [], []
    chunk_name = filepath.name.replace(".pkl.gz", "")

    for local_idx, item in enumerate(data):
        features = item["features"]
        mask = features[1]

        if mask.mean() < min_mask_ratio or len(item["tokens"]) > max_seq_len:
            continue

        padded_tokens = np.zeros(max_seq_len, dtype=np.int64)
        padded_tokens[: len(item["tokens"])] = item["tokens"]

        x_list.append(features)
        y_list.append(padded_tokens)

        meta_rows.append(
            {
                "chunk_name": chunk_name,
                "local_idx": local_idx,
                "expr_str": item["expr_str"],
                "expr_instantiated_str": item["expr_instantiated_str"],
            }
        )

    if x_list:
        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_list), dtype=torch.long)
        torch.save(
            {"points": x_tensor, "tokens": y_tensor}, output_dir / f"{chunk_name}.pt"
        )

    return meta_rows


def run_compilation(
    input_dir: Path, output_dir: Path, max_seq_len=128, min_mask_ratio=0.1, num_cores=4
):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("chunk_*.pkl.gz"))

    print("Компиляция PyTorch тензоров...")
    results = Parallel(n_jobs=num_cores)(
        delayed(process_and_save_chunk)(f, output_dir, max_seq_len, min_mask_ratio)
        for f in tqdm(files)
    )

    all_meta_rows = [row for meta_chunk in results for row in meta_chunk]  # type: ignore

    df_meta = pd.DataFrame(all_meta_rows)
    df_meta.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Готово! Сохранено графиков: {len(df_meta)}. Тензоры в {output_dir}")
