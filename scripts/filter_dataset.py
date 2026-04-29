import os

import pandas as pd
import torch
from tqdm import tqdm

INPUT_DIR = "data/raw_data"
OUTPUT_DIR = "data/clean_data"

MAX_EXTREMA = 7
MAX_BREAKS = 2
NEW_CHUNK_SIZE = 10000


def check_humanity(points_tensor):
    B = points_tensor.shape[0]
    keep_mask = torch.zeros(B, dtype=torch.bool)

    for i in range(B):
        y = points_tensor[i, 0]
        m = points_tensor[i, 1].bool()

        transitions = torch.abs(m[1:].int() - m[:-1].int()).sum().item()
        num_breaks = (transitions + 1) // 2

        if num_breaks > MAX_BREAKS:
            continue

        valid_transitions = m[:-1] & m[1:]
        dy = y[1:] - y[:-1]
        dy = dy[valid_transitions]
        dy = dy[dy != 0]

        if len(dy) > 1:
            sign_changes = (dy[:-1] * dy[1:] < 0).sum().item()
        else:
            sign_changes = 0

        if sign_changes > MAX_EXTREMA:
            continue

        keep_mask[i] = True

    return keep_mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Загрузка старой метадаты...")
    df = pd.read_csv(os.path.join(INPUT_DIR, "metadata.csv"))

    chunk_names = df["chunk_name"].unique()
    chunk_names = sorted(chunk_names, key=lambda x: int(x.split("_")[1]))

    points_buffer, tokens_buffer, meta_buffer = [], [], []
    new_metadata_rows = []

    new_chunk_idx = 0
    total_kept = 0
    total_processed = 0

    print(
        f"Начинаем очистку. Пороги: Макс. изгибов = {MAX_EXTREMA}, Макс. разрывов = {MAX_BREAKS}"
    )

    for c_name in tqdm(chunk_names, desc="Обработка чанков"):
        chunk_path = os.path.join(INPUT_DIR, f"{c_name}.pt")
        if not os.path.exists(chunk_path):
            continue

        data = torch.load(chunk_path, map_location="cpu")
        raw_points = data["points"]
        raw_tokens = data["tokens"]

        chunk_df = df[df["chunk_name"] == c_name].sort_values("local_idx")
        chunk_df = chunk_df.iloc[: raw_points.shape[0]].copy()

        keep_mask = check_humanity(raw_points)

        total_processed += raw_points.shape[0]
        total_kept += keep_mask.sum().item()

        valid_points = raw_points[keep_mask]
        valid_tokens = raw_tokens[keep_mask]
        valid_df = chunk_df[keep_mask.numpy()].copy()

        if len(valid_points) > 0:
            points_buffer.append(valid_points)
            tokens_buffer.append(valid_tokens)
            meta_buffer.append(valid_df)

        current_size = sum([p.shape[0] for p in points_buffer])

        while current_size >= NEW_CHUNK_SIZE:
            all_points = torch.cat(points_buffer, dim=0)
            all_tokens = torch.cat(tokens_buffer, dim=0)
            all_meta = pd.concat(meta_buffer, ignore_index=True)

            chunk_points = all_points[:NEW_CHUNK_SIZE]
            chunk_tokens = all_tokens[:NEW_CHUNK_SIZE]
            chunk_meta = all_meta.iloc[:NEW_CHUNK_SIZE].copy()

            new_c_name = f"chunk_{new_chunk_idx}"
            torch.save(
                {"points": chunk_points, "tokens": chunk_tokens},
                os.path.join(OUTPUT_DIR, f"{new_c_name}.pt"),
            )

            chunk_meta["chunk_name"] = new_c_name
            chunk_meta["local_idx"] = range(NEW_CHUNK_SIZE)
            new_metadata_rows.append(chunk_meta)

            points_buffer = [all_points[NEW_CHUNK_SIZE:]]
            tokens_buffer = [all_tokens[NEW_CHUNK_SIZE:]]
            meta_buffer = [all_meta.iloc[NEW_CHUNK_SIZE:]]

            current_size = points_buffer[0].shape[0]
            new_chunk_idx += 1

    current_size = sum([p.shape[0] for p in points_buffer]) if points_buffer else 0
    if current_size > 0:
        all_points = torch.cat(points_buffer, dim=0)
        all_tokens = torch.cat(tokens_buffer, dim=0)
        all_meta = pd.concat(meta_buffer, ignore_index=True)

        new_c_name = f"chunk_{new_chunk_idx}"
        torch.save(
            {"points": all_points, "tokens": all_tokens},
            os.path.join(OUTPUT_DIR, f"{new_c_name}.pt"),
        )

        all_meta["chunk_name"] = new_c_name
        all_meta["local_idx"] = range(current_size)
        new_metadata_rows.append(all_meta)

    print("\nСохранение нового metadata.csv...")
    if new_metadata_rows:
        new_full_df = pd.concat(new_metadata_rows, ignore_index=True)
        new_full_df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    print("ГОТОВО!")
    if total_processed > 0:
        print(f"Обработано: {total_processed:,} примеров.")
        print(
            f"Осталось (прошло фильтр): {total_kept:,} примеров ({total_kept / total_processed * 100:.1f}%)."
        )
        print(f"Новые файлы лежат в: {OUTPUT_DIR}")
    else:
        print("Внимание: Ни один пример не был обработан.")


if __name__ == "__main__":
    main()
