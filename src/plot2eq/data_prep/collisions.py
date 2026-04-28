import gzip
import pickle
import zlib
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def load_features_chunk(filepath):
    with gzip.open(filepath, "rb") as f:
        data = pickle.load(f)  # type: ignore

    features_list, metadata_list = [], []
    for i, item in enumerate(data):
        features_list.append(item["features"].flatten())
        skel_hash = zlib.crc32(item["expr_str"].encode("utf-8"))
        metadata_list.append(
            (
                filepath.name,
                i,
                len(item["tokens"]),
                skel_hash,
                item["expr_str"],
                item["expr_instantiated_str"],
            )
        )

    return np.array(features_list, dtype=np.float32), metadata_list


def run_collision_removal(
    input_dir: Path, output_dir: Path, ckpt_path: Path, num_cores=4
):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("chunk_*.pkl.gz"))

    print("Загрузка данных для поиска коллизий...")
    results = Parallel(n_jobs=num_cores)(
        delayed(load_features_chunk)(f) for f in tqdm(files)
    )

    all_features, all_metadata = [], []
    for feat, meta in results:  # type: ignore
        all_features.append(feat)
        all_metadata.extend(meta)

    features_matrix = np.vstack(all_features)
    N = len(features_matrix)

    print("Сортировка по Бритве Оккама...")
    sorted_indices = sorted(
        range(N), key=lambda i: (all_metadata[i][2], all_metadata[i][4])
    )
    features_matrix = features_matrix[sorted_indices]
    all_metadata = [all_metadata[i] for i in sorted_indices]

    hash_array = np.array([m[3] for m in all_metadata], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(features_matrix, device=device)
    Hash_tensor = torch.tensor(hash_array, device=device)

    active_mask = torch.ones(N, dtype=torch.bool, device=device)

    X_means = X_tensor.mean(dim=1)
    ALL_INDICES = torch.arange(N, device=device)

    Q_BATCH, T_BATCH, CAND_BATCH = 1024, 65536, 10000
    MAE_THRESHOLD, MAX_ERROR_THRESHOLD, MASK_MAE_TOLERANCE = 0.03, 0.03, 0.012

    for start_idx in tqdm(range(0, N, Q_BATCH), desc="GPU Очистка коллизий"):
        end_idx = min(start_idx + Q_BATCH, N)
        q_indices = ALL_INDICES[start_idx:end_idx]
        q_indices = q_indices[active_mask[start_idx:end_idx]]

        if q_indices.numel() == 0:
            continue

        min_q = q_indices[0].item()
        target_pool = ALL_INDICES[min_q + 1 :]
        target_pool = target_pool[active_mask[min_q + 1 :]]

        if target_pool.numel() == 0:
            continue

        Hash_q, Mean_q = Hash_tensor[q_indices], X_means[q_indices]

        for t_chunk_start in range(0, target_pool.numel(), T_BATCH):
            t_chunk_indices = target_pool[t_chunk_start : t_chunk_start + T_BATCH]
            Hash_t, Mean_t = Hash_tensor[t_chunk_indices], X_means[t_chunk_indices]

            mean_diffs = torch.abs(Mean_q.unsqueeze(1) - Mean_t.unsqueeze(0))
            diff_skel_mask = Hash_q.unsqueeze(1) != Hash_t.unsqueeze(0)

            candidate_mask = (mean_diffs <= MAE_THRESHOLD) & diff_skel_mask
            rows, cols = candidate_mask.nonzero(as_tuple=True)

            if rows.numel() > 0:
                cand_global_q, cand_global_t = q_indices[rows], t_chunk_indices[cols]

                for c_start in range(0, cand_global_q.numel(), CAND_BATCH):
                    c_end = c_start + CAND_BATCH
                    q_sub, t_sub = (
                        cand_global_q[c_start:c_end],
                        cand_global_t[c_start:c_end],
                    )

                    active_sub_mask = active_mask[q_sub] & active_mask[t_sub]
                    q_sub, t_sub = q_sub[active_sub_mask], t_sub[active_sub_mask]

                    if q_sub.numel() == 0:
                        continue

                    X_q_cand, X_t_cand = X_tensor[q_sub], X_tensor[t_sub]
                    y_q, mask_q = X_q_cand[:, :256], X_q_cand[:, 256:]
                    y_t, mask_t = X_t_cand[:, :256], X_t_cand[:, 256:]

                    mask_mae = torch.abs(mask_q - mask_t).mean(dim=1)
                    y_diffs = torch.abs(y_q - y_t)
                    y_mae, y_max_error = y_diffs.mean(dim=1), y_diffs.max(dim=1).values

                    hits = (
                        (mask_mae <= MASK_MAE_TOLERANCE)
                        & (y_mae <= MAE_THRESHOLD)
                        & (y_max_error <= MAX_ERROR_THRESHOLD)
                    )
                    targets_to_delete = t_sub[hits]

                    if targets_to_delete.numel() > 0:
                        active_mask[targets_to_delete] = False

    indices_to_keep = active_mask.cpu().nonzero(as_tuple=True)[0].numpy()
    print(f"Коллизий удалено: {N - len(indices_to_keep)}. Сохраняем результат...")

    # Сохранение (воссоздаем чанки только с выжившими индексами)
    from collections import defaultdict

    keep_dict = defaultdict(set)
    for global_idx in indices_to_keep:
        file_name, local_idx = all_metadata[global_idx][0], all_metadata[global_idx][1]
        keep_dict[file_name].add(local_idx)

    for d in tqdm(files, desc="Запись чанков"):
        if d.name not in keep_dict:
            continue
        with gzip.open(d, "rb") as file:
            chunk = pickle.load(file)

        clean_chunk = [item for i, item in enumerate(chunk) if i in keep_dict[d.name]]
        with gzip.open(output_dir / d.name, "wb") as out_file:
            pickle.dump(clean_chunk, out_file)
