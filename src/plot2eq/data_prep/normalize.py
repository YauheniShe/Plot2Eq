import gzip
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_opening
from tqdm import tqdm


def raw_to_normalized_features(
    points, num_points=256, jump_threshold=0.3, structure_size=3
):
    # 1. Первичная нормализация и маска (Bounding Box)
    x_raw, y_raw = points[:, 0], points[:, 1]
    x_min, x_max = np.min(x_raw), np.max(x_raw)
    y_min, y_max = np.min(y_raw), np.max(y_raw)

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    x_norm = (x_raw - x_min) / x_range
    y_norm = np.full_like(y_raw, 0.5) if y_max == y_min else (y_raw - y_min) / y_range

    dx = np.diff(x_norm)
    gap_indices = np.where(dx > 3 * np.median(dx))[0] if len(dx) > 0 else []

    valid_intervals = []
    start_idx = 0
    for gap_idx in gap_indices:
        valid_intervals.append((x_norm[start_idx], x_norm[gap_idx]))
        start_idx = gap_idx + 1
    valid_intervals.append((x_norm[start_idx], x_norm[-1]))

    x_fixed = np.linspace(0, 1, num_points)
    sort_idx = np.argsort(x_norm)
    y_fixed = np.interp(x_fixed, x_norm[sort_idx], y_norm[sort_idx])

    mask = np.zeros_like(x_fixed, dtype=np.float32)
    for start, end in valid_intervals:
        mask[(x_fixed >= start) & (x_fixed <= end)] = 1.0

    # 2. Удаление ложных асимптот по оси Y
    valid_y_orig = y_fixed[mask == 1]
    if len(valid_y_orig) < 2:
        return None

    y_range_mask = np.nanmax(valid_y_orig) - np.nanmin(valid_y_orig)
    if y_range_mask > 1e-5:
        y_diff = np.abs(np.diff(y_fixed))
        asymptote_indices = np.where(
            (y_diff > y_range_mask * jump_threshold)
            & (mask[:-1] == 1)
            & (mask[1:] == 1)
        )[0]
        for idx in asymptote_indices:
            mask[idx] = 0

    # 3. Удаление ошметков (< 3 пикселей)
    structure = np.ones(structure_size)
    cleaned_mask = binary_opening(mask, structure=structure).astype(int)  # type: ignore
    if np.sum(cleaned_mask) < 5:
        return None

    # 4. Обрезка BBox по X и финальная интерполяция
    valid_indices = np.where(cleaned_mask == 1)[0]
    i_min, i_max = valid_indices[0], valid_indices[-1]
    if i_max == i_min:
        return None

    x_orig = np.linspace(0, 1, len(y_fixed))
    x_min_bbox, x_max_bbox = x_orig[i_min], x_orig[i_max]
    x_new = np.linspace(x_min_bbox, x_max_bbox, num_points)

    y_interp_func = interp1d(
        x_orig, y_fixed, kind="linear", bounds_error=False, fill_value=np.nan
    )
    y_resampled = y_interp_func(x_new)

    mask_interp_func = interp1d(
        x_orig, cleaned_mask, kind="nearest", bounds_error=False, fill_value=0
    )
    mask_resampled = mask_interp_func(x_new).astype(int)

    # 5. Финальный Min-Max Scaling по Y
    valid_y_resampled = y_resampled[mask_resampled == 1]
    if len(valid_y_resampled) == 0 or np.isnan(valid_y_resampled).all():
        return None

    y_min_val, y_max_val = np.nanmin(valid_y_resampled), np.nanmax(valid_y_resampled)
    diff = y_max_val - y_min_val

    y_scaled = (
        np.full_like(y_resampled, 0.5)
        if diff < 1e-6
        else (y_resampled - y_min_val) / diff
    )
    y_scaled[mask_resampled == 0] = 0.0

    return np.vstack((y_scaled.astype(np.float32), mask_resampled.astype(np.float32)))


def run_normalization(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(input_dir.glob("chunk_*.pkl.gz"))

    total_processed = 0
    for filepath in tqdm(input_files, desc="Нормализация и чистка графиков"):
        with gzip.open(filepath, "rb") as f:
            data = pickle.load(f)

        new_data = []
        for item in data:
            features = raw_to_normalized_features(item["points"])
            if features is not None:
                new_data.append(
                    {
                        "expr_str": item["expr_str"],
                        "expr_instantiated_str": item["expr_instantiated_str"],
                        "tokens": item["tokens"],
                        "features": features,
                    }
                )

        if new_data:
            with gzip.open(output_dir / filepath.name, "wb") as f_out:
                pickle.dump(new_data, f_out)
            total_processed += len(new_data)

    print(f"Идеальных графиков сохранено: {total_processed}")
