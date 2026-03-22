import gzip
import logging
import multiprocessing as mp
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.core.cache import clear_cache
from tqdm import tqdm

from src.expression import ExpressionGenerator, TimeoutException, time_limit
from src.tokenizer import InvalidExpressionError, Tokenizer, TokenizerError

BASE_DIR = Path(__file__).resolve().parent.parent
log_path = BASE_DIR / "data" / "data.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.ERROR)


warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class GenConfig:
    max_ops: int
    timeout: int
    steps: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def generate_points(expr, config):
    if expr.has(sp.zoo) or expr.has(sp.oo) or expr.has(sp.nan):
        return None

    height_y = config.max_y - config.min_y

    f = sp.lambdify(sp.Symbol("x"), expr, modules=["numpy", "scipy"])
    x_values = np.linspace(num=config.steps, start=config.min_x, stop=config.max_x)
    step_size = (config.max_x - config.min_x) / (config.steps - 1)
    noise = np.random.uniform(-0.3, 0.3, size=config.steps) * step_size
    x_values[1:-1] += noise[1:-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y_values = f(x_values)

            if np.isscalar(y_values) or np.ndim(y_values) == 0:
                y_values = np.full_like(x_values, y_values, dtype=float)

            if np.iscomplexobj(y_values):
                return None

            y_values = np.asarray(y_values, dtype=float)
        except Exception:
            return None

    mask = (
        np.isfinite(y_values) & (y_values <= config.max_y) & (y_values >= config.min_y)
    )

    if np.sum(mask) / config.steps <= 0.1:
        return None

    valid_indices = np.where(mask)[0]

    if len(valid_indices) < 2:
        return None

    consecutive_mask = np.diff(valid_indices) == 1
    if np.sum(consecutive_mask) < 2:
        return None

    valid_y = y_values[valid_indices]
    valid_dy = np.diff(valid_y)[consecutive_mask]

    if len(valid_dy) == 0:
        return None

    mean_change_rel = np.mean(np.abs(valid_dy)) / height_y
    if mean_change_rel > 0.05:
        return None

    max_jump_rel = np.percentile(np.abs(valid_dy), 98) / height_y
    if max_jump_rel > 0.15:
        return None

    total_variation = np.sum(np.abs(valid_dy))
    if total_variation > height_y * 4:
        return None

    sign_changes = np.sum(np.diff(np.sign(valid_dy)) != 0)
    if sign_changes > 20:
        return None

    return np.column_stack((x_values[mask], y_values[mask]))


def worker_task(args):
    config, seed = args

    random.seed(seed)
    np.random.seed(seed % (2**32))

    generator = ExpressionGenerator(config.max_ops, config.timeout)
    tokenizer = Tokenizer()

    attempts = 0
    while True:
        attempts += 1
        if attempts % 100000 == 0:
            clear_cache()
        try:
            skeleton, orig_expr, expr_instantiated = generator.generate_expr()

            try:
                token_seq = tokenizer.expr_to_token_seq(skeleton)
                if len(token_seq) > 128:
                    continue

            except InvalidExpressionError:
                continue
            except TokenizerError:
                logging.exception(f"Ошибка токенизации: {skeleton}")
                continue

            with time_limit(config.timeout):
                points = generate_points(expr_instantiated, config)

            if points is None:
                continue

            return str(skeleton), str(expr_instantiated), token_seq, points

        except TimeoutException:
            continue

        except Exception:
            continue


class DataGenerator:
    def __init__(
        self,
        max_ops: int,
        steps: int,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        timeout: int,
        output_dir: str | Path,
    ) -> None:
        self.config = GenConfig(
            max_ops=max_ops,
            timeout=timeout,
            steps=steps,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = Tokenizer()

    def save_chunk(self, data, chunk_id):
        filename = self.output_dir / f"chunk_{chunk_id}.pkl.gz"
        with gzip.open(filename, "wb") as f:
            pickle.dump(data, f)

    def generate_data(
        self, size: int, chunk_size: int = 5000, n_jobs: int | None = None
    ):
        if n_jobs is None:
            n_jobs = mp.cpu_count()

        existing_chunks = list(self.output_dir.glob("chunk_*.pkl.gz"))
        chunk_counter = 0
        if existing_chunks:
            max_id = -1
            for f in existing_chunks:
                try:
                    part = f.name.split("_")[1]
                    num = int(part.split(".")[0])
                    if num > max_id:
                        max_id = num
                except (IndexError, ValueError):
                    continue

            chunk_counter = max_id + 1
            print(
                f"Найдено {len(existing_chunks)} существующих файлов. Следующий чанк будет: {chunk_counter}"
            )

        print(f"Запуск генерации на {n_jobs} процессах...")
        start = time.time()

        buffer = []
        total_generated = 0

        skeleton_counts = {}
        MAX_IDENTICAL_SKELETONS = 1000

        def task_generator():
            seed_base = int.from_bytes(os.urandom(4), "little")
            i = 0
            while True:
                yield (self.config, (seed_base + i) % (2**32))
                i += 1

        with mp.Pool(processes=n_jobs, maxtasksperchild=5000) as pool:
            result_iter = pool.imap_unordered(worker_task, task_generator())

            with tqdm(total=size, desc="Генерация", unit="expr") as pbar:
                for result in result_iter:
                    skeleton_str, instantiated_str, token_seq, points = result

                    skeleton_hash = hash(skeleton_str)

                    if skeleton_counts.get(skeleton_hash, 0) >= MAX_IDENTICAL_SKELETONS:
                        continue
                    skeleton_counts[skeleton_hash] = (
                        skeleton_counts.get(skeleton_hash, 0) + 1
                    )

                    item = {
                        "expr_str": skeleton_str,
                        "expr_instantiated_str": instantiated_str,
                        "tokens": token_seq,
                        "points": points,
                    }
                    buffer.append(item)
                    total_generated += 1

                    pbar.update(1)

                    if len(buffer) >= chunk_size:
                        self.save_chunk(buffer, chunk_counter)
                        chunk_counter += 1
                        buffer = []
                        tqdm.write(f"Сохранен чанк {chunk_counter - 1}")

                    if total_generated >= size:
                        pool.terminate()
                        pool.join()
                        break

                if buffer:
                    self.save_chunk(buffer, chunk_counter)

        print(f"\nВсего сгенерировано: {total_generated}")
        print(f"Всего затрачено {time.time() - start:.2f} сек")


if __name__ == "__main__":
    print("\nГенерация трэйна...")
    train_gen = DataGenerator(
        max_ops=7,
        timeout=10,
        steps=500,
        min_x=-10,
        max_x=10,
        min_y=-10,
        max_y=10,
        output_dir=BASE_DIR / "data" / "train",
    )
    train_gen.generate_data(size=10**7, chunk_size=50000)

    val_gen = DataGenerator(
        max_ops=7,
        timeout=10,
        steps=500,
        min_x=-10,
        max_x=10,
        min_y=-10,
        max_y=10,
        output_dir=BASE_DIR / "data" / "val",
    )

    print("\nГенерация валидации...")
    val_gen.generate_data(size=10**5, chunk_size=50000)
