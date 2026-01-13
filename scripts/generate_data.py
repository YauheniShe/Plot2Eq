import numpy as np
import sympy as sp
import warnings
import time
import logging
import multiprocessing as mp
import pickle
import gzip

from pathlib import Path
from dataclasses import dataclass

from src.expression import ExpressionGenerator
from src.tokenizer import Tokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
log_path = BASE_DIR / "data" / "data.log"
logging.basicConfig(filename=log_path, level=logging.ERROR)

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class GenConfig:
    max_depth: int
    steps: int
    const_prob: float
    leaf_prob: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float

def generate_points(expr, config: GenConfig):
    if expr.has(sp.zoo) or expr.has(sp.oo) or expr.has(sp.nan):
        return None
    f = sp.lambdify(sp.Symbol('x'), expr, modules='numpy')
    x_values = np.linspace(num=config.steps, start=config.min_x, stop = config.max_x)
    try:
        y_values = f(x_values)
        if np.isscalar(y_values) or np.ndim(y_values) == 0:
            y_values = np.full_like(x_values, y_values)
        if np.iscomplexobj(y_values):
            return None
    except:
        return None
    mask = np.isfinite(y_values) & (y_values < config.max_y) & (y_values > config.min_y)
    if np.sum(mask) / np.size(mask) <= 0.3:
        return None
    dy = np.diff(y_values)
    diff_mask = mask[:-1] & mask[1:]
    if np.sum(diff_mask) < 2:
        return None
    valid_dy = dy[diff_mask]
    mean_change = np.mean(np.abs(valid_dy))
    if mean_change > 0.5:
        return None
    sign_changes = np.sum(np.diff(np.sign(valid_dy)) != 0)
    if sign_changes > len(valid_dy) * 0.3:
        return None
    y_values[~mask] = 0.0
    return np.vstack((x_values, y_values, mask.astype(float))).T
    
def worker_task(config: GenConfig):
    generator = ExpressionGenerator(config.max_depth, config.const_prob, config.leaf_prob)
    tokenizer = Tokenizer()
    
    while True:
        try:
            expr = generator.generate_expr()
            try:
                token_seq = tokenizer.expr_to_token_seq(expr)
            except RuntimeError:
                continue
            except Exception:
                logging.exception(expr)
                continue

            points = generate_points(expr, config)
            if points is None:
                continue

            return str(expr), token_seq, points
            
        except Exception:
            continue

class DataGenerator:
    def __init__(self, max_depth: int, steps: int, const_prob: float, leaf_prob: float,
                 min_x: float, max_x: float, min_y: float, max_y: float, output_dir: str | Path) -> None:
        self.config = GenConfig(
            max_depth=max_depth,
            steps=steps,
            const_prob=const_prob,
            leaf_prob=leaf_prob,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = ExpressionGenerator(max_depth, const_prob, leaf_prob)
        self.tokenizer = Tokenizer()

    def save_chunk(self, data, chunk_id):
        filename = self.output_dir / f"chunk_{chunk_id}.pkl.gz"
        print(f"Сохранение чанка {chunk_id} ({len(data)} примеров) в {filename}...")
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)

    def generate_data(self, size: int, chunk_size: int = 5000, n_jobs: int | None = None):
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        
        print(f"Запуск генерации на {n_jobs} процессах...")
        start = time.time()

        buffer = []
        chunk_counter = 0
        exprs_hashes = set()
        total_generated = 0

        with mp.Pool(processes=n_jobs) as pool:
            tasks = [self.config] * (size * 10)
            
            result_iter = pool.imap_unordered(worker_task, tasks)
            
            for result in result_iter:
                expr_str, token_seq, points = result
                if expr_str in exprs_hashes:
                    continue
                exprs_hashes.add(expr_str)
                item = {
                    'expr_str': expr_str,
                    'tokens': token_seq,
                    'points': points
                }
                buffer.append(item)
                total_generated += 1

                if len(buffer) >= chunk_size:
                    self.save_chunk(buffer, chunk_counter)
                    chunk_counter += 1
                    buffer = []
                    print(f"Прогресс: {total_generated}/{size}")

                if total_generated >= size:
                    pool.terminate()
                    break

            if buffer:
                self.save_chunk(buffer, chunk_counter)

        print(f"\nВсего сгенерировано: {total_generated}")
        print(f"Всего затрачено {time.time() - start:.2f} сек")


def get_raw_polish_notation(expr):
    tokens = []
    for node in sp.preorder_traversal(expr):
        if node.args:
            tokens.append(node.func.__name__)
        else:
            tokens.append(str(node))
            
    return "[" + " ".join(tokens) + "]"

if __name__ == '__main__':
    print("\nГенерация трэйна...")
    train_gen = DataGenerator(
        max_depth=6,
        steps=500,
        const_prob=0.1, 
        leaf_prob=0.2,
        min_x=-10, max_x=10, 
        min_y=-10, max_y=10,
        output_dir= BASE_DIR / "data" / "train"
    )
    train_gen.generate_data(size=1000000, chunk_size=5000)

    val_gen = DataGenerator(
        max_depth=6, 
        steps=500, 
        const_prob=0.1, 
        leaf_prob=0.2,
        min_x=-10, max_x=10, 
        min_y=-10, max_y=10,
        output_dir= BASE_DIR / "data" / "val"
    )
    
    print("\nГенерация валидации...")
    val_gen.generate_data(size=50000, chunk_size=5000)