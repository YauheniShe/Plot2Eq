import gzip
import pickle
from pathlib import Path

import sympy as sp
from sympy.core.cache import clear_cache
from tqdm import tqdm

from plot2eq.core.tokenizer import InvalidExpressionError, Tokenizer


def get_c_node():
    return sp.Symbol("C", real=True)


def has_top_level_shift(expr: sp.Expr) -> bool:
    if expr.is_Symbol and str(expr).startswith("C"):
        return True
    if expr.is_Add:
        for arg in expr.args:
            if has_top_level_shift(arg):  # type: ignore
                return True
    return False


def replace_cos_unevaluated(expr):
    if not expr.args:
        return expr
    new_args = [replace_cos_unevaluated(arg) for arg in expr.args]
    current_func = expr.func
    new_func = sp.sin if current_func == sp.cos else current_func
    return new_func(*new_args, evaluate=False)


def clean_item(item, tokenizer: Tokenizer):
    try:
        tokens = item["tokens"]
        expr = tokenizer.token_seq_to_expr(tokens)

        if expr.has(sp.sin) and expr.has(sp.cos):
            return None

        processed_expr = replace_cos_unevaluated(expr) if expr.has(sp.cos) else expr

        if not has_top_level_shift(processed_expr):
            processed_expr = sp.Add(processed_expr, get_c_node(), evaluate=False)

        clear_expr = tokenizer.canonicalize_tree_structure(processed_expr)
        new_tokens = tokenizer.expr_to_token_seq(clear_expr)

        return {
            "expr_str": str(clear_expr),
            "expr_instantiated_str": item["expr_instantiated_str"],
            "tokens": new_tokens,
            "points": item["points"],
        }
    except InvalidExpressionError:
        return None


def run_canonicalization(input_dir: Path, output_dir: Path, chunk_size: int = 50000):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer()
    input_files = sorted(input_dir.glob("chunk_*.pkl.gz"))

    if not input_files:
        print("В папке не найдено чанков для обработки.")
        return

    cleaned_buffer = []
    new_chunk_index = 0
    total_processed, total_skipped = 0, 0
    operations_count = 0

    print(f"Найдено {len(input_files)} чанков. Начинаю канонизацию формул...")

    for filepath in tqdm(input_files, desc="Канонизация"):
        with gzip.open(filepath, "rb") as f:
            data = pickle.load(f)

        for item in data:
            operations_count += 1
            if operations_count % 10000 == 0:
                clear_cache()

            cleaned = clean_item(item, tokenizer)
            if cleaned:
                cleaned_buffer.append(cleaned)
            else:
                total_skipped += 1

            if len(cleaned_buffer) >= chunk_size:
                savepath = output_dir / f"chunk_{new_chunk_index}.pkl.gz"
                with gzip.open(savepath, "wb") as f_out:
                    pickle.dump(cleaned_buffer[:chunk_size], f_out)

                total_processed += chunk_size
                cleaned_buffer = cleaned_buffer[chunk_size:]
                new_chunk_index += 1

    if cleaned_buffer:
        savepath = output_dir / f"chunk_{new_chunk_index}.pkl.gz"
        with gzip.open(savepath, "wb") as f_out:
            pickle.dump(cleaned_buffer, f_out)
        total_processed += len(cleaned_buffer)

    print(f"Успешно очищено: {total_processed}. Пропущено ошибок: {total_skipped}.")
