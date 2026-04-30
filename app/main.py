import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager

from scipy.signal import savgol_filter
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cpu_count = os.cpu_count() or 1
os.environ["TORCH_NUM_THREADS"] = str(cpu_count)

# flake8: noqa: E402
# 1. Бинаризация по альфа-каналу
import numpy as np
import sympy as sp
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from plot2eq.config import TrainConfig
from plot2eq.core.tokenizer import Tokenizer
from plot2eq.inference.pipeline import predict_top_k_equations
from plot2eq.models.core_model import Plot2EqModel

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Загрузка модели...")
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)

    tokenizer = Tokenizer()
    config = TrainConfig()
    model = Plot2EqModel(
        vocab_size=len(tokenizer.tokens),
        pad_idx=tokenizer.token_map["<pad>"],
        d_model=config.d_model,
        nhead=config.nhead,
        num_enc_layers=config.num_enc_layers,
        num_dec_layers=config.num_dec_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    checkpoint_path = "../checkpoints/v2/best_model.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict.get("model_state_dict", state_dict))
    model.eval()
    model = torch.compile(model)
    print("Модель готова!")
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    mode: str
    image_base64: str | None = None
    formula: str | None = None
    beam_size: int = Field(default=5, ge=1, le=15, description="Beam size for search")
    top_k: int = Field(default=3, ge=1, le=5, description="Top K results to return")

    length_penalty: float = Field(default=0.01, ge=0.0, le=1.0)
    fast_mode: bool = Field(default=False)
    opt_max_iter: int = Field(default=200, ge=10, le=1000)
    opt_popsize: int = Field(default=15, ge=2, le=50)


class FormulaRequest(BaseModel):
    formula: str


def smooth_segments(y, mask, window_length=9, polyorder=3):
    y_smooth = y.copy()
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return y_smooth

    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)

    for g in groups:
        n = len(g)
        if n >= window_length:
            y_smooth[g] = savgol_filter(y[g], window_length, polyorder)
        elif n > polyorder:
            wl = n if n % 2 != 0 else n - 1
            if wl > polyorder:
                y_smooth[g] = savgol_filter(y[g], wl, polyorder)

    return y_smooth


def process_image_to_math(
    image: Image.Image, num_points=256, max_vertical_ratio=0.2, max_y_jump_ratio=0.15
):
    """
    Извлекает координаты точек из нарисованного графика.

    :param max_vertical_ratio: Максимальная доля от высоты канваса,
                               при которой штрих считается прямой вертикальной линией (игнорируется).
    :param max_y_jump_ratio: Максимально допустимый скачок по Y между двумя соседними
                             интерполированными X. Если скачок больше - линия разрывается.
    """
    image = image.convert("RGBA")
    image_data = np.array(image)
    height, width = image_data.shape[:2]

    # Бинаризация и очистка от шума
    binary_mask = image_data[:, :, 3] > 10

    labels = label(binary_mask)
    clean_mask = np.zeros_like(binary_mask)
    min_span = max(width, height) * 0.03

    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        span_y = max_row - min_row
        span_x = max_col - min_col

        if max(span_x, span_y) >= min_span and region.area >= 10:
            clean_mask[labels == region.label] = True

    binary_mask = clean_mask
    skeleton = skeletonize(binary_mask)

    # Сканирование колонок
    raw_y = np.zeros(width)
    raw_mask = np.zeros(width, dtype=bool)
    prev_y_idx = None

    vertical_threshold = height * max_vertical_ratio

    for x in range(width):
        col = skeleton[:, x]
        y_indices = np.where(col)[0]

        if len(y_indices) == 0 or len(y_indices) > vertical_threshold:
            prev_y_idx = None
            continue

        if prev_y_idx is None:
            best_y_idx = np.mean(y_indices)
        else:
            best_y_idx = min(y_indices, key=lambda idx: abs(idx - prev_y_idx))

        math_y_val = 10.0 - 20.0 * (best_y_idx / height)

        raw_y[x] = math_y_val
        raw_mask[x] = True
        prev_y_idx = best_y_idx

    # Биннинг до num_points
    target_y = np.zeros(num_points)
    target_mask = np.zeros(num_points, dtype=bool)
    x_target = np.linspace(-10, 10, num_points)

    bin_edges = np.linspace(0, width, num_points + 1)

    for i in range(num_points):
        start_idx = int(bin_edges[i])
        end_idx = int(bin_edges[i + 1])

        bin_mask = raw_mask[start_idx:end_idx]
        if np.any(bin_mask):
            target_y[i] = np.mean(raw_y[start_idx:end_idx][bin_mask])
            target_mask[i] = True

    # Разделение на сегменты и фильтрация
    valid_indices = np.where(target_mask)[0]
    if len(valid_indices) > 0:
        y_span = 20.0
        jump_threshold = y_span * max_y_jump_ratio
        y_vals = target_y[valid_indices]

        jumps = np.where(np.abs(np.diff(y_vals)) > jump_threshold)[0]

        for j in jumps:
            target_mask[valid_indices[j]] = False

        valid_indices = np.where(target_mask)[0]
        # -----------------------------------------------------

        if len(valid_indices) > 0:
            groups = np.split(
                valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1
            )
            min_group_len = max(3, int(num_points * 0.02))

            for g in groups:
                if len(g) < min_group_len:
                    target_mask[g] = False

    # Сглаживание сегментов
    target_y = smooth_segments(target_y, target_mask)

    return x_target, target_y, target_mask


def get_ideal_math(formula_str, num_points=256):
    x_math = np.linspace(-10, 10, num_points)
    try:
        x_sym = sp.Symbol("x", real=True)
        if "\\" in formula_str:
            expr = parse_latex(formula_str)
        else:
            formula_str = (
                formula_str.replace("^", "**").replace("np.", "").replace("math.", "")
            )
            t = standard_transformations + (implicit_multiplication_application,)
            expr = parse_expr(formula_str, transformations=t)

        f = sp.lambdify(x_sym, expr, modules=["numpy"])
        y_math = f(x_math)
        if np.isscalar(y_math) or np.ndim(y_math) == 0:
            y_math = np.full_like(x_math, float(y_math))  # type: ignore

        mask = (y_math >= -10) & (y_math <= 10) & np.isfinite(y_math)
        return x_math, y_math, mask
    except Exception:
        return None, None, None


def create_model_tensor(x_math, y_math, mask, num_points=256):
    valid_x, valid_y = x_math[mask], y_math[mask]
    if len(valid_x) < 2:
        return None

    x_min, x_max = np.min(valid_x), np.max(valid_x)
    y_min, y_max = np.min(valid_y), np.max(valid_y)
    x_range = max(x_max - x_min, 1e-6)

    x_norm = (valid_x - x_min) / x_range
    y_norm = (
        np.full_like(valid_y, 0.5)
        if (y_max - y_min < 1e-6)
        else (valid_y - y_min) / (y_max - y_min)
    )

    sort_idx = np.argsort(x_norm)
    x_norm, y_norm = x_norm[sort_idx], y_norm[sort_idx]

    x_target = np.linspace(0, 1, num_points)
    y_interp = np.interp(x_target, x_norm, y_norm)
    mask_interp = np.ones(num_points, dtype=np.float32)

    return torch.tensor(
        np.stack([y_interp, mask_interp]), dtype=torch.float32
    ).unsqueeze(0)


def evaluate_expr_to_points(expr, x_array):
    try:
        x_sym = sp.Symbol("x", real=True)
        f = sp.lambdify(x_sym, expr, modules=["numpy"])
        y_array = f(x_array)
        if np.isscalar(y_array) or np.ndim(y_array) == 0:
            y_array = np.full_like(x_array, float(y_array))  # type: ignore

        y_array = np.where(
            (y_array >= -50) & (y_array <= 50) & np.isfinite(y_array), y_array, np.nan
        )
        return [float(y) if not np.isnan(y) else None for y in y_array]
    except Exception:
        return [None] * len(x_array)


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


def _render_worker(formula_str):
    """Синхронный парсинг для запуска в пуле потоков"""
    x_dense = np.linspace(-10, 10, 500)
    formula_str_clean = (
        formula_str.replace("^", "**").replace("np.", "").replace("math.", "")
    )
    t = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(formula_str_clean, transformations=t)
    y_dense = evaluate_expr_to_points(expr, x_dense)
    return x_dense.tolist(), y_dense


@app.post("/render_formula")
async def render_formula(req: FormulaRequest):
    if len(req.formula) > 200:
        return {"x": [], "y": []}

    loop = asyncio.get_running_loop()
    try:
        x_dense, y_dense = await asyncio.wait_for(
            loop.run_in_executor(None, _render_worker, req.formula), timeout=10.0
        )
        return {"x": x_dense, "y": y_dense}
    except asyncio.TimeoutError:
        return {"x": [], "y": []}
    except Exception:
        return {"x": [], "y": []}


@app.post("/predict")
async def predict(data: PredictRequest):
    loop = asyncio.get_running_loop()

    if data.mode == "draw":
        if not data.image_base64:
            return {"error": "Нет изображения."}
        try:
            if "," in data.image_base64:
                _, encoded = data.image_base64.split(",", 1)
            else:
                encoded = data.image_base64
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        except Exception:
            return {"error": "Неверный или поврежденный формат изображения."}

        try:
            x_math, y_math, mask = await asyncio.wait_for(
                loop.run_in_executor(None, process_image_to_math, image), timeout=5.0
            )
        except asyncio.TimeoutError:
            return {
                "error": "Таймаут: обработка изображения заняла слишком много времени."
            }
        except Exception as e:
            return {"error": f"Ошибка обработки изображения: {str(e)}"}

    else:
        if not data.formula:
            return {"error": "Введите формулу."}
        if len(data.formula) > 200:
            return {"error": "Формула слишком длинная (максимум 200 символов)."}

        try:
            x_math, y_math, mask = await asyncio.wait_for(
                loop.run_in_executor(None, get_ideal_math, data.formula), timeout=2.0
            )
        except asyncio.TimeoutError:
            return {"error": "Таймаут: слишком сложная формула для вычисления."}
        except Exception as e:
            return {"error": f"Ошибка в формуле: {str(e)}"}

    if mask is None or x_math is None or y_math is None or not np.any(mask):
        return {"error": "Не удалось распознать математику."}

    pts_tensor = create_model_tensor(x_math, y_math, mask)
    if pts_tensor is None:
        return {"error": "Слишком мало валидных точек на графике."}

    current_beam_size = min(data.beam_size, 2) if data.fast_mode else data.beam_size

    try:
        results = await loop.run_in_executor(
            None,
            lambda: predict_top_k_equations(
                model,
                pts_tensor,
                tokenizer,
                x_data=x_math[mask],
                y_data=y_math[mask],
                beam_size=current_beam_size,
                top_k=data.top_k,
                length_penalty=data.length_penalty,
                fast_mode=data.fast_mode,
                opt_max_iter=data.opt_max_iter,
                opt_popsize=data.opt_popsize,
            ),
        )
    except Exception as e:
        return {"error": f"Ошибка нейросети при предсказании: {str(e)}"}

    dense_x = np.linspace(-10, 10, 500)
    response_data = []

    for res in results:
        response_data.append(
            {
                "latex": sp.latex(res["best_expr"]),
                "skeleton": res["skeleton"],
                "mse": float(res["mse"]),
                "score": float(res["score"]),
                "pred_y": evaluate_expr_to_points(res["best_expr"], dense_x),
            }
        )

    return {
        "scatter_x": x_math[mask].tolist(),
        "scatter_y": y_math[mask].tolist(),
        "dense_x": dense_x.tolist(),
        "results": response_data,
    }
