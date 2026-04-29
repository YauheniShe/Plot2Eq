import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import sympy as sp
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel
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
    print("Модель готова!")
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    mode: str
    image_base64: str = None  # type: ignore
    formula: str = None  # type: ignore
    beam_size: int = 5
    top_k: int = 3


class FormulaRequest(BaseModel):
    formula: str


def process_image_to_math(image: Image.Image, num_points=256):
    image_data = np.array(image)
    height, width = image_data.shape[:2]
    is_drawn = image_data[:, :, 3] > 10

    y_math_raw = np.zeros(width)
    mask_raw = np.zeros(width, dtype=bool)

    for x in range(width):
        col = is_drawn[:, x]
        if np.any(col):
            y_idx = np.average(np.arange(height), weights=col)
            y_math_raw[x] = 10.0 - 20.0 * (y_idx / height)
            mask_raw[x] = True

    x_orig = np.linspace(-10, 10, width)
    x_target = np.linspace(-10, 10, num_points)

    y_math = (
        np.interp(x_target, x_orig[mask_raw], y_math_raw[mask_raw])
        if np.any(mask_raw)
        else np.zeros(num_points)
    )
    mask = np.interp(x_target, x_orig, mask_raw.astype(float)) > 0.5
    return x_target, y_math, mask


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
    return templates.TemplateResponse("index.html", {"request": request})


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
    if data.mode == "draw":
        if not data.image_base64:
            return {"error": "Нет изображения."}
        header, encoded = data.image_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        x_math, y_math, mask = process_image_to_math(image)
    else:
        if not data.formula:
            return {"error": "Введите формулу."}
        if len(data.formula) > 200:
            return {"error": "Формула слишком длинная (максимум 200 символов)."}

        loop = asyncio.get_running_loop()
        try:
            x_math, y_math, mask = await asyncio.wait_for(
                loop.run_in_executor(None, get_ideal_math, data.formula), timeout=2.0
            )
        except asyncio.TimeoutError:
            return {"error": "Таймаут: слишком сложная формула для вычисления."}
        except Exception as e:
            return {"error": f"Ошибка в формуле: {str(e)}"}

    if mask is None or not np.any(mask):
        return {"error": "Не удалось распознать математику."}

    pts_tensor = create_model_tensor(x_math, y_math, mask)
    if pts_tensor is None:
        return {"error": "Слишком мало точек."}

    results = predict_top_k_equations(
        model,
        pts_tensor,
        tokenizer,
        x_data=x_math[mask],  # type: ignore
        y_data=y_math[mask],  # type: ignore
        beam_size=data.beam_size,
        top_k=data.top_k,
    )

    dense_x = np.linspace(-10, 10, 500)
    response_data = []

    for res in results:
        response_data.append(
            {
                "latex": sp.latex(res["best_expr"]),
                "skeleton": res["skeleton"],
                "mse": float(res["mse"]),
                "pred_y": evaluate_expr_to_points(res["best_expr"], dense_x),
            }
        )

    return {
        "scatter_x": x_math[mask].tolist(),  # type: ignore
        "scatter_y": y_math[mask].tolist(),  # type: ignore
        "dense_x": dense_x.tolist(),
        "results": response_data,
    }
