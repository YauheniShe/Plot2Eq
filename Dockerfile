FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /code

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml ./

COPY --chown=user src/ src/
COPY --chown=user app/ app/
COPY --chown=user checkpoints/ checkpoints/

RUN pip install --no-cache-dir .

WORKDIR /code/app

EXPOSE 7860

CMD["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]