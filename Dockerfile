FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./

COPY . .

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data

CMD["python", "scripts/generate_data.py"]