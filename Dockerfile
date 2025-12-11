# Dockerfile (place at repo root)
FROM python:3.10-slim

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements first for layer caching
COPY app/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# copy app code & model
COPY app /app

# set model path env (matches main.py default)
ENV MODEL_PATH=/app/model/model.pkl
EXPOSE 8080

# Run using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
