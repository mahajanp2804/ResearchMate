# syntax=docker/dockerfile:1

# ResearchMate API: FastAPI, sentence-transformers and FAISS, CPU only.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Hugging Face models are downloaded on first use into this (volume) directory.
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel brings ~2 GB of CUDA libraries
# the app never uses. sentence-transformers then reuses the installed torch.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY requirements.txt .
RUN pip install -r requirements.txt

# An unprivileged user that owns the data and model directories (both volumes).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /models \
    && chown app:app /app/data /models
COPY --chown=app:app app ./app
COPY --chown=app:app evaluation ./evaluation
COPY --chown=app:app scripts ./scripts
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
