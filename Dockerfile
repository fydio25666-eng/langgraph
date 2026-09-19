FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies before copying source to keep Docker layer caching effective.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run the service with a dedicated unprivileged account.
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY app.py auth.py checkpoint.py config.py demo.py graph.py main.py nodes.py rag.py schemas.py healthcheck.py ./
COPY 商品售后手册.txt ./
COPY static ./static
RUN mkdir -p /app/.chroma && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
