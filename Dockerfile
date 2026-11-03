FROM python:3.10-slim AS trainer
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt
COPY src/ ./src/
ARG EPOCHS=10
RUN python -m src.train --classes yes no up down --epochs ${EPOCHS} --seed 42 --out /build/artifacts

FROM python:3.10-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ ./src/
COPY --from=trainer /build/artifacts ./artifacts/
COPY --from=trainer /build/samples ./samples/
ENV MODEL_PATH=/app/artifacts/model.pt PORT=8000 LOG_LEVEL=info PYTHONUNBUFFERED=1
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
