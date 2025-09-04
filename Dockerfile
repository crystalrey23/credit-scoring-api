# Stage 1: Install dependencies dengan cache
FROM python:3.9-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN mkdir -p /root/.cache/pip \
 && pip install --upgrade pip \
 && pip install --default-timeout=100 --retries=5 \
       --cache-dir=/root/.cache/pip \
       -r requirements.txt

# Stage 2: Salin aplikasi dan paket terinstal
FROM python:3.9-slim
WORKDIR /app
COPY --from=deps /root/.cache/pip /root/.cache/pip
COPY --from=deps /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY src/ src/
COPY models/ models/
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
