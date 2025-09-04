[![Build Status](https://github.com/crystalrey23/credit-scoring-api/actions/workflows/ci.yml/badge.svg)](https://github.com/crystalrey23/credit-scoring-api/actions)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Credit Scoring API

## Prasyarat
- Python 3.9 atau lebih baru
- Uvicorn (`pip install uvicorn`)
- Library lainnya (`pip install -r requirements.txt`)

## Endpoints

### Health Check
`GET /health`

Response:
```json
{"status": "ok"}
```

### Predict
`POST /predict`

Body (JSON):
```json
{
  "features": [0.5, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Response (JSON):
```json
{
  "probability": 0.72,
  "label": 1
}
```

## Menjalankan Server
1. Buka terminal pada direktori root proyek (`icame_project`)
2. Jalankan server API:
   ```bash
   uvicorn src.app:app --reload
   ```

## Mengirim Request Prediksi
1. Buka terminal baru (jangan hentikan server di terminal pertama)
2. Buat file `payload.json` berisi input contoh:
   ```bash
   printf '{"features":[0.5,-1.2,0.3,1.0,0.0,0.1,-0.1,0.2,0.9,-0.5,0.4,0.6,-0.3,1.2,-0.4,0.7,0.8,-0.6,0.2,1.1,-0.2,0.05,-0.05,0.33]}' > payload.json
   ```
3. Kirim request dengan `curl`:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     --data @payload.json
   ```
4. Anda akan melihat respons seperti:
   ```json
   {"probability":0.75,"label":1}
   ```

## Docker (Opsional)
1. Buat file `Dockerfile` di root proyek:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN pip install --no-cache-dir -r requirements.txt
   CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8000"]
   ```
2. Bangun image:
   ```bash
   docker build -t credit-api .
   ```
3. Jalankan container:
   ```bash
   docker run -p 8000:8000 credit-api
   ```

## Versi Kontrol & Rilis
1. Commit perubahan:
   ```bash
   git add README.md payload.json Dockerfile
   git commit -m "Add documentation and Dockerfile"
   ```
2. Tag rilis:
   ```bash
   git tag v1.0
   git push
   git push --tags
   ```