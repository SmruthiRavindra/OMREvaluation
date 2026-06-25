# Vision OMR System

An end-to-end Optical Mark Recognition (OMR) system for evaluating handwritten answer sheets
using computer vision and deep learning.

```
Vision_OMR_System/
│
├── backend_ai/          # Python · FastAPI · YOLOv8 · OpenCV
├── mobile_client/       # React Native mobile app
└── data_gateway/        # Node.js · Express · PostgreSQL
```

---

## Architecture

```
Mobile App (React Native)
       │
       │  multipart/form-data  (POST /api/evaluate)
       ▼
Data Gateway (Express · Node.js)         ← persists to PostgreSQL
       │
       │  forwards image  (POST /evaluate)
       ▼
Backend AI (FastAPI · Python)
    ├── preprocess.py   – Bilateral filter + Homography
    ├── localization.py – YOLOv8 bubble detection + custom NMS
    └── classification.py – Filled / Empty / Ambiguous per bubble
```

---

## Quick Start

### 1 · Backend AI

```bash
cd backend_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your weights file
cp /path/to/OMR_best_model.pt weights/

# Run dev server
uvicorn main:app --reload --port 8000
```

### 2 · Data Gateway

```bash
cd data_gateway

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env: set PGUSER, PGPASSWORD, FASTAPI_URL …

# Run dev server
npm run dev
```

### 3 · Mobile Client

```bash
cd mobile_client

npm install

# iOS
cd ios && pod install && cd ..
npm run ios

# Android
npm run android
```

---

## Docker (Backend AI only)

```bash
cd backend_ai
docker build -t vision-omr-backend .
docker run -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  vision-omr-backend
```

---

## Environment Variables

### Data Gateway (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Gateway listening port |
| `FASTAPI_URL` | `http://localhost:8000` | Backend AI URL |
| `PGHOST` | `localhost` | PostgreSQL host |
| `PGPORT` | `5432` | PostgreSQL port |
| `PGDATABASE` | `omr_db` | Database name |
| `PGUSER` | – | DB user |
| `PGPASSWORD` | – | DB password |

---

## API Reference

### Backend AI (FastAPI) — port 8000

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Liveness probe |
| `POST` | `/evaluate` | Run OMR pipeline on uploaded image |
| `GET` | `/docs` | Swagger UI |

### Data Gateway (Express) — port 3000

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/evaluate` | Upload sheet, get grading results |
| `POST` | `/api/submit` | Persist confirmed results |
| `GET` | `/api/history` | Query past evaluations |
