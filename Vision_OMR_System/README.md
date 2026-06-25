# Vision OMR System

An end-to-end Optical Mark Recognition (OMR) system for evaluating handwritten answer sheets
using computer vision and deep learning.

```
Vision_OMR_System/
│
├── backend_ai/          # Python · FastAPI · YOLOv8 · OpenCV
│   ├── core/
│   │   ├── preprocess.py      # Bilateral filter + Homography warp
│   │   ├── localization.py    # YOLOv8 bubble detection + custom NMS
│   │   ├── classification.py  # Filled / Empty / Ambiguous per bubble
│   │   └── scoring.py         # Spatial mapping + answer-key grading
│   ├── tests/
│   │   └── test_pipeline.py   # Pytest test suite
│   ├── weights/
│   │   └── OMR_best_model.pt  # YOLOv8-Nano trained weights
│   └── main.py                # FastAPI endpoints
│
├── data_gateway/        # Node.js · Express · PostgreSQL
│   ├── config/
│   │   ├── database.js        # PostgreSQL pool config
│   │   └── migrations/        # SQL schema migrations
│   └── controllers/
│       └── evaluationController.js
│
├── mobile_client/       # React Native mobile app
│   └── src/
│       ├── components/
│       │   ├── CameraScanner.jsx
│       │   └── ResultsModal.jsx
│       └── services/
│           └── api.js
│
└── docker-compose.yml   # Full-stack orchestration
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
    ├── preprocess.py     – Bilateral filter + Homography
    ├── localization.py   – YOLOv8 bubble detection + custom NMS
    ├── classification.py – Filled / Empty / Ambiguous per bubble
    └── scoring.py        – Spatial mapping + answer-key comparison
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
pip install pytest httpx       # for testing

# Place your weights file
cp /path/to/OMR_best_model.pt weights/

# Run dev server
uvicorn main:app --reload --port 8000

# Run tests
python -m pytest tests/ -v
```

### 2 · Data Gateway

```bash
cd data_gateway

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env: set PGUSER, PGPASSWORD, FASTAPI_URL …

# Run database migration
psql -U omr_user -d omr_db -f config/migrations/001_create_evaluations.sql

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

### 4 · Docker (Full Stack)

```bash
# From project root
docker compose up --build
```

This starts PostgreSQL, the FastAPI backend, and the Express gateway together.
Migrations run automatically on first start.

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
| `GET` | `/health` | Liveness probe |
| `POST` | `/evaluate` | Run full OMR pipeline on uploaded image |
| `POST` | `/answer-key` | Upload answer key for a session |
| `GET` | `/answer-key/{session_id}` | Retrieve stored answer key |
| `POST` | `/score` | Run pipeline + grade against answer key |
| `GET` | `/docs` | Swagger UI |

### Data Gateway (Express) — port 3000

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/evaluate` | Upload sheet, get grading results |
| `POST` | `/api/submit` | Persist confirmed results |
| `GET` | `/api/history` | Query past evaluations |

---

## YOLO Model Details

- **Architecture**: YOLOv8-Nano (6.2 MB)
- **Classes**: `filled` (0), `unfilled` (1), `usn` (2)
- **Training**: 100 epochs on 1,354 augmented images
- **Inference**: ~4.5 ms per sheet

## Testing

```bash
cd backend_ai
python -m pytest tests/ -v
```

Tests cover:
- Image preprocessing (decode, bilateral filter, perspective warp)
- Bubble classification (filled/empty threshold logic)
- NMS deduplication (overlapping boxes, cross-class preservation)
- Spatial scoring (grid mapping, answer key comparison)
- FastAPI endpoints (health, answer key CRUD, input validation)
