/**
 * index.js
 * --------
 * Entry point for the Node.js / Express data gateway.
 *
 * This server acts as an intermediate route server between the React Native
 * mobile client and the Python FastAPI backend:
 *
 *   Mobile Client  →  [POST /api/evaluate]  →  Data Gateway  →  FastAPI
 *                                                    ↓
 *                                              PostgreSQL (persist)
 *                                                    ↓
 *                ←──────────────── JSON Response ───────────────────
 *
 * Routes:
 *   GET  /health           – Liveness probe
 *   POST /api/evaluate     – Proxy image to FastAPI + return results
 *   POST /api/submit       – Persist confirmed results to PostgreSQL
 *   GET  /api/history      – Query evaluation history
 */

import express   from 'express';
import multer    from 'multer';
import cors      from 'cors';
import helmet    from 'helmet';
import morgan    from 'morgan';
import 'dotenv/config';

import {
  evaluateSheet,
  submitResults,
  getHistory,
} from './controllers/evaluationController.js';

import {
  createSession,
  submitStudentResult,
  getSessionResults,
  downloadReport,
  submitAbsentees
} from './controllers/reportController.js';

// ── App setup ──────────────────────────────────────────────────────────────
const app  = express();
const PORT = Number(process.env.PORT ?? 3000);

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(helmet());         // security headers
app.use(cors());           // allow all origins (restrict in production)
app.use(morgan('dev'));    // HTTP request logging
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// ── File upload (memory storage; forwarded directly to FastAPI) ────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: 10 * 1024 * 1024 }, // 10 MB max
  fileFilter: (_req, file, cb) => {
    const allowed = ['image/jpeg', 'image/png', 'image/jpg'];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported MIME type: ${file.mimetype}`), false);
    }
  },
});

// ── Routes ─────────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => res.json({ status: 'ok', service: 'data-gateway' }));

app.post('/api/evaluate', upload.single('file'), evaluateSheet);
app.post('/api/submit',   submitResults);
app.get('/api/history',   getHistory);

// Session & Reports
app.post('/api/sessions', createSession);
app.post('/api/results', submitStudentResult);
app.post('/api/sessions/:sessionId/absentees', submitAbsentees);
app.get('/api/sessions/:sessionId/results', getSessionResults);
app.get('/api/reports/download/:sessionId', downloadReport);

// ── 404 handler ────────────────────────────────────────────────────────────
app.use((_req, res) => {
  res.status(404).json({ error: 'Route not found.' });
});

// ── Global error handler ───────────────────────────────────────────────────
// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  console.error('[Gateway Error]', err.message);
  res.status(err.status ?? 500).json({ error: err.message ?? 'Internal server error.' });
});

// ── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`[Gateway] Listening on port ${PORT}`);
  console.log(`[Gateway] Proxying to FastAPI at: ${process.env.FASTAPI_URL ?? 'http://localhost:8000'}`);
});

export default app;
