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
  evaluateBatchV1,
  getTaskStatusV1,
  proxyAnswerKey,
  proxyReScore,
  proxyDebugEvaluate,
} from './controllers/evaluationController.js';

import {
  createSession,
  submitStudentResult,
  getSessionResults,
  downloadReport,
  submitAbsentees,
  getSession,
  getPendingCount
} from './controllers/reportController.js';
import { runMigrations } from './config/migrator.js';
import pool from './config/database.js';

// ── App setup ──────────────────────────────────────────────────────────────
const app  = express();
const PORT = Number(process.env.PORT ?? 3000);

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(helmet({ crossOriginResourcePolicy: false })); // security headers (allow CORS)
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

// Separated frontend deployment FastAPI proxies
app.post('/answer-key', proxyAnswerKey);
app.post('/re-score', proxyReScore);
app.post('/debug/evaluate', upload.single('file'), proxyDebugEvaluate);

app.get('/debug/db-status', async (_req, res) => {
  try {
    const tablesResult = await pool.query(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema='public'
    `);
    const tables = tablesResult.rows.map(r => r.table_name);
    
    let migrations = [];
    if (tables.includes('schema_migrations')) {
      const migResult = await pool.query('SELECT * FROM schema_migrations ORDER BY applied_at ASC');
      migrations = migResult.rows;
    }
    
    return res.json({
      status: 'connected',
      tables,
      migrations,
    });
  } catch (err) {
    return res.status(500).json({
      status: 'error',
      message: err.message,
      code: err.code,
    });
  }
});

// Async Batch Ingestion Routes (V1)
app.post('/api/v1/evaluate', upload.array('files'), evaluateBatchV1);
app.get('/api/v1/tasks/:taskId', getTaskStatusV1);

// Session & Reports
app.post('/api/sessions', createSession);
app.get('/api/sessions/:sessionId', getSession);
app.get('/api/sessions/:sessionId/pending-count', getPendingCount);
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
const server = app.listen(PORT, async () => {
  console.log(`[Gateway] Listening on port ${PORT}`);
  console.log(`[Gateway] Proxying to FastAPI at: ${process.env.FASTAPI_URL ?? 'http://localhost:8000'}`);
  
  try {
    await runMigrations();
  } catch (err) {
    console.error('[Gateway Startup] Database migration failed:', err.message);
  }
});

// ── WebSocket Proxying ──────────────────────────────────────────────────────
import { WebSocketServer, WebSocket as WSClient } from 'ws';

const wss = new WebSocketServer({ noServer: true });
const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

server.on('upgrade', (request, socket, head) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  
  if (url.pathname === '/ws/evaluate') {
    wss.handleUpgrade(request, socket, head, (ws) => {
      // Derive backend websocket URL
      let targetWsUrl = FASTAPI_URL.replace(/^http/, 'ws');
      if (!targetWsUrl.endsWith('/ws/evaluate')) {
        targetWsUrl = `${targetWsUrl.replace(/\/$/, '')}/ws/evaluate`;
      }
      
      console.log(`[Gateway WS Proxy] Connecting to backend: ${targetWsUrl}`);
      const backendWs = new WSClient(targetWsUrl, {
        headers: {
          'user-agent': request.headers['user-agent'] || 'Mozilla/5.0',
          'origin': request.headers['origin'] || 'https://smruthiravindra.github.io',
        }
      });
      
      backendWs.on('open', () => {
        console.log('[Gateway WS Proxy] Backend connection opened');
      });
      
      backendWs.on('message', (message, isBinary) => {
        if (ws.readyState === ws.OPEN) {
          ws.send(message, { binary: isBinary });
        }
      });
      
      backendWs.on('close', (code, reason) => {
        console.log(`[Gateway WS Proxy] Backend closed: ${code} ${reason}`);
        ws.close(code, reason);
      });
      
      backendWs.on('error', (err) => {
        console.error('[Gateway WS Proxy] Backend error:', err.message);
        ws.close(1011, `Backend connection error to ${targetWsUrl}: ${err.message}`);
      });
      
      ws.on('message', (message, isBinary) => {
        if (backendWs.readyState === backendWs.OPEN) {
          backendWs.send(message, { binary: isBinary });
        }
      });
      
      ws.on('close', (code, reason) => {
        console.log(`[Gateway WS Proxy] Client closed: ${code} ${reason}`);
        backendWs.close(code, reason);
      });
      
      ws.on('error', (err) => {
        console.error('[Gateway WS Proxy] Client error:', err.message);
        backendWs.close(1011, 'Client error');
      });
    });
  } else {
    socket.destroy();
  }
});

export default app;
