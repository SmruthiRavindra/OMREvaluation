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
import path      from 'path';
import crypto    from 'crypto';
import { fileURLToPath } from 'url';
import rateLimit from 'express-rate-limit';
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
  listSessions,
  submitStudentResult,
  getSessionResults,
  downloadReport,
  submitAbsentees,
  getSession,
  getPendingCount,
  getNextRosterUsn,
  deleteSession,
  saveExamVersion,
  listExamVersions
} from './controllers/reportController.js';
import { runMigrations } from './config/migrator.js';
import pool from './config/database.js';
import {
  login,
  logout,
  verifyAuth,
  isAdmin,
  listUsers,
  createUser,
  deleteUser,
  getAdminStats
} from './controllers/authController.js';

import fs        from 'fs';

// ── App setup ──────────────────────────────────────────────────────────────
const app  = express();
const PORT = Number(process.env.PORT ?? 3000);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const candidateStaticPaths = [
  path.join(__dirname, '../backend_ai/static'),
  '/backend_ai/static',
  path.join(__dirname, 'static'),
  path.join(__dirname, '../../backend_ai/static')
];
const staticPath = candidateStaticPaths.find(p => fs.existsSync(p)) || candidateStaticPaths[0];

// ── Middleware ─────────────────────────────────────────────────────────────
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS ?? 'http://localhost:3000')
  .split(',')
  .map(o => o.trim())
  .filter(Boolean);

app.use(helmet({ contentSecurityPolicy: false, crossOriginResourcePolicy: false })); // security headers (allow CORS)
app.use(cors({
  origin: (origin, cb) => {
    // Allow requests with no origin (e.g. mobile apps, curl, server-to-server)
    if (!origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true);
    cb(new Error(`CORS: origin '${origin}' not in allow-list`));
  },
  credentials: true,
}));
app.use(morgan('dev'));    // HTTP request logging
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Serve static frontend files from Gateway
app.use(express.static(staticPath));

// ── File upload (memory storage; forwarded directly to FastAPI) ────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: 10 * 1024 * 1024 }, // 10 MB max
  fileFilter: (_req, file, cb) => {
    const allowed = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported MIME type: ${file.mimetype}`), false);
    }
  },
});

// ── Routes ─────────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => res.json({ status: 'ok', service: 'data-gateway' }));

// Middleware to disable caching for static HTML views so updated CSP headers apply instantly
const noCache = (req, res, next) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  next();
};

// Page Routes serving client-side dynamic interfaces
app.get('/login', noCache, (req, res) => {
  res.sendFile(path.join(staticPath, 'login.html'));
});

app.get('/dashboard', noCache, (req, res) => {
  res.sendFile(path.join(staticPath, 'dashboard.html'));
});

app.get('/admin', noCache, (req, res) => {
  res.sendFile(path.join(staticPath, 'admin.html'));
});

app.get('/', (req, res) => {
  res.redirect('/dashboard');
});

const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Limit each IP to 5 login requests per windowMs
  standardHeaders: 'draft-7',
  legacyHeaders: false,
  message: { error: 'Too many login attempts. Please try again in 15 minutes.' }
  // NOTE: If a reverse proxy is ever placed in front of this service,
  // add app.set('trust proxy', N) and adjust appropriately.
});

// Auth Portal Endpoints
app.post('/api/auth/login', loginLimiter, login);
app.post('/api/auth/logout', logout);

// User Management (Admin Only)
app.get('/api/users', verifyAuth, isAdmin, listUsers);
app.post('/api/users', verifyAuth, isAdmin, createUser);
app.delete('/api/users/:id', verifyAuth, isAdmin, deleteUser);
app.get('/api/admin/stats', verifyAuth, isAdmin, getAdminStats);

// Protected Data Evaluation/Submission Routes
app.post('/api/evaluate', verifyAuth, upload.single('file'), evaluateSheet);
app.post('/api/submit',   verifyAuth, submitResults);
app.get('/api/history',   verifyAuth, getHistory);

// Separated frontend deployment FastAPI proxies
app.post('/answer-key', verifyAuth, proxyAnswerKey);
app.post('/re-score', verifyAuth, proxyReScore);
app.post('/debug/evaluate', verifyAuth, upload.single('file'), proxyDebugEvaluate);

app.get('/debug/db-status', verifyAuth, async (_req, res) => {
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
app.post('/api/v1/evaluate', verifyAuth, upload.array('files'), evaluateBatchV1);
app.get('/api/v1/tasks/:taskId', verifyAuth, getTaskStatusV1);

// Session & Reports (All protected with verifyAuth)
app.post('/api/sessions', verifyAuth, createSession);
app.get('/api/sessions', verifyAuth, listSessions);
app.get('/api/sessions/:sessionId', verifyAuth, getSession);
app.delete('/api/sessions/:sessionId', verifyAuth, isAdmin, deleteSession);
app.get('/api/sessions/:sessionId/pending-count', verifyAuth, getPendingCount);
app.get('/api/sessions/:sessionId/next-usn', verifyAuth, getNextRosterUsn);
app.post('/api/sessions/:sessionId/versions', verifyAuth, saveExamVersion);
app.get('/api/sessions/:sessionId/versions', verifyAuth, listExamVersions);
app.post('/api/results', verifyAuth, submitStudentResult);
app.post('/api/sessions/:sessionId/absentees', verifyAuth, submitAbsentees);
app.get('/api/sessions/:sessionId/results', verifyAuth, getSessionResults);
app.get('/api/reports/download/:sessionId', verifyAuth, downloadReport);

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
    const token = url.searchParams.get('token');
    if (!token) {
      console.warn('[Gateway WS Proxy] Upgrade rejected: Token query param missing');
      socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
      socket.destroy();
      return;
    }

    const tokenHash = crypto.createHash('sha256').update(token).digest('hex');
    // Verify token against database
    pool.query(
      `SELECT 1 FROM user_sessions s 
       WHERE s.token_hash = $1 AND s.expires_at > NOW()`,
      [tokenHash]
    ).then((dbRes) => {
      if (dbRes.rowCount === 0) {
        console.warn('[Gateway WS Proxy] Upgrade rejected: Session expired or invalid');
        socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
        socket.destroy();
        return;
      }

      // Upgrade to websocket if verified
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
        if (code && code !== 1005 && code !== 1006) {
          ws.close(code, reason);
        } else {
          ws.close(1000);
        }
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
        if (code && code !== 1005 && code !== 1006) {
          backendWs.close(code, reason);
        } else {
          backendWs.close(1000);
        }
      });
      
      ws.on('error', (err) => {
        console.error('[Gateway WS Proxy] Client error:', err.message);
        backendWs.close(1011, 'Client error');
      });
      });
    }).catch((err) => {
      console.error('[Gateway WS Proxy] Auth DB error during upgrade:', err.message);
      socket.write('HTTP/1.1 500 Internal Server Error\r\n\r\n');
      socket.destroy();
    });
  } else {
    socket.destroy();
  }
});

export default app;
