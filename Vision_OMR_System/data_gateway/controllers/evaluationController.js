/**
 * evaluationController.js
 * -----------------------
 * Express controller for OMR evaluation requests.
 *
 * Responsibilities:
 *  1. Accept multipart file upload from the mobile client
 *  2. Forward the image to the FastAPI backend (POST /evaluate)
 *  3. Persist the result in PostgreSQL
 *  4. Return the grading payload to the mobile client
 *
 * Routes (registered in index.js):
 *   POST /api/evaluate  → evaluateSheet
 *   POST /api/submit    → submitResults
 *   GET  /api/history   → getHistory
 */

import FormData  from 'form-data';
import axios     from 'axios';
import { query } from '../config/database.js';

// FastAPI service URL (defaults to localhost inside the same Docker network)
const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000';

// ── evaluateSheet ──────────────────────────────────────────────────────────

/**
 * Proxy the uploaded image to FastAPI and return grading results.
 *
 * Expects: multipart/form-data with a 'file' field (JPEG/PNG).
 * Returns: JSON EvaluationResponse from the Python backend.
 */
export async function evaluateSheet(req, res) {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded. Use field name "file".' });
  }

  try {
    // Build a FormData payload to forward to FastAPI
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename:    req.file.originalname || 'sheet.jpg',
      contentType: req.file.mimetype,
    });

    const { data } = await axios.post(`${FASTAPI_URL}/evaluate`, form, {
      headers: form.getHeaders(),
      timeout: 25_000, // 25 s
      maxBodyLength: Infinity,
    });

    return res.json(data);
  } catch (err) {
    const detail =
      err.response?.data?.detail || err.message || 'FastAPI proxy error';
    console.error('[evaluateSheet] error:', detail);
    return res.status(502).json({ error: detail });
  }
}

// ── submitResults ──────────────────────────────────────────────────────────

/**
 * Persist a confirmed evaluation result in PostgreSQL.
 *
 * Body (JSON):
 *  {
 *    student_id?         : string,
 *    session_id?         : string,
 *    filled_count        : number,
 *    empty_count         : number,
 *    ambiguous_count     : number,
 *    needs_manual_review : boolean,
 *    bubbles             : BubbleResult[],
 *    processing_time_ms  : number,
 *  }
 */
export async function submitResults(req, res) {
  const {
    student_id          = null,
    session_id          = null,
    filled_count        = 0,
    empty_count         = 0,
    ambiguous_count     = 0,
    needs_manual_review = false,
    bubbles             = [],
    processing_time_ms  = 0,
  } = req.body ?? {};

  try {
    const result = await query(
      `INSERT INTO evaluations
         (student_id, session_id, filled_count, empty_count, ambiguous_count,
          needs_manual_review, bubbles, processing_time_ms, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
       RETURNING id`,
      [
        student_id,
        session_id,
        filled_count,
        empty_count,
        ambiguous_count,
        needs_manual_review,
        JSON.stringify(bubbles),
        processing_time_ms,
      ],
    );

    return res.status(201).json({ id: result.rows[0].id, saved: true });
  } catch (err) {
    console.error('[submitResults] db error:', err.message);
    return res.status(500).json({ error: 'Failed to persist results.' });
  }
}

// ── getHistory ─────────────────────────────────────────────────────────────

/**
 * Retrieve past evaluations, optionally filtered by studentId / sessionId.
 *
 * Query params:
 *   studentId  (optional)
 *   sessionId  (optional)
 *   limit      (default 20, max 100)
 *   offset     (default 0)
 */
export async function getHistory(req, res) {
  const { studentId, sessionId } = req.query;
  const limit  = Math.min(Number(req.query.limit  ?? 20), 100);
  const offset = Number(req.query.offset ?? 0);

  const conditions = [];
  const params     = [];

  if (studentId) { conditions.push(`student_id = $${params.length + 1}`); params.push(studentId); }
  if (sessionId) { conditions.push(`session_id = $${params.length + 1}`); params.push(sessionId); }

  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  params.push(limit, offset);

  try {
    const result = await query(
      `SELECT id, student_id, session_id, filled_count, empty_count,
              ambiguous_count, needs_manual_review, processing_time_ms, created_at
       FROM evaluations
       ${where}
       ORDER BY created_at DESC
       LIMIT $${params.length - 1} OFFSET $${params.length}`,
      params,
    );
    return res.json({ rows: result.rows, total: result.rowCount });
  } catch (err) {
    console.error('[getHistory] db error:', err.message);
    return res.status(500).json({ error: 'Failed to fetch history.' });
  }
}
