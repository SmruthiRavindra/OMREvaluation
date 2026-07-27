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

  const session_id = req.body?.session_id || 'default';
  let roster = null;
  let use_roster_order = false;
  let assigned_usn = null;
  try {
    const sRes = await query("SELECT roster, use_roster_order FROM exam_sessions WHERE id = $1", [session_id]);
    if (sRes.rows.length > 0) {
      roster = sRes.rows[0].roster;
      use_roster_order = sRes.rows[0].use_roster_order || false;
    }

    if (use_roster_order && roster) {
      const rosterList = typeof roster === 'string' ? JSON.parse(roster) : roster;
      if (Array.isArray(rosterList) && rosterList.length > 0) {
        const absRes = await query("SELECT usn FROM student_results WHERE session_id = $1 AND status = 'ABSENT'", [session_id]);
        const absentees = absRes.rows.map(r => r.usn);
        const activeRoster = rosterList.filter(u => !absentees.includes(u));
        
        const countRes = await query("SELECT COUNT(*) as count FROM student_results WHERE session_id = $1 AND status != 'ABSENT'", [session_id]);
        const savedCount = parseInt(countRes.rows[0]?.count || 0);
        
        if (savedCount < activeRoster.length) {
          assigned_usn = activeRoster[savedCount];
        }
      }
    }
  } catch(e) {
    console.warn('[evaluateSheet] Failed to fetch roster:', e.message);
  }

  try {
    // Build a FormData payload to forward to FastAPI
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename:    req.file.originalname || 'sheet.jpg',
      contentType: req.file.mimetype,
    });
    form.append('session_id', session_id);
    if (req.body?.version) {
      form.append('version', req.body.version);
    }
    if (roster) {
      form.append('roster', typeof roster === 'string' ? roster : JSON.stringify(roster));
    }
    if (assigned_usn) {
      form.append('assigned_usn', assigned_usn);
    }

    const { data } = await axios.post(`${FASTAPI_URL}/evaluate`, form, {
      headers: form.getHeaders(),
      timeout: 45_000, // 45 s
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
    let was_duplicate = false;
    if (session_id && student_id) {
      const existing = await query(
        `SELECT id FROM student_results WHERE session_id = $1 AND usn = $2`,
        [session_id, student_id]
      );
      if (existing.rows.length > 0) {
        was_duplicate = true;
      }
    }

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

    return res.status(201).json({
      id: result.rows[0].id,
      saved: true,
      was_duplicate,
      message: was_duplicate ? `Existing result for USN ${student_id} was updated/overwritten.` : 'Result saved.'
    });
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

export async function evaluateBatchV1(req, res) {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: 'No files uploaded. Use field name "files".' });
  }

  try {
    const form = new FormData();
    for (const file of req.files) {
      form.append('files', file.buffer, {
        filename: file.originalname || 'sheet.jpg',
        contentType: file.mimetype,
      });
    }

    const session_id = req.body.session_id || 'default';
    let roster = null;
    let use_roster_order = false;
    let assigned_usns = null;
    try {
      const sRes = await query("SELECT roster, use_roster_order FROM exam_sessions WHERE id = $1", [session_id]);
      if (sRes.rows.length > 0) {
        roster = sRes.rows[0].roster;
        use_roster_order = sRes.rows[0].use_roster_order || false;
      }

      if (use_roster_order && roster) {
        const rosterList = typeof roster === 'string' ? JSON.parse(roster) : roster;
        if (Array.isArray(rosterList) && rosterList.length > 0) {
          const absRes = await query("SELECT usn FROM student_results WHERE session_id = $1 AND status = 'ABSENT'", [session_id]);
          const absentees = absRes.rows.map(r => r.usn);
          const activeRoster = rosterList.filter(u => !absentees.includes(u));
          
          const countRes = await query("SELECT COUNT(*) as count FROM student_results WHERE session_id = $1 AND status != 'ABSENT'", [session_id]);
          const savedCount = parseInt(countRes.rows[0]?.count || 0);
          
          const sliceUsns = activeRoster.slice(savedCount);
          assigned_usns = sliceUsns;
        }
      }
    } catch(e) {
      console.warn('[evaluateBatchV1] Failed to fetch roster:', e.message);
    }

    form.append('session_id', session_id);
    if (req.body?.version) {
      form.append('version', req.body.version);
    }
    if (roster) {
      form.append('roster', typeof roster === 'string' ? roster : JSON.stringify(roster));
    }
    if (assigned_usns) {
      form.append('assigned_usns', JSON.stringify(assigned_usns));
    }
    form.append('questions_per_column', req.body.questions_per_column || '15');
    form.append('num_columns', req.body.num_columns || '2');
    form.append('options', req.body.options || 'ABCD');

    const { data } = await axios.post(`${FASTAPI_URL}/api/v1/batch-evaluate`, form, {
      headers: form.getHeaders(),
      timeout: 10_000,
      maxBodyLength: Infinity,
    });

    return res.status(202).json(data);
  } catch (err) {
    const detail =
      err.response?.data?.detail || err.message || 'FastAPI proxy error';
    console.error('[evaluateBatchV1] error:', detail);
    return res.status(502).json({ error: detail });
  }
}

export async function getTaskStatusV1(req, res) {
  const { taskId } = req.params;
  try {
    const { data } = await axios.get(`${FASTAPI_URL}/api/v1/tasks/${taskId}`);
    return res.json(data);
  } catch (err) {
    const detail =
      err.response?.data?.detail || err.message || 'FastAPI proxy error';
    console.error('[getTaskStatusV1] error:', detail);
    return res.status(err.response?.status || 502).json({ error: detail });
  }
}

// ── Proxy helpers for separating frontend to static deployment ─────────────

export async function proxyAnswerKey(req, res) {
  try {
    const { data } = await axios.post(`${FASTAPI_URL}/answer-key`, req.body, {
      timeout: 10_000,
    });
    return res.json(data);
  } catch (err) {
    const detail = err.response?.data?.detail || err.message || 'FastAPI proxy error';
    console.error('[proxyAnswerKey] error:', detail);
    return res.status(err.response?.status || 502).json({
      error: `Gateway proxy failed for ${FASTAPI_URL}/answer-key`,
      code: err.code || 'UNKNOWN',
      detail: detail
    });
  }
}

export async function proxyReScore(req, res) {
  try {
    const { data } = await axios.post(`${FASTAPI_URL}/re-score`, req.body, {
      timeout: 15_000,
    });
    return res.json(data);
  } catch (err) {
    const detail = err.response?.data?.detail || err.message || 'FastAPI proxy error';
    console.error('[proxyReScore] error:', detail);
    return res.status(err.response?.status || 502).json({
      error: `Gateway proxy failed for ${FASTAPI_URL}/re-score`,
      code: err.code || 'UNKNOWN',
      detail: detail
    });
  }
}

export async function proxyDebugEvaluate(req, res) {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded. Use field name "file".' });
  }
  
  const session_id = req.body?.session_id || 'default';
  let roster = null;
  let use_roster_order = false;
  let assigned_usn = null;
  try {
    const sRes = await query("SELECT roster, use_roster_order FROM exam_sessions WHERE id = $1", [session_id]);
    if (sRes.rows.length > 0) {
      roster = sRes.rows[0].roster;
      use_roster_order = sRes.rows[0].use_roster_order || false;
    }

    if (use_roster_order && roster) {
      const rosterList = typeof roster === 'string' ? JSON.parse(roster) : roster;
      if (Array.isArray(rosterList) && rosterList.length > 0) {
        const absRes = await query("SELECT usn FROM student_results WHERE session_id = $1 AND status = 'ABSENT'", [session_id]);
        const absentees = absRes.rows.map(r => r.usn);
        const activeRoster = rosterList.filter(u => !absentees.includes(u));
        
        const countRes = await query("SELECT COUNT(*) as count FROM student_results WHERE session_id = $1 AND status != 'ABSENT'", [session_id]);
        const savedCount = parseInt(countRes.rows[0]?.count || 0);
        
        if (savedCount < activeRoster.length) {
          assigned_usn = activeRoster[savedCount];
        }
      }
    }
  } catch(e) {
    console.warn('[proxyDebugEvaluate] Failed to fetch roster:', e.message);
  }

  try {
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'sheet.jpg',
      contentType: req.file.mimetype,
    });
    form.append('session_id', session_id);
    if (req.body?.version) {
      form.append('version', req.body.version);
    }
    if (roster) {
      form.append('roster', typeof roster === 'string' ? roster : JSON.stringify(roster));
    }
    if (assigned_usn) {
      form.append('assigned_usn', assigned_usn);
    }

    const { data } = await axios.post(`${FASTAPI_URL}/debug/evaluate`, form, {
      headers: form.getHeaders(),
      timeout: 45_000,
      maxBodyLength: Infinity,
    });
    return res.json(data);
  } catch (err) {
    const detail = err.response?.data?.detail || err.message || 'FastAPI proxy error';
    console.error('[proxyDebugEvaluate] error:', detail);
    return res.status(err.response?.status || 502).json({
      error: `Gateway proxy failed for ${FASTAPI_URL}/debug/evaluate`,
      code: err.code || 'UNKNOWN',
      detail: detail
    });
  }
}
