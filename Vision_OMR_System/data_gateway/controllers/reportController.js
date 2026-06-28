/**
 * reportController.js
 * -------------------
 * Express controller for exam sessions, student results, and report generation.
 *
 * Routes:
 *   POST /api/sessions                        -> createSession
 *   POST /api/results                         -> submitStudentResult
 *   GET  /api/sessions/:sessionId/results     -> getSessionResults
 *   GET  /api/reports/download/:sessionId     -> downloadReport
 */

import { query } from '../config/database.js';
import ExcelJS from 'exceljs';
import PDFDocument from 'pdfkit';

// ── Session Management ─────────────────────────────────────────────────────

export async function createSession(req, res) {
  const { id, subject, section, exam_date, total_questions } = req.body;

  if (!id || !subject) {
    return res.status(400).json({ error: 'Missing required fields: id, subject' });
  }

  try {
    await query(
      `INSERT INTO exam_sessions (id, subject, section, exam_date, total_questions, expected_students, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, NOW())`,
      [id, subject, section || null, exam_date || new Date(), total_questions || 30, req.body.expected_students || 0]
    );
    return res.status(201).json({ message: 'Session created successfully', id });
  } catch (err) {
    console.error('[createSession] error:', err.message);
    if (err.code === '23505') { // unique_violation
      return res.status(409).json({ error: 'Session ID already exists' });
    }
    return res.status(500).json({ error: 'Failed to create session' });
  }
}

// ── Student Results ────────────────────────────────────────────────────────

export async function submitStudentResult(req, res) {
  const {
    session_id,
    usn,
    score = 0,
    total = 0,
    correct = 0,
    incorrect = 0,
    unanswered = 0,
    multiple_marked = 0,
    score_percent = 0.00,
    per_question = []
  } = req.body;

  if (!session_id || !usn) {
    return res.status(400).json({ error: 'Missing required fields: session_id, usn' });
  }

  try {
    const result = await query(
      `INSERT INTO student_results 
        (session_id, usn, score, total, correct, incorrect, unanswered, multiple_marked, score_percent, per_question, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
       RETURNING id`,
      [session_id, usn, score, total, correct, incorrect, unanswered, multiple_marked, score_percent, JSON.stringify(per_question)]
    );
    return res.status(201).json({ id: result.rows[0].id, saved: true });
  } catch (err) {
    console.error('[submitStudentResult] error:', err.message);
    return res.status(500).json({ error: 'Failed to persist student result' });
  }
}

export async function getSessionResults(req, res) {
  const { sessionId } = req.params;

  try {
    const result = await query(
      `SELECT * FROM student_results WHERE session_id = $1 ORDER BY usn ASC`,
      [sessionId]
    );
    return res.json({ rows: result.rows, total: result.rowCount });
  } catch (err) {
    console.error('[getSessionResults] error:', err.message);
    return res.status(500).json({ error: 'Failed to fetch session results' });
  }
}

export async function submitAbsentees(req, res) {
  const { session_id, usns = [] } = req.body;
  if (!session_id || !Array.isArray(usns)) {
    return res.status(400).json({ error: 'Missing session_id or invalid usns array' });
  }
  
  if (usns.length === 0) return res.json({ saved: 0 });

  try {
    let savedCount = 0;
    for (const usn of usns) {
      if (!usn.trim()) continue;
      await query(
        `INSERT INTO student_results 
          (session_id, usn, score, total, correct, incorrect, unanswered, multiple_marked, score_percent, status, created_at)
         VALUES ($1, $2, 0, 0, 0, 0, 0, 0, 0.00, 'ABSENT', NOW())`,
        [session_id, usn.trim()]
      );
      savedCount++;
    }
    return res.status(201).json({ saved: savedCount });
  } catch (err) {
    console.error('[submitAbsentees] error:', err.message);
    return res.status(500).json({ error: 'Failed to save absentees' });
  }
}

// ── Report Generation ──────────────────────────────────────────────────────

const PASS_THRESHOLD = 40.0; // Configurable pass rate threshold

export async function downloadReport(req, res) {
  const { sessionId } = req.params;
  const format = (req.query.format || 'excel').toLowerCase();

  try {
    // 1. Fetch Session Metadata
    const sessionRes = await query(`SELECT * FROM exam_sessions WHERE id = $1`, [sessionId]);
    if (sessionRes.rowCount === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }
    const session = sessionRes.rows[0];

    // 2. Fetch Student Results
    const resultsRes = await query(
      `SELECT * FROM student_results WHERE session_id = $1 ORDER BY usn ASC`,
      [sessionId]
    );
    const results = resultsRes.rows;

    // 3. Compute Summary Statistics
    let totalScore = 0;
    let passCount = 0;
    let highestScore = 0;
    let lowestScore = session.total_questions;

    results.forEach(r => {
      totalScore += Number(r.score_percent);
      if (Number(r.score_percent) >= PASS_THRESHOLD) passCount++;
      if (r.score > highestScore) highestScore = r.score;
      if (r.score < lowestScore) lowestScore = r.score;
    });

    const classAverage = results.length > 0 ? (totalScore / results.length).toFixed(2) : 0;
    const passRate = results.length > 0 ? ((passCount / results.length) * 100).toFixed(2) : 0;
    if (results.length === 0) lowestScore = 0;

    const stats = {
      totalStudents: results.length,
      expectedStudents: session.expected_students,
      absentCount: results.filter(r => r.status === 'ABSENT').length,
      classAverage,
      passRate,
      highestScore,
      lowestScore
    };

    // 4. Generate the requested format
    if (format === 'excel') {
      return await generateExcel(res, session, results, stats);
    } else if (format === 'pdf') {
      return await generatePDF(res, session, results, stats);
    } else {
      return res.status(400).json({ error: 'Invalid format. Use ?format=excel or ?format=pdf' });
    }

  } catch (err) {
    console.error('[downloadReport] error:', err.message);
    return res.status(500).json({ error: 'Failed to generate report' });
  }
}

// ── Format Generators ──────────────────────────────────────────────────────

async function generateExcel(res, session, results, stats) {
  const workbook = new ExcelJS.Workbook();
  const sheet = workbook.addWorksheet('OMR Results');

  // Header Styles
  const titleFont = { name: 'Arial', size: 16, bold: true, color: { argb: 'FFFFFFFF' } };
  const headerFont = { name: 'Arial', size: 12, bold: true };
  const titleFill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF4F46E5' } }; // Indigo

  // Add Metadata Header
  sheet.mergeCells('A1:F1');
  const titleCell = sheet.getCell('A1');
  titleCell.value = `OMR Evaluation Report: ${session.subject}`;
  titleCell.font = titleFont;
  titleCell.fill = titleFill;
  titleCell.alignment = { vertical: 'middle', horizontal: 'center' };
  sheet.getRow(1).height = 30;

  sheet.addRow(['Session ID:', session.id, '', 'Date:', session.exam_date ? new Date(session.exam_date).toLocaleDateString() : 'N/A']);
  sheet.addRow(['Section:', session.section || 'N/A', '', 'Expected Students:', stats.expectedStudents]);
  sheet.addRow(['Total Present:', stats.totalStudents - stats.absentCount, '', 'Absentees:', stats.absentCount]);
  sheet.addRow([]); // Blank line

  // Add Summary Statistics
  sheet.addRow(['Summary Statistics']);
  sheet.getRow(5).font = headerFont;
  sheet.addRow(['Class Average:', `${stats.classAverage}%`, '', 'Pass Rate:', `${stats.passRate}%`]);
  sheet.addRow(['Highest Score:', stats.highestScore, '', 'Lowest Score:', stats.lowestScore]);
  sheet.addRow([]); // Blank line

  // Data Table
  const tableStartRow = 9;
  sheet.getRow(tableStartRow).values = ['USN', 'Score', 'Total', 'Correct', 'Incorrect', 'Percentage', 'Status'];
  sheet.getRow(tableStartRow).font = headerFont;
  sheet.getRow(tableStartRow).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFF3F4F6' } };

  results.forEach(r => {
    const percent = Number(r.score_percent);
    const status = r.status === 'ABSENT' ? 'ABSENT' : (percent >= PASS_THRESHOLD ? 'PASS' : 'FAIL');
    sheet.addRow([
      r.usn,
      r.score,
      r.total,
      r.correct,
      r.incorrect,
      `${percent.toFixed(2)}%`,
      status
    ]);
  });

  // Adjust column widths
  sheet.columns.forEach((col, i) => {
    col.width = i === 0 ? 20 : 15; // Make USN column wider
    col.alignment = { horizontal: 'left' };
  });

  // Send response
  res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
  res.setHeader('Content-Disposition', `attachment; filename="report_${session.id}.xlsx"`);
  
  await workbook.xlsx.write(res);
  res.end();
}

async function generatePDF(res, session, results, stats) {
  const doc = new PDFDocument({ margin: 50 });

  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('Content-Disposition', `attachment; filename="report_${session.id}.pdf"`);
  
  doc.pipe(res);

  // Helper for drawing lines
  const drawLine = (y) => {
    doc.moveTo(50, y).lineTo(550, y).strokeColor('#e5e7eb').stroke();
  };

  // Header
  doc.fontSize(20).fillColor('#4F46E5').text(`OMR Evaluation Report`, { align: 'center' });
  doc.moveDown(0.5);
  doc.fontSize(14).fillColor('#111827').text(session.subject, { align: 'center' });
  doc.moveDown(1);

  // Metadata & Stats
  doc.fontSize(10).fillColor('#4b5563');
  
  const leftCol = 50;
  const rightCol = 300;
  let currentY = doc.y;

  doc.text(`Session ID: ${session.id}`, leftCol, currentY);
  doc.text(`Date: ${session.exam_date ? new Date(session.exam_date).toLocaleDateString() : 'N/A'}`, rightCol, currentY);
  currentY += 15;
  doc.text(`Section: ${session.section || 'N/A'}`, leftCol, currentY);
  doc.text(`Expected Students: ${stats.expectedStudents}`, rightCol, currentY);
  currentY += 15;
  doc.text(`Total Present: ${stats.totalStudents - stats.absentCount}`, leftCol, currentY);
  doc.text(`Absentees: ${stats.absentCount}`, rightCol, currentY);
  
  currentY += 25;
  doc.fontSize(12).fillColor('#111827').text('Summary Statistics', leftCol, currentY);
  currentY += 15;
  doc.fontSize(10).fillColor('#4b5563');
  
  doc.text(`Class Average: ${stats.classAverage}%`, leftCol, currentY);
  doc.text(`Pass Rate: ${stats.passRate}%`, rightCol, currentY);
  currentY += 15;
  doc.text(`Highest Score: ${stats.highestScore}`, leftCol, currentY);
  doc.text(`Lowest Score: ${stats.lowestScore}`, rightCol, currentY);

  currentY += 30;
  drawLine(currentY);
  currentY += 10;

  // Table Header
  const cols = { usn: 50, score: 180, total: 240, percent: 300, status: 380 };
  
  doc.fontSize(10).fillColor('#111827');
  doc.text('USN', cols.usn, currentY);
  doc.text('Score', cols.score, currentY);
  doc.text('Total', cols.total, currentY);
  doc.text('%', cols.percent, currentY);
  doc.text('Status', cols.status, currentY);
  
  currentY += 15;
  drawLine(currentY);
  currentY += 10;

  // Table Data
  doc.fillColor('#4b5563');
  for (const r of results) {
    if (currentY > 700) {
      doc.addPage();
      currentY = 50;
      // Re-draw header on new page
      doc.fillColor('#111827');
      doc.text('USN', cols.usn, currentY);
      doc.text('Score', cols.score, currentY);
      doc.text('Total', cols.total, currentY);
      doc.text('%', cols.percent, currentY);
      doc.text('Status', cols.status, currentY);
      currentY += 15;
      drawLine(currentY);
      currentY += 10;
      doc.fillColor('#4b5563');
    }

    const percent = Number(r.score_percent);
    const status = percent >= PASS_THRESHOLD ? 'PASS' : 'FAIL';
    
    // Status color
    if (status === 'PASS') doc.fillColor('#059669'); // Green
    else if (status === 'FAIL') doc.fillColor('#DC2626'); // Red
    else doc.fillColor('#6B7280'); // Gray

    doc.text(r.usn, cols.usn, currentY);
    doc.fillColor('#4b5563'); // Reset text color
    
    if (r.status === 'ABSENT') {
      doc.text('-', cols.score, currentY);
      doc.text('-', cols.total, currentY);
      doc.text('ABSENT', cols.percent, currentY);
      doc.fillColor('#6B7280'); // Gray for absent
    } else {
      doc.text(r.score.toString(), cols.score, currentY);
      doc.text(r.total.toString(), cols.total, currentY);
      doc.text(`${percent.toFixed(2)}%`, cols.percent, currentY);
      if (status === 'PASS') doc.fillColor('#059669');
      else doc.fillColor('#DC2626');
    }
    
    doc.text(status, cols.status, currentY);
    doc.fillColor('#4b5563');

    currentY += 20;
  }

  doc.end();
}

export async function getSession(req, res) {
  const { sessionId } = req.params;
  try {
    const sessionRes = await query(`SELECT * FROM exam_sessions WHERE id = $1`, [sessionId]);
    if (sessionRes.rows.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }
    return res.status(200).json(sessionRes.rows[0]);
  } catch (err) {
    console.error('[getSession] error:', err.message);
    return res.status(500).json({ error: 'Failed to retrieve session' });
  }
}

