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
  const { id, subject, section, exam_date, total_questions, roster, use_roster_order } = req.body;

  if (!id || !subject) {
    return res.status(400).json({ error: 'Missing required fields: id, subject' });
  }

  try {
    await query(
      `INSERT INTO exam_sessions (id, subject, section, exam_date, total_questions, expected_students, roster, use_roster_order, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
       ON CONFLICT (id) DO UPDATE SET
         subject = EXCLUDED.subject,
         section = EXCLUDED.section,
         expected_students = EXCLUDED.expected_students,
         roster = EXCLUDED.roster,
         use_roster_order = EXCLUDED.use_roster_order`,
      [
        id,
        subject,
        section || null,
        exam_date || new Date(),
        total_questions || 30,
        req.body.expected_students || 0,
        JSON.stringify(roster || []),
        use_roster_order || false
      ]
    );
    return res.status(201).json({ message: 'Session initialized/updated successfully', id });
  } catch (err) {
    console.error('[createSession] error:', err.message);
    return res.status(500).json({ error: 'Failed to initialize session' });
  }
}

export async function listSessions(req, res) {
  try {
    const result = await query(
      `SELECT id, subject, section, exam_date, total_questions, expected_students, roster, use_roster_order, created_at 
       FROM exam_sessions 
       ORDER BY created_at DESC`
    );
    return res.json({ rows: result.rows });
  } catch (err) {
    console.error('[listSessions] error:', err.message);
    return res.status(500).json({ error: 'Failed to retrieve sessions' });
  }
}

// ── Student Results ────────────────────────────────────────────────────────

export async function submitStudentResult(req, res) {
  const {
    session_id,
    usn,
    version = 'DEFAULT',
    score = 0,
    total = 0,
    correct = 0,
    incorrect = 0,
    unanswered = 0,
    multiple_marked = 0,
    score_percent = 0.00,
    per_question = [],
    annotated_image = null
  } = req.body;

  if (!session_id || !usn) {
    return res.status(400).json({ error: 'Missing required fields: session_id, usn' });
  }

  const cleanVersion = (version || 'DEFAULT').toUpperCase();

  try {
    // Enforce expected students count limit check
    const sessionRes = await query("SELECT expected_students FROM exam_sessions WHERE id = $1", [session_id]);
    if (sessionRes.rows.length > 0) {
      const expected = parseInt(sessionRes.rows[0].expected_students, 10) || 0;
      if (expected > 0) {
        // Count other students already stored for this session
        const countRes = await query(
          "SELECT COUNT(*) as count FROM student_results WHERE session_id = $1 AND usn != $2",
          [session_id, usn]
        );
        const currentCount = parseInt(countRes.rows[0].count, 10) || 0;
        if (currentCount >= expected) {
          return res.status(400).json({
            error: `Session limit reached. Cannot add more than ${expected} student results.`
          });
        }
      }
    }

    const result = await query(
      `INSERT INTO student_results 
        (session_id, usn, version, score, total, correct, incorrect, unanswered, multiple_marked, score_percent, per_question, annotated_image, status, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'PRESENT', NOW())
       ON CONFLICT (session_id, usn)
       DO UPDATE SET
         version = EXCLUDED.version,
         score = EXCLUDED.score,
         total = EXCLUDED.total,
         correct = EXCLUDED.correct,
         incorrect = EXCLUDED.incorrect,
         unanswered = EXCLUDED.unanswered,
         multiple_marked = EXCLUDED.multiple_marked,
         score_percent = EXCLUDED.score_percent,
         per_question = EXCLUDED.per_question,
         annotated_image = EXCLUDED.annotated_image,
         status = 'PRESENT',
         created_at = NOW()
       RETURNING id`,
      [session_id, usn, cleanVersion, score, total, correct, incorrect, unanswered, multiple_marked, score_percent, JSON.stringify(per_question), annotated_image]
    );
    return res.status(201).json({ id: result.rows[0].id, saved: true, version: cleanVersion });
  } catch (err) {
    console.error('[submitStudentResult] error:', err.message);
    return res.status(500).json({ error: 'Failed to persist student result' });
  }
}

export async function saveExamVersion(req, res) {
  const { sessionId } = req.params;
  const { version = 'DEFAULT', answers = {} } = req.body;

  if (!sessionId) {
    return res.status(400).json({ error: 'Missing session ID' });
  }

  const verName = (version || 'DEFAULT').toUpperCase();

  try {
    await query(
      `INSERT INTO exam_versions (session_id, version, answers, created_at)
       VALUES ($1, $2, $3, NOW())
       ON CONFLICT (session_id, version)
       DO UPDATE SET
         answers = EXCLUDED.answers,
         created_at = NOW()`,
      [sessionId, verName, JSON.stringify(answers)]
    );
    return res.status(201).json({ message: `Version ${verName} answer key saved`, version: verName });
  } catch (err) {
    console.error('[saveExamVersion] error:', err.message);
    return res.status(500).json({ error: 'Failed to save exam version answer key' });
  }
}

export async function listExamVersions(req, res) {
  const { sessionId } = req.params;
  try {
    const result = await query(
      `SELECT version, answers, created_at FROM exam_versions WHERE session_id = $1 ORDER BY version ASC`,
      [sessionId]
    );
    return res.json({ rows: result.rows });
  } catch (err) {
    console.error('[listExamVersions] error:', err.message);
    return res.status(500).json({ error: 'Failed to retrieve exam versions' });
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
         VALUES ($1, $2, 0, 0, 0, 0, 0, 0, 0.00, 'ABSENT', NOW())
         ON CONFLICT (session_id, usn)
         DO UPDATE SET
           score = 0,
           total = 0,
           correct = 0,
           incorrect = 0,
           unanswered = 0,
           multiple_marked = 0,
           score_percent = 0.00,
           status = 'ABSENT',
           per_question = NULL,
           created_at = NOW()`,
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
      return await generatePDF(res, session, results, stats, req.query.detail === 'true');
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

async function generatePDF(res, session, results, stats, detail = false) {
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
  const cols = { usn: 50, correct: 160, incorrect: 205, unanswered: 250, multiple: 295, score: 340, total: 385, percent: 430, status: 495 };
  
  doc.fontSize(9).fillColor('#111827');
  doc.text('USN', cols.usn, currentY);
  doc.text('Correct', cols.correct, currentY);
  doc.text('Wrong', cols.incorrect, currentY);
  doc.text('Blank', cols.unanswered, currentY);
  doc.text('Multi', cols.multiple, currentY);
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
      doc.text('Correct', cols.correct, currentY);
      doc.text('Wrong', cols.incorrect, currentY);
      doc.text('Blank', cols.unanswered, currentY);
      doc.text('Multi', cols.multiple, currentY);
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
    if (r.status === 'ABSENT') {
      doc.fillColor('#6B7280'); // Gray
    } else if (status === 'PASS') {
      doc.fillColor('#059669'); // Green
    } else {
      doc.fillColor('#DC2626'); // Red
    }

    doc.text(r.usn, cols.usn, currentY);
    doc.fillColor('#4b5563'); // Reset text color
    
    if (r.status === 'ABSENT') {
      doc.text('-', cols.correct, currentY);
      doc.text('-', cols.incorrect, currentY);
      doc.text('-', cols.unanswered, currentY);
      doc.text('-', cols.multiple, currentY);
      doc.text('-', cols.score, currentY);
      doc.text('-', cols.total, currentY);
      doc.text('ABSENT', cols.percent, currentY);
    } else {
      doc.text((r.correct ?? 0).toString(), cols.correct, currentY);
      doc.text((r.incorrect ?? 0).toString(), cols.incorrect, currentY);
      doc.text((r.unanswered ?? 0).toString(), cols.unanswered, currentY);
      doc.text((r.multiple_marked ?? 0).toString(), cols.multiple, currentY);
      doc.text(r.score.toString(), cols.score, currentY);
      doc.text(r.total.toString(), cols.total, currentY);
      doc.text(`${percent.toFixed(2)}%`, cols.percent, currentY);
      if (status === 'PASS') doc.fillColor('#059669');
      else doc.fillColor('#DC2626');
    }
    
    doc.text(r.status === 'ABSENT' ? 'ABSENT' : status, cols.status, currentY);
    doc.fillColor('#4b5563');

    currentY += 20;
  }

  // Detailed per-student section with OMR image
  if (detail) {
    for (const r of results) {
      if (r.status === 'ABSENT') continue;

      doc.addPage();
      
      // Page Header
      doc.fontSize(16).fillColor('#4F46E5').text(`Student OMR Sheet Breakdown`, 50, 40);
      doc.fontSize(8).fillColor('#9ca3af').text(`Generated: ${new Date().toLocaleString()}`, 50, 60);
      doc.moveTo(50, 70).lineTo(550, 70).strokeColor('#e5e7eb').stroke();

      // Student info block
      doc.fontSize(10).fillColor('#111827');
      doc.text(`USN / Candidate ID:`, 50, 85);
      doc.font('Helvetica-Bold').text(r.usn, 180, 85);
      doc.font('Helvetica');

      doc.text(`Score:`, 50, 100);
      const scorePct = Number(r.score_percent);
      const statusColor = scorePct >= PASS_THRESHOLD ? '#059669' : '#DC2626';
      doc.fillColor(statusColor).font('Helvetica-Bold').text(`${r.score} / ${r.total} (${scorePct.toFixed(2)}%)`, 180, 100);
      doc.fillColor('#111827').font('Helvetica');
      
      // Table Header for questions
      doc.fontSize(11).font('Helvetica-Bold').text(`Grading Details`, 50, 125);
      doc.moveTo(50, 137).lineTo(550, 137).strokeColor('#e5e7eb').stroke();

      const col1X = 50;
      const col2X = 300;
      const startY = 155;
      
      doc.fontSize(8).fillColor('#9ca3af');
      doc.text("Q#    Marked      Correct      Status", col1X, startY - 12);
      doc.text("Q#    Marked      Correct      Status", col2X, startY - 12);
      doc.moveTo(col1X, startY - 5).lineTo(col1X + 220, startY - 5).stroke();
      doc.moveTo(col2X, startY - 5).lineTo(col2X + 220, startY - 5).stroke();

      const perQ = r.per_question || [];
      const totalQ = r.total || 30;

      for (let i = 0; i < totalQ; i++) {
        const q = perQ[i];
        const isSecondCol = i >= Math.ceil(totalQ / 2);
        const xOffset = isSecondCol ? col2X : col1X;
        const yOffset = startY + (i % Math.ceil(totalQ / 2)) * 12;

        doc.fontSize(9).fillColor('#4b5563');
        const qNum = i + 1;

        if (q) {
          doc.text(`${qNum}`, xOffset, yOffset);
          
          const marked = q.marked_options && q.marked_options.length > 0 ? q.marked_options.join(",") : "—";
          doc.text(`${marked}`, xOffset + 25, yOffset);

          const correctOpt = q.correct_option || "—";
          doc.text(`${correctOpt}`, xOffset + 70, yOffset);

          if (q.status === 'correct') {
            doc.fillColor('#059669').font('Helvetica-Bold').text("PASS", xOffset + 115, yOffset);
          } else if (q.status === 'incorrect' || q.status === 'multiple_marked' || q.status === 'ambiguous') {
            doc.fillColor('#DC2626').font('Helvetica-Bold').text("FAIL", xOffset + 115, yOffset);
          } else {
            doc.fillColor('#6b7280').font('Helvetica').text("BLANK", xOffset + 115, yOffset);
          }
          doc.font('Helvetica');
        } else {
          doc.text(`${qNum}`, xOffset, yOffset);
          doc.text("—", xOffset + 25, yOffset);
          doc.text("—", xOffset + 70, yOffset);
          doc.fillColor('#6b7280').text("BLANK", xOffset + 115, yOffset);
        }
      }

      // Add annotated image page if available
      if (r.annotated_image) {
        try {
          doc.addPage();
          
          doc.fontSize(14).fillColor('#4F46E5').text(`Annotated OMR Scan — ${r.usn}`, 50, 40);
          doc.moveTo(50, 55).lineTo(550, 55).strokeColor('#e5e7eb').stroke();

          const base64Data = r.annotated_image.replace(/^data:image\/\w+;base64,/, '');
          const imgBuffer = Buffer.from(base64Data, 'base64');
          
          doc.image(imgBuffer, 50, 75, { width: 500, height: 680 });
        } catch (imgErr) {
          console.error(`Failed to embed OMR image in PDF for ${r.usn}:`, imgErr.message);
          doc.fontSize(12).fillColor('#DC2626').text(`[Error rendering OMR sheet scan image: ${imgErr.message}]`, 50, 100);
        }
      }
    }
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

export async function getPendingCount(req, res) {
  const { sessionId } = req.params;
  try {
    const sessionRes = await query(`SELECT * FROM exam_sessions WHERE id = $1`, [sessionId]);
    if (sessionRes.rowCount === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }
    const session = sessionRes.rows[0];
    const resultsCountRes = await query(
      `SELECT COUNT(*) as count FROM student_results WHERE session_id = $1`,
      [sessionId]
    );
    const submittedCount = parseInt(resultsCountRes.rows[0].count, 10);
    const expected = parseInt(session.expected_students, 10) || 0;
    const pending = Math.max(0, expected - submittedCount);
    
    return res.json({
      session_id: sessionId,
      expected_students: expected,
      submitted_students: submittedCount,
      pending_students: pending
    });
  } catch (err) {
    console.error('[getPendingCount] error:', err.message);
    return res.status(500).json({ error: 'Failed to fetch pending count' });
  }
}

export async function getNextRosterUsn(req, res) {
  const { sessionId } = req.params;
  try {
    const sRes = await query("SELECT roster, use_roster_order FROM exam_sessions WHERE id = $1", [sessionId]);
    if (sRes.rows.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }
    const { roster } = sRes.rows[0];
    if (!roster) {
      return res.json({ next_usn: null });
    }
    const rosterList = typeof roster === 'string' ? JSON.parse(roster) : roster;
    if (!Array.isArray(rosterList) || rosterList.length === 0) {
      return res.json({ next_usn: null });
    }
    
    // Fetch absentees
    const absRes = await query("SELECT usn FROM student_results WHERE session_id = $1 AND status = 'ABSENT'", [sessionId]);
    const absentees = absRes.rows.map(r => r.usn);
    const activeRoster = rosterList.filter(u => !absentees.includes(u));
    
    // Fetch count of already saved (non-absent) students
    const countRes = await query("SELECT COUNT(*) as count FROM student_results WHERE session_id = $1 AND status != 'ABSENT'", [sessionId]);
    const savedCount = parseInt(countRes.rows[0]?.count || 0);
    
    const nextUsn = savedCount < activeRoster.length ? activeRoster[savedCount] : null;
    return res.json({ next_usn: nextUsn });
  } catch (err) {
    console.error('[getNextRosterUsn] error:', err.message);
    return res.status(500).json({ error: 'Failed to fetch next USN' });
  }
}

export async function deleteSession(req, res) {
  const { sessionId } = req.params;
  try {
    const result = await query("DELETE FROM exam_sessions WHERE id = $1 RETURNING id", [sessionId]);
    return res.json({ 
      message: 'Session deleted successfully.', 
      id: sessionId,
      alreadyDeleted: result.rows.length === 0 
    });
  } catch (err) {
    console.error('[deleteSession] error:', err.message);
    return res.status(500).json({ error: 'Failed to delete session.' });
  }
}

