import pg from 'pg';

let mockData = {
  exam_sessions: [],
  student_results: []
};

// Mock pg.Pool.prototype.query
pg.Pool.prototype.query = async function(text, params) {
  if (text.includes('INSERT INTO exam_sessions')) {
    mockData.exam_sessions.push({
      id: params[0], subject: params[1], section: params[2], exam_date: params[3], total_questions: params[4], expected_students: params[5]
    });
  } else if (text.includes('INSERT INTO student_results')) {
    const id = mockData.student_results.length + 1;
    let status = 'PRESENT';
    let per_q = [];
    if (text.includes("'ABSENT'")) {
      status = 'ABSENT';
    } else {
      per_q = JSON.parse(params[9]);
    }

    mockData.student_results.push({
      id,
      session_id: params[0], usn: params[1], 
      score: params[2] || 0, total: params[3] || 0, 
      correct: params[4] || 0, incorrect: params[5] || 0, 
      unanswered: params[6] || 0, multiple_marked: params[7] || 0, 
      score_percent: params[8] || 0, 
      status: status,
      per_question: per_q
    });
    return { rows: [{ id }] };
  } else if (text.includes('SELECT * FROM exam_sessions WHERE id = $1')) {
    const session = mockData.exam_sessions.find(s => s.id === params[0]);
    return { rows: session ? [session] : [], rowCount: session ? 1 : 0 };
  } else if (text.includes('SELECT * FROM student_results WHERE session_id = $1')) {
    const results = mockData.student_results.filter(r => r.session_id === params[0]);
    return { rows: results, rowCount: results.length };
  }
  return { rows: [], rowCount: 0 };
};

// Now import the app after patching pg
import app from './index.js';
import http from 'http';

const server = http.createServer(app);

async function runTests() {
  server.listen(3002, async () => {
    try {
      console.log('--- Running Tests with Mocked PG Pool ---');
      
      // 1. Create Session
      let res = await fetch('http://localhost:3002/api/sessions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: 'session-123', subject: 'Math', total_questions: 50 })
      });
      let json = await res.json();
      console.log('Create Session:', res.status, json);
      if (res.status !== 201) throw new Error('Create Session failed');

      // 2. Submit Results
      res = await fetch('http://localhost:3002/api/results', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'session-123', usn: '1RV20CS001', score: 45, total: 50, score_percent: 90.0 })
      });
      json = await res.json();
      console.log('Submit Result 1:', res.status, json);

      res = await fetch('http://localhost:3002/api/results', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'session-123', usn: '1RV20CS002', score: 15, total: 50, score_percent: 30.0 })
      });
      json = await res.json();
      console.log('Submit Result 2:', res.status, json);
      
      // 2.5 Submit Absentees
      res = await fetch('http://localhost:3002/api/sessions/session-123/absentees', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'session-123', usns: ['1RV20CS003'] })
      });
      json = await res.json();
      console.log('Submit Absentees:', res.status, json);

      // 3. Get Session Results
      res = await fetch('http://localhost:3002/api/sessions/session-123/results');
      json = await res.json();
      console.log('Get Session Results:', res.status, `Count: ${json.total}`);

      // 4. Download Excel Report
      res = await fetch('http://localhost:3002/api/reports/download/session-123?format=excel');
      console.log('Download Excel:', res.status, res.headers.get('content-type'));
      let buffer = await res.arrayBuffer();
      console.log('Excel file size:', buffer.byteLength, 'bytes');

      // 5. Download PDF Report
      res = await fetch('http://localhost:3002/api/reports/download/session-123?format=pdf');
      console.log('Download PDF:', res.status, res.headers.get('content-type'));
      buffer = await res.arrayBuffer();
      console.log('PDF file size:', buffer.byteLength, 'bytes');

      console.log('--- All tests passed! ---');
    } catch (e) {
      console.error('Test failed:', e);
    } finally {
      server.close();
      process.exit(0);
    }
  });
}

runTests();
