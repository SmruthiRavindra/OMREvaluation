/**
 * authController.js
 * ----------------
 * Authentication controller using Node.js native crypto for password hashing,
 * salting, and session management. Works with the PostgreSQL user schema.
 */

import crypto from 'crypto';
import pool from '../config/database.js';

// ── Passwords Utility Functions ──────────────────────────────────────────────
export function hashPassword(password) {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.pbkdf2Sync(password, salt, 1000, 64, 'sha512').toString('hex');
  return { salt, hash };
}

export function verifyPassword(password, salt, hash) {
  const verifyHash = crypto.pbkdf2Sync(password, salt, 1000, 64, 'sha512').toString('hex');
  return hash === verifyHash;
}

// ── Auth Route Handlers ──────────────────────────────────────────────────────

/**
 * POST /api/auth/login
 * Body: { username, password }
 */
export async function login(req, res) {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required.' });
  }

  try {
    // 1. Fetch user by username
    const userRes = await pool.query(
      'SELECT id, username, password_hash, salt, role FROM users WHERE username = $1',
      [username.trim().toLowerCase()]
    );

    if (userRes.rowCount === 0) {
      return res.status(401).json({ error: 'Invalid username or password.' });
    }

    const user = userRes.rows[0];

    // 2. Verify password hash
    const isCorrect = verifyPassword(password, user.salt, user.password_hash);
    if (!isCorrect) {
      return res.status(401).json({ error: 'Invalid username or password.' });
    }

    // 3. Create session token (64-char secure hex)
    const token = crypto.randomBytes(32).toString('hex');
    const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours expiry

    await pool.query(
      'INSERT INTO user_sessions (user_id, token, expires_at) VALUES ($1, $2, $3)',
      [user.id, token, expiresAt]
    );

    // 4. Return token and user details
    return res.json({
      token,
      user: {
        username: user.username,
        role: user.role
      }
    });
  } catch (err) {
    console.error('[Auth Login Error]', err);
    return res.status(500).json({ error: 'Database error during authentication.' });
  }
}

/**
 * POST /api/auth/logout
 */
export async function logout(req, res) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.json({ success: true, message: 'Already logged out.' });
  }

  const token = authHeader.split(' ')[1];
  try {
    await pool.query('DELETE FROM user_sessions WHERE token = $1', [token]);
    return res.json({ success: true, message: 'Logged out successfully.' });
  } catch (err) {
    console.error('[Auth Logout Error]', err);
    return res.status(500).json({ error: 'Database error during logout.' });
  }
}

// ── Auth Middleware ──────────────────────────────────────────────────────────

export async function verifyAuth(req, res, next) {
  let token = null;
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    token = authHeader.split(' ')[1];
  } else if (req.query.token) {
    token = req.query.token;
  }

  if (!token) {
    return res.status(401).json({ error: 'Unauthorized. Auth token missing.' });
  }
  try {
    // Find active, unexpired session
    const sessionRes = await pool.query(
      `SELECT s.token, u.id as user_id, u.username, u.role, s.expires_at 
       FROM user_sessions s 
       JOIN users u ON s.user_id = u.id 
       WHERE s.token = $1 AND s.expires_at > NOW()`,
      [token]
    );

    if (sessionRes.rowCount === 0) {
      return res.status(401).json({ error: 'Unauthorized. Session expired or invalid.' });
    }

    // Attach user object to request
    const s = sessionRes.rows[0];
    req.user = {
      id: s.user_id,
      username: s.username,
      role: s.role
    };
    return next();
  } catch (err) {
    console.error('[Auth Middleware Error]', err);
    return res.status(500).json({ error: 'Authentication check failed.' });
  }
}

export function isAdmin(req, res, next) {
  if (!req.user || req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Forbidden. Admin privileges required.' });
  }
  return next();
}

// ── User Management (Admin Only) ─────────────────────────────────────────────

/**
 * GET /api/users
 */
export async function listUsers(req, res) {
  try {
    const usersRes = await pool.query(
      'SELECT id, username, role, created_at FROM users ORDER BY created_at DESC'
    );
    return res.json({ users: usersRes.rows });
  } catch (err) {
    console.error('[Auth listUsers Error]', err);
    return res.status(500).json({ error: 'Failed to retrieve users.' });
  }
}

/**
 * POST /api/users
 * Body: { username, password, role }
 */
export async function createUser(req, res) {
  const { username, password, role } = req.body;

  if (!username || !password || !role) {
    return res.status(400).json({ error: 'Username, password, and role are required.' });
  }

  const normalizedUser = username.trim().toLowerCase();
  const normalizedRole = role.trim().toLowerCase();

  if (normalizedRole !== 'admin' && normalizedRole !== 'faculty') {
    return res.status(400).json({ error: 'Role must be either "admin" or "faculty".' });
  }

  try {
    // Check if user already exists
    const checkRes = await pool.query('SELECT 1 FROM users WHERE username = $1', [normalizedUser]);
    if (checkRes.rowCount > 0) {
      return res.status(400).json({ error: 'Username already exists.' });
    }

    // Hash password and insert
    const { salt, hash } = hashPassword(password);
    const result = await pool.query(
      'INSERT INTO users (username, password_hash, salt, role) VALUES ($1, $2, $3, $4) RETURNING id, username, role, created_at',
      [normalizedUser, hash, salt, normalizedRole]
    );

    return res.status(201).json({
      success: true,
      user: result.rows[0]
    });
  } catch (err) {
    console.error('[Auth createUser Error]', err);
    return res.status(500).json({ error: 'Failed to create user.' });
  }
}

/**
 * DELETE /api/users/:id
 */
export async function deleteUser(req, res) {
  const userId = Number(req.params.id);

  if (isNaN(userId)) {
    return res.status(400).json({ error: 'Invalid user ID.' });
  }

  // Prevent admin from deleting themselves
  if (req.user.id === userId) {
    return res.status(400).json({ error: 'Cannot delete your own administrator account.' });
  }

  try {
    const delRes = await pool.query('DELETE FROM users WHERE id = $1 RETURNING username', [userId]);
    if (delRes.rowCount === 0) {
      return res.status(404).json({ error: 'User not found.' });
    }
    return res.json({
      success: true,
      message: `User "${delRes.rows[0].username}" deleted successfully.`
    });
  } catch (err) {
    console.error('[Auth deleteUser Error]', err);
    return res.status(500).json({ error: 'Failed to delete user.' });
  }
}

/**
 * GET /api/admin/stats
 */
export async function getAdminStats(req, res) {
  try {
    const sessionsRes = await pool.query('SELECT COUNT(*) FROM exam_sessions');
    const resultsRes = await pool.query("SELECT COUNT(*) FROM student_results WHERE status = 'PRESENT'");
    const usersRes = await pool.query('SELECT COUNT(*) FROM users');
    const avgRes = await pool.query("SELECT AVG(score_percent) FROM student_results WHERE status = 'PRESENT'");

    return res.json({
      sessionsCount: parseInt(sessionsRes.rows[0].count || 0),
      resultsCount: parseInt(resultsRes.rows[0].count || 0),
      usersCount: parseInt(usersRes.rows[0].count || 0),
      averageScore: parseFloat(avgRes.rows[0].avg || 0).toFixed(1)
    });
  } catch (err) {
    console.error('[Auth getAdminStats Error]', err);
    return res.status(500).json({ error: 'Failed to retrieve admin statistics.' });
  }
}
