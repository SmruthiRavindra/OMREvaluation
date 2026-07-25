-- =============================================================================
-- 002_create_sessions_results.sql
-- Vision OMR System — Data Gateway
--
-- Creates the `exam_sessions` and `student_results` tables for tracking
-- exam metadata and per-student score breakdowns.
--
-- Usage:
--   psql -U omr_user -d omr_db -f config/migrations/002_create_sessions_results.sql
-- =============================================================================

BEGIN;

-- ── exam_sessions ────────────────────────────────────────────────────────────
-- Stores metadata about a specific exam session.

CREATE TABLE IF NOT EXISTS exam_sessions (
    id              VARCHAR(50)     PRIMARY KEY,
    subject         VARCHAR(100)    NOT NULL,
    section         VARCHAR(20),
    exam_date       DATE            NOT NULL DEFAULT CURRENT_DATE,
    total_questions INTEGER         NOT NULL DEFAULT 30,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ── student_results ──────────────────────────────────────────────────────────
-- Stores the graded results for a specific student in a specific session.

CREATE TABLE IF NOT EXISTS student_results (
    id              SERIAL          PRIMARY KEY,
    session_id      VARCHAR(50)     REFERENCES exam_sessions(id) ON DELETE CASCADE,
    usn             VARCHAR(20)     NOT NULL,
    score           INTEGER         NOT NULL DEFAULT 0,
    total           INTEGER         NOT NULL DEFAULT 0,
    correct         INTEGER         NOT NULL DEFAULT 0,
    incorrect       INTEGER         NOT NULL DEFAULT 0,
    unanswered      INTEGER         NOT NULL DEFAULT 0,
    multiple_marked INTEGER         NOT NULL DEFAULT 0,
    score_percent   NUMERIC(5,2)    DEFAULT 0.00,
    per_question    JSONB,          -- detailed breakdown per question
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ── Indexes ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_student_results_session
    ON student_results (session_id);

CREATE INDEX IF NOT EXISTS idx_student_results_usn
    ON student_results (usn);

CREATE INDEX IF NOT EXISTS idx_exam_sessions_created
    ON exam_sessions (created_at DESC);

COMMIT;
