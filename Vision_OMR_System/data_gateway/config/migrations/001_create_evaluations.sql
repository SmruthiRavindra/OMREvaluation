-- =============================================================================
-- 001_create_evaluations.sql
-- Vision OMR System — Data Gateway
--
-- Creates the core `evaluations` table used by evaluationController.js
-- to persist confirmed grading results.
--
-- Usage:
--   psql -U omr_user -d omr_db -f config/migrations/001_create_evaluations.sql
-- =============================================================================

BEGIN;

-- ── Main table ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS evaluations (
    id                  SERIAL          PRIMARY KEY,
    student_id          VARCHAR(20),                        -- USN / roll number
    session_id          VARCHAR(50),                        -- exam session identifier
    filled_count        INTEGER         NOT NULL DEFAULT 0,
    empty_count         INTEGER         NOT NULL DEFAULT 0,
    ambiguous_count     INTEGER         NOT NULL DEFAULT 0,
    needs_manual_review BOOLEAN         NOT NULL DEFAULT FALSE,
    bubbles             JSONB,                              -- full per-bubble detail array
    processing_time_ms  INTEGER         DEFAULT 0,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ── Indexes ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_evaluations_student
    ON evaluations (student_id);

CREATE INDEX IF NOT EXISTS idx_evaluations_session
    ON evaluations (session_id);

CREATE INDEX IF NOT EXISTS idx_evaluations_created
    ON evaluations (created_at DESC);

COMMIT;
