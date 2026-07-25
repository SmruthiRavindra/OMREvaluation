-- =============================================================================
-- 009_add_exam_versions.sql
-- Vision OMR System — Data Gateway
--
-- Adds support for multi-version exam answer keys (A/B/C/D) per session
-- and tracks student exam version in student_results.
-- =============================================================================

BEGIN;

CREATE TABLE IF NOT EXISTS exam_versions (
    id          SERIAL          PRIMARY KEY,
    session_id  VARCHAR(50)     NOT NULL REFERENCES exam_sessions(id) ON DELETE CASCADE,
    version     VARCHAR(10)     NOT NULL, -- 'A', 'B', 'C', 'D' or 'DEFAULT'
    answers     JSONB           NOT NULL, -- { "1": "A", "2": "C", ... }
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_session_version UNIQUE (session_id, version)
);

ALTER TABLE student_results
    ADD COLUMN IF NOT EXISTS version VARCHAR(10) DEFAULT 'DEFAULT';

CREATE INDEX IF NOT EXISTS idx_exam_versions_session
    ON exam_versions (session_id);

COMMIT;
