-- =============================================================================
-- 005_add_missing_columns.sql
-- Vision OMR System — Data Gateway
--
-- Adds columns that were used in code but never created by earlier migrations.
-- Idempotent: safe to run multiple times (uses IF NOT EXISTS / DO blocks).
-- =============================================================================

-- Add expected_students to exam_sessions (used by createSession)
ALTER TABLE exam_sessions ADD COLUMN IF NOT EXISTS expected_students INTEGER NOT NULL DEFAULT 0;

-- Add status to student_results (used by submitStudentResult UPSERT)
ALTER TABLE student_results ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'PRESENT';
