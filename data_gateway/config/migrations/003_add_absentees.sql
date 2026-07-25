-- =============================================================================
-- 003_add_absentees.sql
-- Vision OMR System — Data Gateway
--
-- Adds expected_students to exam_sessions and a status column to student_results
-- to handle absent students.
-- =============================================================================

BEGIN;

ALTER TABLE exam_sessions 
ADD COLUMN IF NOT EXISTS expected_students INTEGER NOT NULL DEFAULT 0;

ALTER TABLE student_results
ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'PRESENT';

COMMIT;
