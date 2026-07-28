-- =============================================================================
-- 012_evaluations_unique_session_student.sql
-- Vision OMR System — Data Gateway
--
-- Deduplicates evaluations by (session_id, student_id) keeping the latest ID,
-- and adds a unique constraint on (session_id, student_id) for UPSERT.
-- =============================================================================

BEGIN;

DELETE FROM evaluations a USING evaluations b
WHERE a.id < b.id
  AND a.session_id = b.session_id
  AND a.student_id = b.student_id
  AND a.student_id IS NOT NULL;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'uq_evaluations_session_student'
  ) THEN
    ALTER TABLE evaluations
      ADD CONSTRAINT uq_evaluations_session_student
      UNIQUE (session_id, student_id);
  END IF;
END $$;

COMMIT;
