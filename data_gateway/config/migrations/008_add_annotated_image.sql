-- Vision OMR System — Data Gateway
-- Migration: Add annotated_image TEXT column to student_results table.

BEGIN;

ALTER TABLE student_results
ADD COLUMN IF NOT EXISTS annotated_image TEXT;

COMMIT;
