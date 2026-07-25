-- =============================================================================
-- 006_add_roster_columns.sql
-- Vision OMR System — Data Gateway
--
-- Adds roster and use_roster_order columns to exam_sessions table.
-- =============================================================================

-- Add roster to exam_sessions
ALTER TABLE exam_sessions ADD COLUMN IF NOT EXISTS roster JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Add use_roster_order to exam_sessions
ALTER TABLE exam_sessions ADD COLUMN IF NOT EXISTS use_roster_order BOOLEAN NOT NULL DEFAULT FALSE;
