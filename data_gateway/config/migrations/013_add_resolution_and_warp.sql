-- =============================================================================
-- 013_add_resolution_and_warp.sql
-- Vision OMR System — Data Gateway
--
-- Adds image_resolution, warp_status, and is_warped to evaluations table.
-- =============================================================================

BEGIN;

ALTER TABLE evaluations
  ADD COLUMN IF NOT EXISTS image_resolution VARCHAR(30),
  ADD COLUMN IF NOT EXISTS warp_status VARCHAR(30),
  ADD COLUMN IF NOT EXISTS is_warped BOOLEAN DEFAULT FALSE;

COMMIT;
