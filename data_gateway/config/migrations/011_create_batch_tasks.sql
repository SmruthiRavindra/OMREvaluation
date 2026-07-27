-- Migration: 012_create_batch_tasks.sql
-- Description: Create batch_tasks table to persist background batch progress across restarts.

CREATE TABLE IF NOT EXISTS batch_tasks (
    task_id VARCHAR(64) PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'processing',
    total_sheets INT NOT NULL DEFAULT 0,
    processed_sheets INT NOT NULL DEFAULT 0,
    results JSONB DEFAULT '[]'::jsonb,
    errors JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_batch_tasks_created ON batch_tasks (created_at DESC);
