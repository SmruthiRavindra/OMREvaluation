-- Migration: 007_create_users.sql
-- Description: Create users and sessions tables for college authentication, and seed default faculty and admin accounts.

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    salt VARCHAR(64) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'faculty',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(128) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Seed default accounts
-- Admin: admin / admin2026
INSERT INTO users (username, password_hash, salt, role)
VALUES ('admin', '8323976d44b25ab63f7995808dd58c9dfe2f2f71acaa7bd4ff21c6dce8b909a1a8ea7b7ddc903a12add996bfde1a906751590b16f3a7b5c2d66db42220b9aeb5', '807543f41ce26224c4bbf26a9b9c03cb', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Faculty: faculty / faculty2026
INSERT INTO users (username, password_hash, salt, role)
VALUES ('faculty', '31fb4d6fe7b51ab084dad87d36ebea568c2eb66db16ef1061f89f51df923446a2a9977e15f0ef37e8a040923887a11b4d6b052b7b4e23f68fd0fefef74b57e07', '031c17056b1e0fc31f3f3a8d308b7aeb', 'faculty')
ON CONFLICT (username) DO NOTHING;
