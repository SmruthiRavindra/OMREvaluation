-- Migration: 011_update_seed_passwords.sql
-- Description: Update seed user passwords to bcrypt hashes for admin2026 and faculty2026.

UPDATE users
SET password_hash = '$2b$12$al/h4eMIjgqHkaFzfH3zdeMXRU/XjvIUx5g6ZgwnZnNXSpm3AWYX6', salt = ''
WHERE username = 'admin';

UPDATE users
SET password_hash = '$2b$12$SMqflLn/ls0JolYvzsogUOWQh7elmf6DrTyY/j5leqPV/BuOdQnjW', salt = ''
WHERE username = 'faculty';
