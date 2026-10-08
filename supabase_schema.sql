-- ==============================================================================
-- SnapClass AI Attendance System - Supabase PostgreSQL Schema
-- ==============================================================================
-- Run this script in your Supabase SQL Editor:
-- 1. Go to https://app.supabase.com
-- 2. Select your project -> SQL Editor -> New Query
-- 3. Paste this script and click "Run"
-- ==============================================================================

-- 1. TEACHERS TABLE
CREATE TABLE IF NOT EXISTS public.teachers (
    teacher_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. STUDENTS TABLE
CREATE TABLE IF NOT EXISTS public.students (
    student_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL,
    face_embedding JSONB DEFAULT NULL,
    voice_embedding JSONB DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. SUBJECTS TABLE
CREATE TABLE IF NOT EXISTS public.subjects (
    subject_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subject_code TEXT NOT NULL,
    name TEXT NOT NULL,
    section TEXT NOT NULL,
    teacher_id BIGINT REFERENCES public.teachers(teacher_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 4. SUBJECT_STUDENTS (Enrollment Junction Table)
CREATE TABLE IF NOT EXISTS public.subject_students (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id BIGINT REFERENCES public.students(student_id) ON DELETE CASCADE,
    subject_id BIGINT REFERENCES public.subjects(subject_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL,
    CONSTRAINT unique_enrollment UNIQUE (student_id, subject_id)
);

-- 5. ATTENDANCE_LOGS TABLE
CREATE TABLE IF NOT EXISTS public.attendance_logs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_id BIGINT REFERENCES public.students(student_id) ON DELETE CASCADE,
    subject_id BIGINT REFERENCES public.subjects(subject_id) ON DELETE CASCADE,
    timestamp TEXT NOT NULL,
    is_present BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Disable Row Level Security (RLS) for seamless prototype operation
ALTER TABLE public.teachers DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.students DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.subjects DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.subject_students DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.attendance_logs DISABLE ROW LEVEL SECURITY;

-- Grant permissions for anon/authenticated roles
GRANT ALL ON TABLE public.teachers TO anon, authenticated, service_role;
GRANT ALL ON TABLE public.students TO anon, authenticated, service_role;
GRANT ALL ON TABLE public.subjects TO anon, authenticated, service_role;
GRANT ALL ON TABLE public.subject_students TO anon, authenticated, service_role;
GRANT ALL ON TABLE public.attendance_logs TO anon, authenticated, service_role;
