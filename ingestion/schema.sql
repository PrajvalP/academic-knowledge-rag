-- Database schema for academic-knowledge-rag.
-- Safe to run more than once (IF NOT EXISTS everywhere).

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source document (PDF).
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    file          TEXT UNIQUE NOT NULL,          -- e.g. tari29_syllabus_2025.pdf
    title         TEXT NOT NULL,
    doc_type      TEXT NOT NULL,                 -- syllabus | policy | internal
    course_code   TEXT,                          -- NULL for policies
    school        TEXT,
    year          INT,
    valid_from    DATE,                          -- from the document text
    access_level  TEXT NOT NULL DEFAULT 'public'
                  CHECK (access_level IN ('public', 'student', 'professor', 'admin')),
    source_url    TEXT,
    superseded_by INT REFERENCES documents(id)   -- filled in Stage 7
);

-- One row per chunk; many chunks belong to one document.
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id    TEXT UNIQUE NOT NULL,            -- e.g. tari29_syllabus_2025#004
    section     TEXT,
    page_start  INT,
    page_end    INT,
    text        TEXT NOT NULL,                   -- shown to the user
    embed_text  TEXT NOT NULL,                   -- text + context header
    num_words   INT,
    embedding   VECTOR(1024),                    -- filled in Stage 2 (bge-m3 = 1024 numbers)
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx      ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- Added in Stage 2 (created by embed.py after the vectors exist):
-- CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Added in Stage 6: users who can log in to the API.
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,                 -- bcrypt hash, never the password
    role          TEXT NOT NULL
                  CHECK (role IN ('student', 'professor', 'admin'))
);
