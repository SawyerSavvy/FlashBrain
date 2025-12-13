-- RAG Service Supabase Vector Store Migration
-- Run this in Supabase SQL Editor to set up pgvector for document storage and semantic search

-- ============================================================
-- 1. Enable pgvector Extension
-- ============================================================
-- Create the pgvector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 2. Documents Table
-- ============================================================
-- Stores metadata about uploaded documents

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    source TEXT NOT NULL, -- URL, file path, or source identifier
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'text', 'url', 'file')),
    metadata JSONB DEFAULT '{}'::jsonb, -- Additional metadata (author, date, etc.)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    client_id UUID, -- Optional: for multi-tenant support
    project_id UUID, -- Optional: for project-specific documents
    organization_id UUID -- Optional: for organization-specific documents
);

-- Indexes for efficient document querying
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_client_id ON documents(client_id);
CREATE INDEX IF NOT EXISTS idx_documents_project_id ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_documents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at_trigger
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_documents_updated_at();

-- ============================================================
-- 3. Document Chunks Table (with pgvector)
-- ============================================================
-- Stores document chunks with embeddings for semantic search

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL, -- Order of chunk within document
    content TEXT NOT NULL,
    embedding vector(768), -- Google embedding-001 uses 768 dimensions
    metadata JSONB DEFAULT '{}'::jsonb, -- Page number, section, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_document_chunk UNIQUE(document_id, chunk_index)
);

-- Indexes for efficient chunk retrieval
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_chunk_index ON document_chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_document_chunks_created_at ON document_chunks(created_at);

-- Vector similarity search index (HNSW for fast approximate nearest neighbor search)
-- This index enables fast semantic similarity queries
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- 4. Helper Functions for Vector Search
-- ============================================================

-- Function to perform similarity search
-- Returns chunks ordered by cosine similarity to query embedding
CREATE OR REPLACE FUNCTION search_document_chunks(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 10,
    filter_document_id UUID DEFAULT NULL,
    filter_client_id UUID DEFAULT NULL,
    filter_project_id UUID DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    FROM document_chunks dc
    JOIN documents d ON d.id = dc.document_id
    WHERE
        dc.embedding IS NOT NULL
        AND (1 - (dc.embedding <=> query_embedding)) >= match_threshold
        AND (filter_document_id IS NULL OR dc.document_id = filter_document_id)
        AND (filter_client_id IS NULL OR d.client_id = filter_client_id)
        AND (filter_project_id IS NULL OR d.project_id = filter_project_id)
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get document with chunk count
CREATE OR REPLACE FUNCTION get_document_with_chunk_count(document_uuid UUID)
RETURNS TABLE (
    id UUID,
    title TEXT,
    source TEXT,
    source_type TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    client_id UUID,
    project_id UUID,
    chunk_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.title,
        d.source,
        d.source_type,
        d.metadata,
        d.created_at,
        d.updated_at,
        d.client_id,
        d.project_id,
        COUNT(dc.id) as chunk_count
    FROM documents d
    LEFT JOIN document_chunks dc ON dc.document_id = d.id
    WHERE d.id = document_uuid
    GROUP BY d.id, d.title, d.source, d.source_type, d.metadata, d.created_at, d.updated_at, d.client_id, d.project_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 5. Row Level Security (RLS) Policies
-- ============================================================
-- Enable RLS for security (optional, adjust based on your needs)

-- Uncomment if you want RLS enabled:
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Policy examples (adjust based on your auth requirements):
-- DROP POLICY IF EXISTS documents_client_policy ON documents;
-- CREATE POLICY documents_client_policy ON documents
--     FOR ALL
--     USING (client_id = auth.uid()::text);

-- Service role bypass (for backend operations)
-- DROP POLICY IF EXISTS documents_service_role_policy ON documents;
-- CREATE POLICY documents_service_role_policy ON documents
--     FOR ALL
--     TO service_role
--     USING (true)
--     WITH CHECK (true);

-- ============================================================
-- 6. Verification Queries
-- ============================================================

-- Check extension is enabled
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check tables were created
SELECT tablename FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN ('documents', 'document_chunks');

-- Check indexes
SELECT indexname FROM pg_indexes
WHERE schemaname = 'public'
AND tablename IN ('documents', 'document_chunks');

-- Check vector index exists
SELECT indexname, indexdef FROM pg_indexes
WHERE schemaname = 'public'
AND tablename = 'document_chunks'
AND indexname = 'idx_document_chunks_embedding';

