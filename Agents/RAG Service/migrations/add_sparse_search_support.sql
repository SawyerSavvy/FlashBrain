-- Enable extensions helpful for lexical search (if not already present)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Add a generated tsvector column for full-text (sparse) search over chunks
ALTER TABLE document_chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;

-- Index the tsvector column for fast lexical search
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsv
  ON document_chunks
  USING gin (content_tsv);

-- Optional: store a normalized lexical length metric to aid hybrid scoring
-- (document length in tokens). Can be used to normalize ts_rank outputs.
ALTER TABLE document_chunks
  ADD COLUMN IF NOT EXISTS lexical_token_count integer
  GENERATED ALWAYS AS (array_length(string_to_array(coalesce(content, ''), ' '), 1)) STORED;

COMMENT ON COLUMN document_chunks.content_tsv IS 'Sparse lexical representation for full-text search (English).';
COMMENT ON COLUMN document_chunks.lexical_token_count IS 'Approximate token count for length normalization in hybrid scoring.';
