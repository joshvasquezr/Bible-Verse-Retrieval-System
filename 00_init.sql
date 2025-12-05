-- enable pgvector for storing embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- verses table: stores kjv verses w/ word2vec embeddings
CREATE TABLE IF NOT EXISTS verses (
    id SERIAL PRIMARY KEY,
    book VARCHAR(50) NOT NULL,
    chapter INTEGER NOT NULL,
    verse_num INTEGER NOT NULL,
    text TEXT NOT NULL,
    reference VARCHAR(100) NOT NULL UNIQUE,
    primary_theme VARCHAR(50),
    embedding vector(100),  -- word2vec embedding (100 dims)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_verses_reference ON verses(reference);
CREATE INDEX IF NOT EXISTS idx_verses_theme ON verses(primary_theme);
CREATE INDEX IF NOT EXISTS idx_verses_book ON verses(book);

-- index for similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS idx_verses_embedding ON verses
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);