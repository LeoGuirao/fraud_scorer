-- scripts/init.sql
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS cases (
  id            TEXT PRIMARY KEY,
  title         TEXT,
  status        TEXT DEFAULT 'new',
  created_at    TEXT NOT NULL,
  updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  id            TEXT PRIMARY KEY,
  case_id       TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  filename      TEXT,
  filepath      TEXT,
  file_hash     TEXT NOT NULL,
  mime_type     TEXT,
  size_bytes    INTEGER,
  page_count    INTEGER,
  language      TEXT,
  ocr_success   INTEGER DEFAULT 0,
  created_at    TEXT NOT NULL,
  UNIQUE(file_hash, case_id)
);

CREATE TABLE IF NOT EXISTS ocr_results (
  document_id   TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  raw_text      TEXT,
  key_value_pairs JSON,
  tables        JSON,
  entities      JSON,
  confidence    JSON,
  metadata      JSON,
  errors        JSON,
  engine        TEXT,
  engine_version TEXT,
  processed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extracted_data (
  document_id     TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  document_type   TEXT,
  entities        JSON,
  key_value_pairs JSON,
  extra           JSON,
  extractor_version TEXT,
  processed_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id            TEXT PRIMARY KEY,
  case_id       TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  purpose       TEXT,
  llm_model     TEXT,
  params        JSON,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_analyses (
  id              TEXT PRIMARY KEY,
  document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  run_id          TEXT REFERENCES runs(id) ON DELETE SET NULL,
  content_analysis   JSON,
  visual_analysis    JSON,
  contextual_analysis JSON,
  summary         TEXT,
  report_points   JSON,
  alerts          JSON,
  model           TEXT,
  temperature     REAL,
  processed_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_case ON documents(case_id);
