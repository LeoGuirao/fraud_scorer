-- Extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Tabla principal de siniestros
CREATE TABLE IF NOT EXISTS siniestros (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    numero_siniestro VARCHAR(100) UNIQUE NOT NULL,
    fecha_creacion TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    estado VARCHAR(50) DEFAULT 'pendiente',
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Tabla de documentos
CREATE TABLE IF NOT EXISTS documentos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    siniestro_id UUID REFERENCES siniestros(id) ON DELETE CASCADE,
    tipo_documento VARCHAR(50) NOT NULL,
    nombre_archivo VARCHAR(255) NOT NULL,
    url_s3 TEXT NOT NULL,
    texto_raw TEXT,
    datos_extraidos JSONB DEFAULT '{}'::jsonb,
    fecha_carga TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de análisis de fraude
CREATE TABLE IF NOT EXISTS analisis_fraude (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    siniestro_id UUID REFERENCES siniestros(id) ON DELETE CASCADE,
    score_fraude DECIMAL(3,2) CHECK (score_fraude >= 0 AND score_fraude <= 1),
    reglas_activadas JSONB DEFAULT '[]'::jsonb,
    explicacion TEXT,
    fecha_analisis TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Índices para performance
CREATE INDEX idx_siniestros_estado ON siniestros(estado);
CREATE INDEX idx_documentos_siniestro ON documentos(siniestro_id);
CREATE INDEX idx_analisis_siniestro ON analisis_fraude(siniestro_id);
