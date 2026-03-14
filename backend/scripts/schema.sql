-- Supabase Schema for StockSense AI

CREATE TYPE job_status AS ENUM ('pending', 'scraping', 'embedding', 'analyzing', 'generating_report', 'completed', 'failed');
CREATE TYPE doc_source AS ENUM ('BSE', 'NSE', 'SEBI');
CREATE TYPE social_platform AS ENUM ('twitter', 'reddit');

CREATE TABLE IF NOT EXISTS stocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL UNIQUE,
    company_name VARCHAR(255) NOT NULL,
    isin VARCHAR(20),
    sector VARCHAR(100),
    exchange VARCHAR(10) DEFAULT 'NSE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS report_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    status job_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    celery_task_id VARCHAR(255),
    error_msg TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES report_jobs(id) ON DELETE CASCADE,
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    markdown_content TEXT NOT NULL,
    css_score INTEGER,
    sentiment_themes JSONB,
    swot_json JSONB,
    bull_cases_json JSONB,
    bear_cases_json JSONB,
    competitors_json JSONB,
    sources_json JSONB,
    pdf_path VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_as_of TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_objects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    source doc_source NOT NULL,
    doc_type VARCHAR(50) NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    url VARCHAR(500) NOT NULL,
    text TEXT,
    ocr_used BOOLEAN DEFAULT FALSE,
    parse_confidence REAL DEFAULT 1.0,
    chroma_doc_id VARCHAR(255),
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS news_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    headline VARCHAR(500) NOT NULL,
    source VARCHAR(100) NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    url VARCHAR(500) NOT NULL,
    body TEXT,
    sentiment_score REAL,
    sentiment_label VARCHAR(20),
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS social_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID REFERENCES stocks(id) ON DELETE CASCADE,
    platform social_platform NOT NULL,
    post_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    author VARCHAR(100),
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    url VARCHAR(500),
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
