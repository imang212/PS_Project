-- Create user (cluster-level)
DO
$$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'hailo_user') THEN
        CREATE USER hailo_user WITH PASSWORD 'hailo_pass';
    END IF;
END
$$;

-- Create DB only if missing
SELECT 'CREATE DATABASE hailo_db OWNER hailo_user'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'hailo_db'
)\gexec


-- Connect to hailo_db
\connect hailo_db;

-- Tables
CREATE TABLE IF NOT EXISTS detections (
    id BIGSERIAL PRIMARY KEY,  -- BIGSERIAL for high-volume data
    timestamp_ TIMESTAMPTZ NOT NULL,
    frame_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    x1 FLOAT NOT NULL CHECK (x1 >= 0),
    y1 FLOAT NOT NULL CHECK (y1 >= 0),
    x2 FLOAT NOT NULL CHECK (x2 >= 0),
    y2 FLOAT NOT NULL CHECK (y2 >= 0),
    track_id INT,
    CHECK (x2 > x1 AND y2 > y1)  
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_det_time  ON detections(timestamp_);
CREATE INDEX IF NOT EXISTS idx_det_timestamp_desc ON detections(timestamp_ DESC);  -- For recent detections queries
CREATE INDEX IF NOT EXISTS idx_det_class_timestamp_desc ON detections(class_name, timestamp_ DESC);  -- For class-specific time queries
CREATE INDEX IF NOT EXISTS idx_det_class ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_det_track ON detections(track_id);
CREATE INDEX IF NOT EXISTS idx_det_class_time ON detections(class_name, timestamp_);
CREATE INDEX IF NOT EXISTS idx_det_track_time ON detections(track_id, timestamp_);

-- Grant privileges
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hailo_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hailo_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO hailo_user;

