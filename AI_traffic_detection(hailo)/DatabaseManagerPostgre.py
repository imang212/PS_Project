from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_batch

class PostgreDatabaseManager:
    """
    PostgreSQL database manager with connection pooling.    
    Features:
    - Connection pooling for efficient resource usage
    - Batch inserts for high-throughput scenarios
    - Automatic transaction management
    - Optimized indexes for fast queries
    """   
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize PostgreSQL connection pool. 
        Args:
            db_config: Database configuration dictionary
                'host': 'localhost',
                'port': 5432,
                'database': 'traffic_db',
                'user': 'postgres',
                'password': 'your_password',
                'min_connections': 1,
                'max_connections': 10
            }
        """
        self.db_config = db_config or {}
        self.pool = self._create_pool()
        self.init_database()
        print("[DatabaseManager] PostgreSQL initialized")
    
    def _create_pool(self) -> ThreadedConnectionPool:
        """Create PostgreSQL connection pool."""
        try:
            pool = ThreadedConnectionPool(
                minconn=self.db_config.get('min_connections', 1),
                maxconn=self.db_config.get('max_connections', 10),
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'traffic_db'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', '')
            )
            print("[DatabaseManager] Connection pool created")
            return pool
        except Exception as e:
            print(f"[DatabaseManager] Failed to create pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool."""
        return self.pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool."""
        if conn: self.pool.putconn(conn)
    
    def init_database(self):
        """Create database tables and indexes."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Create detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    frame_id INTEGER NOT NULL,
                    label VARCHAR(50) NOT NULL,
                    confidence REAL NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create indexes
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_detections_track_id ON detections(track_id)''')
            # Commit changes
            conn.commit()
            print("[DatabaseManager] Schema initialized")
        except Exception as e:
            print(f"[DatabaseManager] Error initializing database: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def insert_detection(self, detection_data: Dict[str, Any]):
        """Insert single detection."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, frame_id, label, confidence, x1, y1, x2, y2, track_id, total_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (detection_data['timestamp'], detection_data['frame_id'], detection_data['label'], detection_data['confidence'],
                detection_data['x1'], detection_data['y1'], detection_data['x2'], detection_data['y2'], detection_data['track_id'], 
                detection_data['total_count']
            ))
            conn.commit()
        except Exception as e:
            print(f"[DatabaseManager] Error inserting detection: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def insert_batch_detections(self, detections: List[Dict[str, Any]]):
        """Insert multiple detections in batch."""
        if not detections: 
            return
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            query = '''
                INSERT INTO detections 
                (timestamp, frame_id, label, confidence, x1, y1, x2, y2, track_id, total_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            data = [
                (d['timestamp'], d['frame_id'], d['label'], d['confidence'],
                 d['x1'], d['y1'], d['x2'], d['y2'], d['track_id'], d['total_count'])
                for d in detections
            ]
            execute_batch(cursor, query, data, page_size=100)
            conn.commit()
        except Exception as e:
            print(f"[DatabaseManager] Error in batch insert: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def get_recent_detections(self, minutes: int = 5, limit: int = 100):
        """Retrieve recent detections."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT timestamp, frame_id, label, confidence, x1, y1, x2, y2, track_id, total_count
                FROM detections
                WHERE timestamp > NOW() - INTERVAL '%s minutes'
                ORDER BY timestamp DESC
                LIMIT %s
            ''', (minutes, limit))
            columns = ['timestamp', 'frame_id', 'label', 'confidence', 
                      'x1', 'y1', 'x2', 'y2', 'track_id', 'total_count']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def get_statistics(self, hours: int = 1):
        """Get aggregated statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT label, COUNT(DISTINCT track_id) as count, AVG(confidence) as avg_confidence
                FROM detections
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                GROUP BY label
            ''', (hours,))
            return {row[0]: {'count': row[1], 'avg_confidence': row[2]} for row in cursor.fetchall()}
        finally:
            cursor.close()
            self.return_connection(conn)

    def close(self):
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()
            print("[DatabaseManager] Connection pool closed")