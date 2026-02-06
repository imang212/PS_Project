from typing import List, Dict, Any, Optional, Callable
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_batch, RealDictCursor
import time
from datetime import datetime
import threading
import json
from MQTTClient import MQTTPublisher
from collections import deque
    
class PostgreDatabaseManager:
    """
    PostgreSQL database manager with connection pooling and MQTT integration.  
    Features:
    Connection pooling for efficient resource usage
    - Batch inserts for high-throughput scenarios
    - Automatic transaction management
    - Connection testing and health checks
    - MQTT message listener with buffering
    - Optimized indexes for fast queries
    """   
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize PostgreSQL connection pool. 
        Args:
            db_config: Database configuration dictionary
            {
                'host': '192.168.37.56',
                'port': 5432,
                'database': 'hailo_db',
                'user': 'hailo_user',
                'password': 'hailo_pass',
                'min_connections': 1,
                'max_connections': 10
            }
        """
        self.db_config = db_config
        self.pool = None
        # MQTT integration
        self.mqtt_client = None
        self.mqtt_listener_active = False
        self.message_buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_thread = None
        # Configuration for MQTT buffering
        self.batch_size = 10
        self.batch_timeout = 5.0
        self.last_flush_time = time.time()
        self.message_parser: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]] = self._default_message_parser
        self.message_buffer = deque(maxlen=15)
        self.inserted_track_ids = set()
        # Initialize pool
        self._initialize()
    
    def _initialize(self):
        """Initialize database with connection testing."""
        # Create pool
        self.pool = self._create_pool()
        # Verify pool is working
        if not self.verify_pool():
            raise ConnectionError("Connection pool verification failed")
        print("[DatabaseManager] PostgreSQL initialized successfully")
    
    def _create_pool(self) -> ThreadedConnectionPool:
        """Create PostgreSQL connection pool."""
        try:
            pool = ThreadedConnectionPool(
                minconn=self.db_config.get('min_connections', 1),
                maxconn=self.db_config.get('max_connections', 10),
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'hailo_db'),
                user=self.db_config.get('user', 'hailo_user'),
                password=self.db_config.get('password', '')
            )
            return pool
        except Exception as e:
            print(f"[DatabaseManager] Failed to create pool: {e}")
            raise
    
    def verify_pool(self) -> bool:
        """Verify connection pool is working."""
        try:
            conn = self.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()[0]
                cursor.close()
                self.return_connection(conn)
                return result == 1
            return False
        except Exception as e:
            print(f"[DatabaseManager] Pool verification failed: {e}")
            return False
    
    def get_connection(self):
        """Get connection from pool."""
        if not self.pool: 
            raise ConnectionError("Connection pool not initialized")
        return self.pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool."""
        if conn and self.pool:
            self.pool.putconn(conn)
    
    def _process_mqtt_message(self, msg):
        """Process incoming MQTT messages."""
        try:
            # Parse JSON payload
            payload = json.loads(msg.payload.decode('utf-8'))
            print(f"[DatabaseManager] MQTT message received on topic {msg.topic}")
            # Parse message to detection format
            detections = self.message_parser(msg.topic, payload)
            if not detections:
                return
            with self.buffer_lock:
                for det in detections:
                    track_id = det.get("track_id")
                    # Skip duplicates
                    if track_id is not None and track_id in self.inserted_track_ids:
                        continue
                    # Add to buffer (deque automatically drops oldest if maxlen exceeded)
                    self.message_buffer.append(det)
                    if track_id is not None:
                        self.inserted_track_ids.add(track_id)
            #if len(self.message_buffer) >= self.message_buffer.maxlen:
            #    self._flush_buffer()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in MQTT message: {e}")
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
        
    def _default_message_parser(self, topic: str, payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Enhanced parser for MQTT messages. Supports multiple message formats:
        Format 1 - Direct detection:
            {"timestamp": "2025-01-15T10:30:00", "frame_id": 12345, "class_name": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2], "track_id": 42}
        Format 2 - Separate coordinates:
            {"timestamp": "2025-01-15T10:30:00", "frame_id": 12345, "class_name": "person", "confidence": 0.95, "x1": 100, "y1": 150, "x2": 200, "y2": 350, "track_id": 42}
        Format 3 - Nested data:
            {"data": {"class_name": "person", ...}}
        Format 4 - Batch detections (NEW):
            {"timestamp": "2026-01-16T09:41:59.457452", "frame_count": 120, "detections": [{...}, {...}, ...]}
        Returns:
            List of detection dictionaries (can be single item or multiple for batch)
            None if parsing fails
        """
        #print(f"[DatabaseManager] Parsing MQTT message for topic {topic}, payload keys: {payload.keys()}")
        frame_count = payload.get("frame_count", 0)
        detections = []
        for det in payload["detections"]:
            #print(f"[DatabaseManager] Parsing detection: {det}")
            # Extract class name (support different field names)
            bbox = det.get('bbox')
            track_id = det.get("track_id")
            if track_id is not None and track_id in self.inserted_track_ids:
                continue  # skip duplicates
            # mark as inserted
            if track_id is not None:
                self.inserted_track_ids.add(track_id)
            # Build detection dictionary
            detection = {
                'timestamp_': det['timestamp'] if 'timestamp' in det else payload.get('timestamp', datetime.now().isoformat()),
                'frame_id': int(frame_count),
                'class_name': str(det['class_name']),
                'confidence': float(det['confidence']),
                'x1': float(bbox[0]),
                'y1': float(bbox[1]),
                'x2': float(bbox[2]),
                'y2': float(bbox[3]),
                'track_id': int(det['track_id']) if 'track_id' in det and det['track_id'] else None,
            }
            detections.append(detection)
            # Validate bounding box (skip for normalized coordinates)
            self.insert_batch_detections(detections)
        #print(f"[DatabaseManager] Parsed detections: {detections}")                    
        print(f"[DatabaseManager] Parsed {len(payload['detections'])} detections from MQTT message")
        return detections
    
    def insert_batch_detections(self, detections: List[Dict[str, Any]]):
        """
        Insert multiple detections in batch (much faster than individual inserts).
        Args:
            detections: List of detection dictionaries
        """
        if not detections: 
            return
        conn = self.get_connection()
        cursor = conn.cursor()
        #print("[DatabaseManager] Inserting batch detections...")
        try:
            query = '''
                INSERT INTO detections (timestamp_, frame_id, class_name, confidence, x1, y1, x2, y2, track_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            data = [
                (
                    d.get('timestamp_'), 
                    d.get('frame_id'), 
                    d.get('class_name'), 
                    d.get('confidence'), 
                    d.get('x1'), 
                    d.get('y1'), 
                    d.get('x2'), 
                    d.get('y2'), 
                    d.get('track_id'),
                ) 
                for d in detections
            ]
            execute_batch(cursor, query, data, page_size=100)
            conn.commit()
            print(f"[DatabaseManager] Inserted {len(detections)} detections")
        except Exception as e:
            print(f"[DatabaseManager] Error in batch insert: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def get_recent_detections(self, minutes: int = 5, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve recent detections.
        Args:
            minutes: Time window in minutes
            limit: Maximum number of results
        Returns:
            List of detection dictionaries
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Use proper parameterized query with interval
            cursor.execute('''
                SELECT id, timestamp_, frame_id, class_name, confidence, x1, y1, x2, y2, track_id
                FROM detections
                WHERE timestamp_ > NOW() - make_interval(mins => %s)
                ORDER BY timestamp_ DESC
                LIMIT %s
            ''', (minutes, limit))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        finally:
            cursor.close()
            self.return_connection(conn)

    def get_counts_detections(self, minutes: int = 5, per: str = "10 minutes") -> List[Dict[str, Any]]:
        """
        Get counts of detections within a time interval.
        Args:
            minutes: Time interval in minutes
            per: Time unit for grouping (e.g., "minute", "hour")
        Returns:
            Dictionary with counts
        """
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        if per == "1 minute":
            time_bucket = "date_trunc('minute', timestamp_)"
        elif per == "10 minutes":
            time_bucket = """
                date_trunc('minute', timestamp_)
                - INTERVAL '1 minute' * (EXTRACT(MINUTE FROM timestamp_)::int % 10)
            """
        elif per == "1 hour":
            time_bucket = "date_trunc('hour', timestamp_)"
        elif per == "1 day":
            time_bucket = "date_trunc('day', timestamp_)"
        else:
            raise ValueError("Unsupported interval")
        try:
            query = f"""
                SELECT
                    {time_bucket} AS time_bucket,
                    COUNT(*) AS detection_count
                FROM detections
                WHERE timestamp_ > NOW() - INTERVAL %s
                GROUP BY time_bucket
                ORDER BY time_bucket
            """
            cursor.execute(query, (f"{minutes} minutes"))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        finally:
            cursor.close()
            self.return_connection(conn)

    def get_statistics(self, hours: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated statistics by class.
        Args:
            hours: Time window in hours
        Returns:
            Dictionary mapping class_name to statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT class_name, COUNT(*) as total_detections, COUNT(DISTINCT track_id) as unique_tracks,
                    AVG(confidence) as avg_confidence,
                    MIN(confidence) as min_confidence,
                    MAX(confidence) as max_confidence
                FROM detections
                WHERE timestamp_ > NOW() - INTERVAL '%s hours'
                AND track_id IS NOT NULL
                GROUP BY class_name
                ORDER BY total_detections DESC
            ''', (hours,))

            results = {}
            for row in cursor.fetchall():
                results[row[0]] = {
                    'total_detections': row[1],
                    'unique_tracks': row[2],
                    'avg_confidence': float(row[3]) if row[3] else 0,
                    'min_confidence': float(row[4]) if row[4] else 0,
                    'max_confidence': float(row[5]) if row[5] else 0
                }
            return results
        finally:
            cursor.close()
            self.return_connection(conn)

    # HEATLH CHECK    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get database health status.
        Returns:
            Dictionary with health metrics
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Get total detections
            cursor.execute("SELECT COUNT(*) FROM detections;")
            total_count = cursor.fetchone()[0]
            # Get latest detection time
            cursor.execute("SELECT MAX(timestamp_) FROM detections;")
            latest = cursor.fetchone()[0]
            # Get database size
            cursor.execute(""" SELECT pg_size_pretty(pg_database_size(current_database())); """)
            db_size = cursor.fetchone()[0]
            return { 
                'status': 'healthy', 
                'total_detections': total_count,
                'latest_detection': latest,
                'database_size': db_size,
            }
        except Exception as e:
            return { 'status': 'unhealthy', 'error': str(e) }
        finally:
            cursor.close()
            self.return_connection(conn)

    # CLOSE POOL
    def close(self):
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()
            print("[DatabaseManager] Connection pool closed")

# Example usage / test
def test_database_manager():
    # Initialize your existing MQTT client
    mqtt_client = MQTTPublisher( 
        broker_host="mqtt.portabo.cz", 
        broker_port=8883, 
        topic="/videoanalyza", 
        client_id = "partik_psql_test_listener",
        username="videoanalyza", 
        password="phdA9ZNW1vfkXdJkhhbP"
    )
    # Test configuration
    config = { 'host': '192.168.37.31', 'port': 5432, 'database': 'hailo_db', 'user': 'hailo_user', 'password': 'hailo_pass', 'min_connections': 2, 'max_connections': 10 }
    try:
        # Initialize database manager (includes connection testing)
        db = PostgreDatabaseManager(config)
        # ConnecT to MQTT client
        mqtt_client.connect({"db": db})
        time.sleep(2)  # Wait for connection
        # Get health status
        health = db.get_health_status()
        print(f"\n[Test] Database Health: {health}")
        # Test single insert
        print("\n[Test] Testing single detection insert...")
        test_detection = {'timestamp_': datetime.now(), 'frame_id': 1, 'class_name': 'person', 'confidence': 0.95, 'x1': 100, 'y1': 150, 'x2': 200, 'y2': 350, 'track_id': 1}
        #db.insert_detection(test_detection)
        #print("[Test] Single insert successful")
        ## Test batch insert
        #print("\n[Test] Testing batch insert...")
        #batch_detections = [
        #    { 'timestamp_': datetime.now(), 'frame_id': 2, 'class_name': 'car', 'confidence': 0.88, 'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150, 'track_id': 2 },
        #    { 'timestamp_': datetime.now(), 'frame_id': 2, 'class_name': 'car', 'confidence': 0.92, 'x1': 200, 'y1': 100, 'x2': 300, 'y2': 200, 'track_id': 3 }
        #]
        #db.insert_batch_detections(batch_detections)
        #print("[Test] ✓ Batch insert successful")
        # Test retrieval
        print("\n[Test] Testing data retrieval...")
        recent = db.get_recent_detections(minutes=5, limit=10)
        print(f"[Test] Retrieved {len(recent)} recent detections")
        # Test statistics
        print("\n[Test] Testing statistics...")
        stats = db.get_statistics(hours=1)
        print(f"[Test] Statistics: {stats}")
        print("\n[Test] All tests passed!")
    except Exception as e:
        print(f"\n[Test] Test failed: {e}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'db' in locals():
            db.close()
            mqtt_client.disconnect()

def main():
    # Test configuration
    config = { 'host': '192.168.37.31', 'port': 5432, 'database': 'hailo_db', 'user': 'hailo_user', 'password': 'hailo_pass', 'min_connections': 2, 'max_connections': 10 }
    try:
        # Initialize database manager (includes connection testing)
        db = PostgreDatabaseManager(config)
        # Initialize your existing MQTT client
        mqtt_client = MQTTPublisher( 
            broker_host="mqtt.portabo.cz", 
            broker_port=8883, 
            topic="/videoanalyza", 
            client_id = "partik_psql_test_listener",
            username="videoanalyza", 
            password="phdA9ZNW1vfkXdJkhhbP"
        )
        mqtt_client.client.user_data_set({"db": db})
        mqtt_client.client.on_message = mqtt_client.on_message
        # ConnecT to MQTT client
        mqtt_client.connect({"db": db})
        #mqtt_client.loop_start()
        time.sleep(2)  # Wait for connection
        batch_timeout=1.0  # Or every 10 seconds
        # Monitor statistics
        last_stats_time = time.time()
        while True:
            time.sleep(5)
            # Print statistics every 30 seconds
            if time.time() - last_stats_time >= 30:
                # Also print database health
                health = db.get_health_status()
                print(f"Database: {health['total_detections']} total detections, " f"Size: {health['database_size']}")
                #print("Counts: ", db.get_counts_detections(minutes=560, per="1 hour"))
                last_stats_time = time.time()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'db' in locals():
            db.close()
        if 'mqtt_client' in locals():
            mqtt_client.disconnect()
        print("✓ Shutdown complete")
    

if __name__ == "__main__":
    #test_database_manager()
    main()