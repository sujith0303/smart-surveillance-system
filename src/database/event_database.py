# Day 5-7: Event Database & Advanced Event Detection
# File: event_database.py

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

class SurveillanceEventDB:
    """
    Database for storing and querying surveillance events
    Stores detections, tracks, and behaviors for fast retrieval
    """
    
    def __init__(self, db_path='surveillance.db'):
        """Initialize SQLite database"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema"""
        
        # Events table: All detection events
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                frame_number INTEGER,
                camera_id TEXT DEFAULT 'cam1',
                object_type TEXT NOT NULL,
                track_id INTEGER,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                color TEXT,
                attributes TEXT,
                video_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Behaviors table: Detected suspicious behaviors
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS behaviors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                behavior_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                duration REAL,
                severity TEXT DEFAULT 'low',
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tracks table: Object tracking information
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                track_id INTEGER PRIMARY KEY,
                object_type TEXT,
                color TEXT,
                first_seen REAL,
                last_seen REAL,
                duration REAL,
                total_detections INTEGER,
                camera_id TEXT DEFAULT 'cam1',
                attributes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Incidents table: High-level security incidents
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                track_ids TEXT,
                camera_id TEXT,
                resolved BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for fast queries
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
            ON events(timestamp)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_object_type 
            ON events(object_type)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_track_id 
            ON events(track_id)
        ''')
        self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_behaviors_timestamp 
            ON behaviors(timestamp)
        ''')
        
        self.conn.commit()
    
    def insert_event(self, event: Dict):
        """Insert detection event into database"""
        self.cursor.execute('''
            INSERT INTO events (
                timestamp, frame_number, object_type, track_id, 
                confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                color, attributes, video_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.get('timestamp'),
            event.get('frame_number'),
            event.get('object_type'),
            event.get('track_id'),
            event.get('confidence'),
            event.get('bbox', [0,0,0,0])[0],
            event.get('bbox', [0,0,0,0])[1],
            event.get('bbox', [0,0,0,0])[2],
            event.get('bbox', [0,0,0,0])[3],
            event.get('attributes', {}).get('color'),
            json.dumps(event.get('attributes', {})),
            event.get('video_path')
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_behavior(self, track_id: int, behavior: Dict):
        """Insert detected behavior"""
        severity = self._assess_severity(behavior['type'])
        
        self.cursor.execute('''
            INSERT INTO behaviors (
                track_id, behavior_type, timestamp, duration, 
                severity, details
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            track_id,
            behavior.get('type'),
            behavior.get('timestamp'),
            behavior.get('duration'),
            severity,
            json.dumps(behavior)
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_track(self, track_id: int, track_data: Dict):
        """Insert or update track information"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO tracks (
                track_id, object_type, color, first_seen, last_seen,
                duration, total_detections, attributes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            track_id,
            track_data.get('attributes', {}).get('type'),
            track_data.get('attributes', {}).get('color'),
            track_data.get('first_seen'),
            track_data.get('last_seen'),
            track_data.get('duration'),
            track_data.get('total_detections'),
            json.dumps(track_data.get('attributes', {}))
        ))
        self.conn.commit()
    
    def create_incident(self, incident_type: str, timestamp: float, 
                       severity: str, description: str, 
                       track_ids: List[int] = None):
        """Create security incident record"""
        self.cursor.execute('''
            INSERT INTO incidents (
                incident_type, timestamp, severity, description, track_ids
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            incident_type,
            timestamp,
            severity,
            description,
            json.dumps(track_ids) if track_ids else None
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def query_events_by_time(self, start_time: float, end_time: float,
                            object_type: Optional[str] = None):
        """Query events within time range"""
        if object_type:
            self.cursor.execute('''
                SELECT * FROM events 
                WHERE timestamp BETWEEN ? AND ?
                AND object_type = ?
                ORDER BY timestamp
            ''', (start_time, end_time, object_type))
        else:
            self.cursor.execute('''
                SELECT * FROM events 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_time, end_time))
        
        return self._format_results(self.cursor.fetchall())
    
    def query_events_by_attribute(self, attribute: str, value: str):
        """Query events by attribute (e.g., color='red')"""
        if attribute == 'color':
            self.cursor.execute('''
                SELECT * FROM events 
                WHERE color = ?
                ORDER BY timestamp DESC
            ''', (value,))
        else:
            # Search in JSON attributes
            self.cursor.execute('''
                SELECT * FROM events 
                WHERE attributes LIKE ?
                ORDER BY timestamp DESC
            ''', (f'%"{attribute}": "{value}"%',))
        
        return self._format_results(self.cursor.fetchall())
    
    def query_behaviors(self, behavior_type: Optional[str] = None,
                       severity: Optional[str] = None):
        """Query suspicious behaviors"""
        query = 'SELECT * FROM behaviors WHERE 1=1'
        params = []
        
        if behavior_type:
            query += ' AND behavior_type = ?'
            params.append(behavior_type)
        
        if severity:
            query += ' AND severity = ?'
            params.append(severity)
        
        query += ' ORDER BY timestamp DESC'
        
        self.cursor.execute(query, params)
        return self._format_results(self.cursor.fetchall())
    
    def query_track_history(self, track_id: int):
        """Get complete history of a tracked object"""
        # Get track info
        self.cursor.execute('''
            SELECT * FROM tracks WHERE track_id = ?
        ''', (track_id,))
        track = self.cursor.fetchone()
        
        # Get all events for this track
        self.cursor.execute('''
            SELECT * FROM events WHERE track_id = ?
            ORDER BY timestamp
        ''', (track_id,))
        events = self.cursor.fetchall()
        
        # Get behaviors for this track
        self.cursor.execute('''
            SELECT * FROM behaviors WHERE track_id = ?
            ORDER BY timestamp
        ''', (track_id,))
        behaviors = self.cursor.fetchall()
        
        return {
            'track': track,
            'events': self._format_results(events),
            'behaviors': self._format_results(behaviors)
        }
    
    def get_statistics(self, start_time: Optional[float] = None,
                      end_time: Optional[float] = None):
        """Get surveillance statistics"""
        time_filter = ''
        params = []
        
        if start_time and end_time:
            time_filter = 'WHERE timestamp BETWEEN ? AND ?'
            params = [start_time, end_time]
        
        # Total detections by type
        self.cursor.execute(f'''
            SELECT object_type, COUNT(*) as count
            FROM events
            {time_filter}
            GROUP BY object_type
        ''', params)
        detections_by_type = dict(self.cursor.fetchall())
        
        # Total behaviors by type
        self.cursor.execute(f'''
            SELECT behavior_type, COUNT(*) as count
            FROM behaviors
            {time_filter}
            GROUP BY behavior_type
        ''', params)
        behaviors_by_type = dict(self.cursor.fetchall())
        
        # Total unique tracks
        self.cursor.execute(f'''
            SELECT COUNT(DISTINCT track_id) FROM events
            {time_filter}
        ''', params)
        total_tracks = self.cursor.fetchone()[0]
        
        return {
            'detections_by_type': detections_by_type,
            'behaviors_by_type': behaviors_by_type,
            'total_unique_tracks': total_tracks
        }
    
    def _assess_severity(self, behavior_type: str) -> str:
        """Assess severity of behavior"""
        high_severity = ['weapon_detected', 'fighting', 'trespassing']
        medium_severity = ['loitering', 'erratic_movement', 'fast_movement']
        
        if behavior_type in high_severity:
            return 'high'
        elif behavior_type in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def _format_results(self, results):
        """Format SQL results to dictionaries"""
        if not results:
            return []
        
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Advanced Event Detector
class AdvancedEventDetector:
    """
    Detects complex events and patterns
    Goes beyond simple object detection
    """
    
    def __init__(self, db: SurveillanceEventDB):
        self.db = db
    
    def detect_crowd_gathering(self, timestamp: float, 
                              window: float = 10.0,
                              threshold: int = 5):
        """
        Detect if multiple people gathered in same time window
        
        Args:
            timestamp: Current timestamp
            window: Time window in seconds
            threshold: Minimum people count
        """
        events = self.db.query_events_by_time(
            timestamp - window, 
            timestamp,
            object_type='person'
        )
        
        if len(events) >= threshold:
            return {
                'type': 'crowd_gathering',
                'count': len(events),
                'timestamp': timestamp,
                'severity': 'medium' if len(events) < 10 else 'high'
            }
        return None
    
    def detect_abandoned_object(self, track_history: Dict, 
                               stationary_threshold: float = 30.0):
        """
        Detect if object left stationary (abandoned bag, etc)
        
        Args:
            track_history: Track data from database
            stationary_threshold: Seconds to consider abandoned
        """
        track = track_history['track']
        
        if not track:
            return None
        
        # Check if object hasn't moved
        duration = track['duration']
        object_type = track['object_type']
        
        # Only flag certain object types as potentially abandoned
        suspicious_objects = ['backpack', 'suitcase', 'handbag']
        
        if (object_type in suspicious_objects and 
            duration > stationary_threshold):
            return {
                'type': 'abandoned_object',
                'object_type': object_type,
                'duration': duration,
                'track_id': track['track_id'],
                'severity': 'high'
            }
        return None
    
    def detect_unusual_activity_time(self, timestamp: float):
        """Detect activity during unusual hours (e.g., 2-5 AM)"""
        # Convert timestamp to hour of day
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        
        # Define unusual hours (2 AM - 5 AM)
        if 2 <= hour < 5:
            return {
                'type': 'unusual_time_activity',
                'hour': hour,
                'timestamp': timestamp,
                'severity': 'medium'
            }
        return None


# Usage Example
if __name__ == "__main__":
    # Initialize database
    db = SurveillanceEventDB('surveillance.db')
    
    # Example: Insert some events
    event1 = {
        'timestamp': 100.5,
        'frame_number': 1005,
        'object_type': 'person',
        'track_id': 1,
        'confidence': 0.95,
        'bbox': [100, 200, 300, 500],
        'attributes': {'color': 'red', 'type': 'person'},
        'video_path': 'cam1.mp4'
    }
    db.insert_event(event1)
    
    # Query events
    events = db.query_events_by_time(0, 200)
    print(f"Found {len(events)} events between 0-200 seconds")
    
    # Query by attribute
    red_objects = db.query_events_by_attribute('color', 'red')
    print(f"Found {len(red_objects)} red objects")
    
    # Get statistics
    stats = db.get_statistics()
    print("\nStatistics:", json.dumps(stats, indent=2))
    
    # Advanced detection
    detector = AdvancedEventDetector(db)
    crowd = detector.detect_crowd_gathering(100.0)
    if crowd:
        print(f"\n Crowd detected: {crowd}")
    
    db.close()
