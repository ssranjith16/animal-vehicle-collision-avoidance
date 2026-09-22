"""
Data logging and statistics module
"""

import json
import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DataLogger:
    """
    Log system events, detections, and statistics
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize data logger
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log files
        self.detection_log = self.log_dir / f"detections_{self.session_id}.csv"
        self.event_log = self.log_dir / f"events_{self.session_id}.json"
        self.stats_log = self.log_dir / f"statistics_{self.session_id}.json"
        
        # Initialize CSV
        self._init_csv()
        
        # In-memory storage
        self.events = []
        self.statistics = {
            'total_frames': 0,
            'total_detections': 0,
            'alerts_triggered': 0,
            'danger_alerts': 0,
            'warning_alerts': 0,
            'caution_alerts': 0,
            'detections_by_class': {},
            'average_speed': 0,
            'max_speed': 0,
            'min_speed': float('inf'),
            'session_start': time.time()
        }
        
        # For running average calculation
        self._speed_sum = 0.0
        
        logger.info(f"Data logger initialized: session={self.session_id}")
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.detection_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame_id', 'class_name', 'confidence',
                'x1', 'y1', 'x2', 'y2', 'distance', 'ttc', 'speed',
                'alert_level'
            ])
    
    def log_detection(self, frame_id: int, detection: Dict, 
                     distance: float = None, ttc: float = None,
                     speed: float = None, alert_level: str = None):
        """
        Log a single detection
        
        Args:
            frame_id: Frame number
            detection: Detection dictionary
            distance: Distance to object in meters
            ttc: Time to collision in seconds
            speed: Vehicle speed in km/h
            alert_level: Current alert level
        """
        x1, y1, x2, y2 = detection['bbox']
        
        with open(self.detection_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), frame_id, detection['class_name'],
                detection['confidence'], x1, y1, x2, y2,
                distance if distance else '',
                ttc if ttc else '',
                speed if speed else '',
                alert_level if alert_level else ''
            ])
        
        # Update statistics
        self.statistics['total_detections'] += 1
        class_name = detection['class_name']
        self.statistics['detections_by_class'][class_name] = \
            self.statistics['detections_by_class'].get(class_name, 0) + 1
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a system event
        
        Args:
            event_type: Type of event (e.g., 'alert', 'speed_change', 'error')
            data: Event data
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }
        
        self.events.append(event)
        
        # Update alert statistics
        if event_type == 'alert':
            self.statistics['alerts_triggered'] += 1
            level = data.get('level', '')
            if level == 'DANGER':
                self.statistics['danger_alerts'] += 1
            elif level == 'WARNING':
                self.statistics['warning_alerts'] += 1
            elif level == 'CAUTION':
                self.statistics['caution_alerts'] += 1
    
    def update_statistics(self, frame_id: int, speed: float):
        """Update session statistics"""
        # Update frame count (start from 1, not 0)
        self.statistics['total_frames'] = frame_id + 1
        
        # Update max/min speed
        self.statistics['max_speed'] = max(self.statistics['max_speed'], speed)
        self.statistics['min_speed'] = min(self.statistics['min_speed'], speed)
        
        # Update running average using sum method (avoids division by zero)
        self._speed_sum += speed
        n = self.statistics['total_frames']
        if n > 0:
            self.statistics['average_speed'] = self._speed_sum / n
    
    def save_detection_image(self, frame: np.ndarray, detections: List[Dict],
                            frame_id: int):
        """Save frame with detections"""
        if frame_id % 30 == 0:  # Save every 30th frame to save space
            img_dir = self.log_dir / f"images_{self.session_id}"
            img_dir.mkdir(exist_ok=True)
            
            filename = img_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(filename), frame)
    
    def save_session(self):
        """Save session data to disk"""
        # Add session end time
        self.statistics['session_end'] = time.time()
        self.statistics['session_duration'] = \
            self.statistics['session_end'] - self.statistics['session_start']
        
        # Save events
        with open(self.event_log, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'events': self.events,
                'event_count': len(self.events)
            }, f, indent=2)
        
        # Save statistics
        with open(self.stats_log, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'statistics': self.statistics
            }, f, indent=2)
        
        logger.info(f"Session saved: {self.session_id}")
        logger.info(f"  Frames: {self.statistics['total_frames']}")
        logger.info(f"  Detections: {self.statistics['total_detections']}")
        logger.info(f"  Alerts: {self.statistics['alerts_triggered']}")
    
    def get_summary(self) -> Dict:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'duration': time.time() - self.statistics['session_start'],
            'total_frames': self.statistics['total_frames'],
            'total_detections': self.statistics['total_detections'],
            'alerts_triggered': self.statistics['alerts_triggered'],
            'average_speed': self.statistics['average_speed'],
            'top_detected': sorted(
                self.statistics['detections_by_class'].items(),
                key=lambda x: x[1], reverse=True
            )[:5] if self.statistics['detections_by_class'] else []
        }