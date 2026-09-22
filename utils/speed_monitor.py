"""
Vehicle speed monitoring module
"""

import time
import threading
from typing import Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SpeedMonitor:
    """
    Monitor and track vehicle speed from various sources
    Supports simulation, GPS, OBD-II, and manual input
    """
    
    def __init__(self, source: str = 'simulation', 
                 default_speed: float = 30.0):
        """
        Initialize speed monitor
        
        Args:
            source: Speed source ('simulation', 'gps', 'obd', 'manual')
            default_speed: Default speed in km/h
        """
        self.source = source
        self.current_speed = default_speed
        self.speed_history = []
        self.max_history = 100
        
        # For GPS
        self.gps_available = False
        self.last_gps_update = 0
        
        # For OBD-II simulation
        self.obd_available = False
        
        # Threading
        self.monitoring = False
        self.monitor_thread = None
        self.speed_callbacks = []
        
        # Speed limits
        self.max_speed = 120.0  # km/h
        self.min_speed = 0.0
        
        logger.info(f"Speed monitor initialized: source={source}, default={default_speed} km/h")
    
    def start_monitoring(self):
        """Start continuous speed monitoring"""
        if self.source in ['gps', 'obd']:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info(f"Started {self.source} speed monitoring")
    
    def stop_monitoring(self):
        """Stop speed monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped speed monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            if self.source == 'gps':
                self._read_gps_speed()
            elif self.source == 'obd':
                self._read_obd_speed()
            
            time.sleep(0.5)  # Update at 2 Hz
    
    def _read_gps_speed(self):
        """Read speed from GPS module"""
        # Placeholder for actual GPS implementation
        # In real implementation, read from serial GPS
        try:
            # Simulate GPS speed reading
            pass
        except Exception as e:
            logger.error(f"GPS read error: {e}")
    
    def _read_obd_speed(self):
        """Read speed from OBD-II interface"""
        # Placeholder for actual OBD-II implementation
        # In real implementation, use python-OBD library
        try:
            # Simulate OBD speed reading
            pass
        except Exception as e:
            logger.error(f"OBD read error: {e}")
    
    def set_speed(self, speed: float):
        """Manually set current speed"""
        speed = max(self.min_speed, min(speed, self.max_speed))
        
        if speed != self.current_speed:
            self.current_speed = speed
            self.speed_history.append((time.time(), speed))
            
            # Trim history
            if len(self.speed_history) > self.max_history:
                self.speed_history.pop(0)
            
            # Notify callbacks
            self._notify_callbacks(speed)
            
            logger.debug(f"Speed updated: {speed:.1f} km/h")
    
    def get_speed(self) -> float:
        """Get current speed in km/h"""
        return self.current_speed
    
    def get_speed_ms(self) -> float:
        """Get current speed in m/s"""
        return self.current_speed * (1000 / 3600)
    
    def get_average_speed(self, window_seconds: float = 10.0) -> float:
        """Get average speed over time window"""
        if not self.speed_history:
            return self.current_speed
        
        current_time = time.time()
        recent_speeds = [
            speed for t, speed in self.speed_history
            if current_time - t <= window_seconds
        ]
        
        if not recent_speeds:
            return self.current_speed
        
        return np.mean(recent_speeds)
    
    def add_callback(self, callback: Callable[[float], None]):
        """Add speed change callback"""
        self.speed_callbacks.append(callback)
    
    def _notify_callbacks(self, speed: float):
        """Notify all callbacks of speed change"""
        for callback in self.speed_callbacks:
            try:
                callback(speed)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def simulate_acceleration(self, target_speed: float, 
                             acceleration_rate: float = 2.0,
                             duration: float = 5.0):
        """
        Simulate acceleration to target speed
        
        Args:
            target_speed: Target speed in km/h
            acceleration_rate: Acceleration in km/h per second
            duration: Duration of acceleration
        """
        steps = int(duration * 10)  # 10 updates per second
        speed_step = (target_speed - self.current_speed) / steps
        
        for i in range(steps):
            new_speed = self.current_speed + speed_step
            self.set_speed(new_speed)
            time.sleep(0.1)
    
    def simulate_deceleration(self, target_speed: float,
                             deceleration_rate: float = 3.0,
                             duration: float = 3.0):
        """Simulate deceleration to target speed"""
        self.simulate_acceleration(target_speed, -deceleration_rate, duration)
    
    def get_speed_status(self) -> dict:
        """Get comprehensive speed status"""
        return {
            'current_speed': self.current_speed,
            'average_speed_10s': self.get_average_speed(10.0),
            'average_speed_30s': self.get_average_speed(30.0),
            'max_speed_recent': max([s for _, s in self.speed_history[-30:]] + [0]),
            'min_speed_recent': min([s for _, s in self.speed_history[-30:]] + [0]),
            'source': self.source,
            'timestamp': time.time()
        }