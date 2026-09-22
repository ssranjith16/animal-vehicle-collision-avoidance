"""
Distance estimation using monocular vision and triangle similarity
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DistanceEstimator:
    """
    Estimates distance to objects using known real-world dimensions
    and focal length of the camera.
    """
    
    def __init__(self, focal_length: float = 625, 
                 known_heights: Dict[str, float] = None):
        """
        Initialize distance estimator
        
        Args:
            focal_length: Camera focal length in pixels
            known_heights: Dictionary of known object heights in meters
        """
        self.focal_length = focal_length
        self.known_heights = known_heights or {
            'cow': 1.4,
            'dog': 0.6,
            'horse': 1.6,
            'default': 1.2
        }
        
        # For depth estimation refinement
        self.use_advanced_depth = False
        self.depth_history = {}
        self.history_size = 5
        
    def calculate_distance(self, object_height_pixels: float, 
                          object_class: str = 'default') -> float:
        """
        Calculate distance using triangle similarity
        
        Distance = (Focal Length * Real Height) / Pixel Height
        
        Args:
            object_height_pixels: Height of bounding box in pixels
            object_class: Class name for known height lookup
            
        Returns:
            Distance in meters
        """
        if object_height_pixels <= 0:
            return float('inf')
        
        real_height = self.known_heights.get(object_class, 
                                            self.known_heights['default'])
        
        distance = (self.focal_length * real_height) / object_height_pixels
        
        # Apply distance smoothing
        distance = self._smooth_distance(object_class, distance)
        
        return distance
    
    def _smooth_distance(self, object_id: str, distance: float) -> float:
        """Apply moving average to reduce distance jitter"""
        if object_id not in self.depth_history:
            self.depth_history[object_id] = []
        
        self.depth_history[object_id].append(distance)
        
        if len(self.depth_history[object_id]) > self.history_size:
            self.depth_history[object_id].pop(0)
        
        return np.mean(self.depth_history[object_id])
    
    def calculate_from_bbox(self, bbox: Tuple[int, int, int, int], 
                           object_class: str = 'default') -> Dict[str, float]:
        """
        Calculate distance and other metrics from bounding box
        
        Args:
            bbox: (x1, y1, x2, y2) coordinates
            object_class: Class name for known height lookup
            
        Returns:
            Dictionary with distance, position, and size metrics
        """
        x1, y1, x2, y2 = bbox
        height_pixels = y2 - y1
        width_pixels = x2 - x1
        
        distance = self.calculate_distance(height_pixels, object_class)
        
        # Calculate relative position
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return {
            'distance': distance,
            'height_pixels': height_pixels,
            'width_pixels': width_pixels,
            'center_x': center_x,
            'center_y': center_y,
            'aspect_ratio': width_pixels / height_pixels if height_pixels > 0 else 0
        }
    
    def calculate_ttc(self, distance: float, speed_kmh: float) -> float:
        """
        Calculate Time to Collision (TTC)
        
        Args:
            distance: Distance to object in meters
            speed_kmh: Vehicle speed in km/h
            
        Returns:
            TTC in seconds
        """
        if speed_kmh <= 0:
            return float('inf')
        
        speed_ms = speed_kmh * (1000 / 3600)  # Convert km/h to m/s
        return distance / speed_ms
    
    def calibrate_focal_length(self, known_distance: float, 
                              known_height: float, 
                              pixel_height: float) -> float:
        """
        Calibrate focal length for a specific camera
        
        F = (Pixel Height * Known Distance) / Known Real Height
        
        Args:
            known_distance: Actual distance to object in meters
            known_height: Actual height of object in meters
            pixel_height: Measured pixel height of object
            
        Returns:
            Calculated focal length
        """
        if known_height <= 0 or known_distance <= 0:
            raise ValueError("Height and distance must be positive")
        
        focal_length = (pixel_height * known_distance) / known_height
        self.focal_length = focal_length
        
        logger.info(f"Calibrated focal length: {focal_length:.2f} pixels")
        return focal_length
    
    def estimate_animal_size(self, distance: float, pixel_height: float) -> float:
        """
        Estimate actual size of detected animal
        Useful for classifying unknown animals
        """
        if distance <= 0:
            return 0
        
        return (pixel_height * distance) / self.focal_length
    
    def get_warning_level(self, ttc: float, distance: float) -> str:
        """
        Determine warning level based on TTC and distance
        
        Returns:
            'SAFE', 'CAUTION', 'WARNING', or 'DANGER'
        """
        if ttc > 5.0 or distance > 50:
            return 'SAFE'
        elif ttc > 3.0 or distance > 30:
            return 'CAUTION'
        elif ttc > 2.0 or distance > 15:
            return 'WARNING'
        else:
            return 'DANGER'