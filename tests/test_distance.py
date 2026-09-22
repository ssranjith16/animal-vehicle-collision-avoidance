"""
Unit tests for distance estimator
"""

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.distance_estimator import DistanceEstimator

class TestDistanceEstimator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.estimator = DistanceEstimator(focal_length=625)
        
    def test_distance_calculation(self):
        """Test basic distance calculation"""
        # Known parameters: focal_length=625, real_height=1.2m
        pixel_height = 150
        expected_distance = (625 * 1.2) / 150  # 5.0 meters
        
        distance = self.estimator.calculate_distance(pixel_height, 'default')
        self.assertAlmostEqual(distance, expected_distance, places=2)
        
    def test_zero_pixel_height(self):
        """Test distance calculation with zero pixel height"""
        distance = self.estimator.calculate_distance(0, 'default')
        self.assertEqual(distance, float('inf'))
        
    def test_ttc_calculation(self):
        """Test Time to Collision calculation"""
        distance = 30.0  # meters
        speed_kmh = 60.0  # km/h
        
        # 60 km/h = 16.67 m/s
        # TTC = 30 / 16.67 ≈ 1.8 seconds
        expected_ttc = 30.0 / (60.0 * 1000 / 3600)
        
        ttc = self.estimator.calculate_ttc(distance, speed_kmh)
        self.assertAlmostEqual(ttc, expected_ttc, places=2)
        
    def test_ttc_zero_speed(self):
        """Test TTC with zero speed"""
        ttc = self.estimator.calculate_ttc(10.0, 0)
        self.assertEqual(ttc, float('inf'))
        
    def test_warning_levels(self):
        """Test warning level determination"""
        # Danger
        level = self.estimator.get_warning_level(ttc=1.0, distance=10.0)
        self.assertEqual(level, 'DANGER')
        
        # Warning
        level = self.estimator.get_warning_level(ttc=2.5, distance=20.0)
        self.assertEqual(level, 'WARNING')
        
        # Caution
        level = self.estimator.get_warning_level(ttc=4.0, distance=40.0)
        self.assertEqual(level, 'CAUTION')
        
        # Safe
        level = self.estimator.get_warning_level(ttc=10.0, distance=100.0)
        self.assertEqual(level, 'SAFE')
        
    def test_focal_length_calibration(self):
        """Test focal length calibration"""
        known_distance = 5.0
        known_height = 1.0
        pixel_height = 125
        
        focal_length = self.estimator.calibrate_focal_length(
            known_distance, known_height, pixel_height
        )
        
        expected_focal_length = (125 * 5.0) / 1.0
        self.assertEqual(focal_length, expected_focal_length)

if __name__ == '__main__':
    unittest.main()