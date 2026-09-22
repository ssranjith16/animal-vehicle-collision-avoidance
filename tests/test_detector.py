
"""
Unit tests for animal detector
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.detector import AnimalDetector
from config import config

class TestAnimalDetector(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.detector = AnimalDetector(
            model_path=str(config.MODEL_PATH),
            confidence_threshold=0.5
        )
        
        # Create test image
        cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_model_loaded(self):
        """Test if model loaded successfully"""
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.class_names)
        
    def test_detection_empty_frame(self):
        """Test detection on empty frame"""
        detections = self.detector.detect(self.test_image)
        self.assertEqual(len(detections), 0)
        
    def test_confidence_threshold(self):
        """Test confidence threshold filtering"""
        self.detector.confidence_threshold = 0.9
        detections = self.detector.detect(self.test_image)
        self.assertEqual(len(detections), 0)
        
    def test_target_classes(self):
        """Test target class filtering"""
        self.assertIsNotNone(self.detector.target_classes)
        self.assertIn(16, self.detector.target_classes)  # Dog class
        
    def test_iou_calculation(self):
        """Test IoU calculation"""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)
        iou = self.detector._calculate_iou(bbox1, bbox2)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)
        
        # Perfect overlap
        iou = self.detector._calculate_iou(bbox1, bbox1)
        self.assertEqual(iou, 1.0)
        
        # No overlap
        bbox3 = (200, 200, 300, 300)
        iou = self.detector._calculate_iou(bbox1, bbox3)
        self.assertEqual(iou, 0.0)

if __name__ == '__main__':
    unittest.main()