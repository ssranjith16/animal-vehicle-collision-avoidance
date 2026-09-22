"""
Animal detection module using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class AnimalDetector:
    """
    YOLO-based animal detection system
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', 
                 confidence_threshold: float = 0.5,
                 target_classes: List[int] = None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            target_classes: List of class IDs to detect (None = all classes)
        """
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = YOLO(model_path)
            if self.device == 'cuda':
                self.model.to('cuda')
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # COCO class names
        self.class_names = self.model.names
        
        # Target animal classes
        self.target_classes = target_classes or [16, 17, 18, 19, 20]  # Default animals
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
        # Detection history for tracking
        self.detection_history = []
        self.max_history = 30
        
    def detect(self, frame: np.ndarray, 
               verbose: bool = False) -> List[Dict]:
        """
        Detect animals in a frame
        
        Args:
            frame: Input image (BGR format)
            verbose: Print detection details
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(frame, verbose=verbose)[0]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter by class and confidence
                if cls_id in self.target_classes and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection = {
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id],
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    
                    detections.append(detection)
        
        # Update history
        self.detection_history.append(detections)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        return detections
    
    def detect_with_tracking(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and track animals (basic IoU tracking)
        """
        current_detections = self.detect(frame)
        
        # Add tracking IDs based on IoU with previous detections
        if len(self.detection_history) > 1:
            prev_detections = self.detection_history[-2]
            current_detections = self._assign_tracking_ids(
                current_detections, prev_detections
            )
        
        return current_detections
    
    def _assign_tracking_ids(self, current: List[Dict], 
                            previous: List[Dict]) -> List[Dict]:
        """Assign tracking IDs using IoU matching"""
        if not previous:
            for i, det in enumerate(current):
                det['track_id'] = i
            return current
        
        for det in current:
            best_iou = 0
            best_id = -1
            
            for prev_det in previous:
                iou = self._calculate_iou(det['bbox'], prev_det['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = prev_det.get('track_id', -1)
            
            if best_id >= 0:
                det['track_id'] = best_id
            else:
                det['track_id'] = max([d.get('track_id', -1) for d in current] + [-1]) + 1
        
        return current
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        # Areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times[-100:])  # Last 100 frames
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if not self.inference_times:
            return 0.0
        return 1000.0 / self.get_average_inference_time()
    
    def draw_detections(self, frame: np.ndarray, 
                        detections: List[Dict],
                        distances: Dict[int, float] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            distances: Optional dictionary mapping track_id to distance
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            track_id = det.get('track_id', -1)
            
            # Color based on distance/warning level
            color = (0, 255, 0)  # Default green
            if distances and track_id in distances:
                dist = distances[track_id]
                if dist < 15:
                    color = (0, 0, 255)  # Red - danger
                elif dist < 30:
                    color = (0, 255, 255)  # Yellow - warning
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            if track_id >= 0:
                label = f"ID:{track_id} {label}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw distance if available
            if distances and track_id in distances:
                dist_text = f"{distances[track_id]:.1f}m"
                cv2.putText(annotated, dist_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated
    
    def save_model(self, path: str):
        """Save the current model"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_custom_model(self, path: str):
        """Load a custom trained model"""
        self.model = YOLO(path)
        self.class_names = self.model.names
        logger.info(f"Custom model loaded from {path}")