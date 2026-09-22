#!/usr/bin/env python3
"""
Animal Collision Avoidance System - Main Application
Prevents animal-vehicle collisions on highways using computer vision
"""

import cv2
import time
import signal
import sys
import argparse
from pathlib import Path
import logging
import numpy as np

# Import project modules
from config import config
from models.detector import AnimalDetector
from utils.distance_estimator import DistanceEstimator
from utils.alert_system import AlertSystem, AlertLevel
from utils.speed_monitor import SpeedMonitor
from utils.data_logger import DataLogger

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnimalCollisionAvoidanceSystem:
    """
    Main system class for animal collision avoidance
    """
    
    def __init__(self, video_source=None, use_custom_model=False):
        """
        Initialize the collision avoidance system
        
        Args:
            video_source: Video source (0 for webcam, or file path)
            use_custom_model: Use custom trained model
        """
        logger.info("="*50)
        logger.info("Animal Collision Avoidance System Initializing")
        logger.info("="*50)
        
        # Video source
        self.video_source = video_source or config.VIDEO_SOURCE
        self.cap = None
        self.frame_width = config.CAMERA_WIDTH
        self.frame_height = config.CAMERA_HEIGHT
        self.frame_count = 0
        self.total_frames_processed = 0
        
        # Initialize components
        self._init_components(use_custom_model)
        
        # State
        self.running = False
        self.paused = False
        self.alert_active = False
        
        # Video writer
        self.video_writer = None
        if config.SAVE_OUTPUT:
            self._init_video_writer()
        
        # Performance tracking
        self.fps_history = []
        self.last_frame_time = time.time()
        self.processing_times = []
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("System initialization complete")
    
    def _init_components(self, use_custom_model):
        """Initialize all system components"""
        # Model path
        model_path = config.CUSTOM_MODEL_PATH if use_custom_model else config.MODEL_PATH
        
        # Detector
        logger.info(f"Loading model: {model_path}")
        self.detector = AnimalDetector(
            model_path=str(model_path),
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            target_classes=list(config.ANIMAL_CLASSES.keys())
        )
        
        # Distance estimator
        self.distance_estimator = DistanceEstimator(
            focal_length=config.FOCAL_LENGTH,
            known_heights=config.KNOWN_ANIMAL_HEIGHTS
        )
        
        # Alert system
        self.alert_system = AlertSystem(
            sound_file=str(config.ALERT_SOUND_PATH) if config.ALERT_SOUND_PATH.exists() else None,
            use_gpio=config.USE_GPIO,
            buzzer_pin=config.BUZZER_PIN,
            led_pin=config.LED_PIN,
            cooldown_time=config.ALERT_COOLDOWN
        )
        
        # Speed monitor
        self.speed_monitor = SpeedMonitor(
            source='simulation',
            default_speed=config.DEFAULT_SPEED
        )
        
        # Data logger
        self.data_logger = DataLogger(config.LOGS_DIR)
        
        # Add speed callback for logging
        self.speed_monitor.add_callback(self._on_speed_change)
    
    def _init_video_writer(self):
        """Initialize video writer for saving output"""
        output_dir = config.OUTPUT_VIDEO_PATH
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"output_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            config.CAMERA_FPS,
            (self.frame_width, self.frame_height)
        )
        logger.info(f"Video output: {output_path}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.stop()
    
    def start(self):
        """Start the collision avoidance system"""
        # Open video capture
        if isinstance(self.video_source, int):
            self.cap = cv2.VideoCapture(self.video_source)
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        else:
            self.cap = cv2.VideoCapture(str(self.video_source))
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.video_source}")
            return False
        
        # Get actual video properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if actual_width > 0 and actual_height > 0:
            self.frame_width = actual_width
            self.frame_height = actual_height
        
        self.running = True
        self.frame_count = 0
        self.total_frames_processed = 0
        
        logger.info(f"System started with source: {self.video_source}")
        logger.info(f"Resolution: {self.frame_width}x{self.frame_height}")
        logger.info(f"Video FPS: {actual_fps:.2f}")
        logger.info("Press 'q' to quit, 'p' to pause, '+/-' to adjust speed")
        logger.info("Press 's' to save screenshot, 'r' to reset stats")
        
        # Main loop
        try:
            self._main_loop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def _main_loop(self):
        """Main processing loop"""
        self.frame_count = 0
        self.total_frames_processed = 0
        
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow("Animal Collision Avoidance System", processed_frame)
                
                # Save output
                if self.video_writer:
                    # Ensure frame is correct size for video writer
                    if processed_frame.shape[1] != self.frame_width or processed_frame.shape[0] != self.frame_height:
                        processed_frame = cv2.resize(processed_frame, (self.frame_width, self.frame_height))
                    self.video_writer.write(processed_frame)
                
                # Update FPS
                self._update_fps()
                
                self.frame_count += 1
                self.total_frames_processed += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_keyboard(key):
                break
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Annotated frame
        """
        frame_start_time = time.time()
        
        # Resize frame if needed (for consistent processing)
        original_shape = frame.shape
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Detect animals
        detections = self.detector.detect_with_tracking(frame)
        
        # Get current speed
        current_speed = self.speed_monitor.get_speed()
        
        # Process each detection
        distances = {}
        alert_triggered = False
        highest_alert_level = AlertLevel.SAFE
        
        for detection in detections:
            # Calculate distance
            try:
                distance_info = self.distance_estimator.calculate_from_bbox(
                    detection['bbox'],
                    detection['class_name']
                )
                distance = distance_info['distance']
                
                # Store distance
                track_id = detection.get('track_id', -1)
                if track_id >= 0:
                    distances[track_id] = distance
                
                # Calculate TTC
                ttc = self.distance_estimator.calculate_ttc(distance, current_speed)
                
                # Determine alert level
                alert_level_str = self.distance_estimator.get_warning_level(ttc, distance)
                alert_level = AlertLevel[alert_level_str]
                
                if alert_level.value > highest_alert_level.value:
                    highest_alert_level = alert_level
                
                # Check 2-second rule
                if ttc < config.TWO_SECOND_RULE and distance < config.MIN_ALERT_DISTANCE:
                    alert_triggered = True
                    
                    # Log detection with alert
                    self.data_logger.log_detection(
                        self.frame_count, detection,
                        distance=distance, ttc=ttc,
                        speed=current_speed,
                        alert_level=alert_level_str
                    )
                    
                    # Trigger alert
                    self.alert_system.trigger_alert(alert_level, distance, ttc)
                else:
                    # Log detection without alert
                    self.data_logger.log_detection(
                        self.frame_count, detection,
                        distance=distance, ttc=ttc,
                        speed=current_speed
                    )
                    
            except Exception as e:
                logger.debug(f"Error processing detection: {e}")
                continue
        
        # Clear alert if no danger
        if not alert_triggered:
            self.alert_system.clear_alert()
        
        # Draw detections
        annotated_frame = self.detector.draw_detections(frame, detections, distances)
        
        # Add system overlay
        annotated_frame = self._add_system_overlay(annotated_frame, 
                                                   len(detections),
                                                   current_speed,
                                                   highest_alert_level)
        
        # Add visual alerts
        annotated_frame = self.alert_system.get_visual_overlay(annotated_frame)
        
        # Update statistics (use frame_count + 1 to avoid zero division)
        self.data_logger.update_statistics(self.frame_count, current_speed)
        
        # Save detection image periodically
        if config.SAVE_DETECTION_IMAGES and len(detections) > 0:
            self.data_logger.save_detection_image(annotated_frame, detections, self.frame_count)
        
        # Track processing time
        processing_time = (time.time() - frame_start_time) * 1000
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return annotated_frame
    
    def _add_system_overlay(self, frame: np.ndarray, num_detections: int,
                           speed: float, alert_level: AlertLevel) -> np.ndarray:
        """Add system information overlay"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # System status
        status_color = (0, 255, 0)
        if alert_level == AlertLevel.DANGER:
            status_color = (0, 0, 255)
            status_text = "!!! DANGER - BRAKE NOW !!!"
        elif alert_level == AlertLevel.WARNING:
            status_color = (0, 255, 255)
            status_text = "WARNING - Animal Detected - Slow Down"
        elif alert_level == AlertLevel.CAUTION:
            status_color = (0, 165, 255)
            status_text = "CAUTION - Animal Ahead"
        else:
            status_text = "SYSTEM ACTIVE - Monitoring"
        
        # Draw status
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw speed
        speed_text = f"Vehicle Speed: {speed:.1f} km/h"
        if speed > config.MAX_SYSTEM_SPEED:
            speed_color = (0, 0, 255)
            speed_text += " (Reduce speed!)"
        else:
            speed_color = (255, 255, 255)
        
        cv2.putText(frame, speed_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
        
        # Draw detection count
        det_text = f"Animals Detected: {num_detections}"
        cv2.putText(frame, det_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS and processing time
        if self.fps_history:
            fps = np.mean(self.fps_history[-10:])
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw inference time
        inf_time = self.detector.get_average_inference_time()
        if inf_time > 0:
            time_text = f"Inference: {inf_time:.1f}ms"
            cv2.putText(frame, time_text, (w - 180, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw processing time
        if self.processing_times:
            avg_proc_time = np.mean(self.processing_times[-30:])
            proc_text = f"Total: {avg_proc_time:.1f}ms"
            cv2.putText(frame, proc_text, (w - 180, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame counter
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (w - 150, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw controls hint at bottom
        controls = "Controls: [Q]uit | [P]ause | [+/-] Speed | [S]creenshot | [R]eset"
        text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, controls, (text_x, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_history.append(fps)
        
        self.last_frame_time = current_time
        
        # Keep only last 100 values
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)
    
    def _handle_keyboard(self, key: int) -> bool:
        """
        Handle keyboard input
        
        Returns:
            False if should quit, True otherwise
        """
        if key == ord('q'):
            logger.info("User requested quit")
            return False
        
        elif key == ord('p'):
            self.paused = not self.paused
            state = "Paused" if self.paused else "Resumed"
            logger.info(f"System {state}")
            self.data_logger.log_event('system_state', {'state': state})
        
        elif key == ord('+') or key == ord('='):
            new_speed = self.speed_monitor.get_speed() + 5
            self.speed_monitor.set_speed(new_speed)
            logger.info(f"Speed increased to {new_speed:.1f} km/h")
        
        elif key == ord('-') or key == ord('_'):
            new_speed = max(0, self.speed_monitor.get_speed() - 5)
            self.speed_monitor.set_speed(new_speed)
            logger.info(f"Speed decreased to {new_speed:.1f} km/h")
        
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = config.DATA_DIR / f"screenshot_{timestamp}.jpg"
            logger.info(f"Screenshot functionality - implement if needed")
        
        elif key == ord('r'):
            # Reset statistics
            self.frame_count = 0
            self.fps_history.clear()
            self.processing_times.clear()
            logger.info("Statistics reset")
            self.data_logger.log_event('reset_stats', {})
        
        return True
    
    def _on_speed_change(self, speed: float):
        """Callback for speed changes"""
        self.data_logger.log_event('speed_change', {'speed': speed})
    
    def stop(self):
        """Stop the system"""
        logger.info("Stopping system...")
        self.running = False
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        
        # Release video capture
        if self.cap:
            self.cap.release()
        
        # Release video writer
        if self.video_writer:
            self.video_writer.release()
        
        # Cleanup components
        if hasattr(self, 'alert_system'):
            self.alert_system.cleanup()
        
        if hasattr(self, 'speed_monitor'):
            self.speed_monitor.stop_monitoring()
        
        # Save session data
        if hasattr(self, 'data_logger'):
            summary = self.data_logger.get_summary()
            logger.info("="*50)
            logger.info("Session Summary:")
            logger.info("="*50)
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
            self.data_logger.save_session()
        
        # Close windows
        cv2.destroyAllWindows()
        
        logger.info("Cleanup complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Animal Collision Avoidance System - Prevents animal-vehicle collisions on highways',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use default webcam
  python main.py --source 0                # Use webcam
  python main.py --source video.mp4        # Process video file
  python main.py --source 0 --speed 40     # Set initial speed to 40 km/h
  python main.py --model custom            # Use custom trained model
        """
    )
    parser.add_argument('--source', '-s', type=str, default='0',
                       help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--model', '-m', type=str, default='default',
                       choices=['default', 'custom'],
                       help='Model to use (default or custom trained)')
    parser.add_argument('--speed', type=float, default=config.DEFAULT_SPEED,
                       help=f'Initial speed in km/h (default: {config.DEFAULT_SPEED})')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable video output saving')
    
    args = parser.parse_args()
    
    # Parse video source
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    use_custom = args.model == 'custom'
    
    # Override save setting if requested
    if args.no_save:
        config.SAVE_OUTPUT = False
    
    # Create and start system
    system = AnimalCollisionAvoidanceSystem(
        video_source=video_source,
        use_custom_model=use_custom
    )
    
    # Set initial speed
    system.speed_monitor.set_speed(args.speed)
    
    # Start system
    system.start()

if __name__ == "__main__":
    main()