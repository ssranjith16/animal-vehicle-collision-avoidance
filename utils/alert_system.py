"""
Alert system for collision warnings
"""

import time
import threading
import logging
from enum import Enum
from typing import Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try importing audio library
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not available. Audio alerts disabled.")

# Try importing GPIO (for Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available. Hardware alerts disabled.")

class AlertLevel(Enum):
    """Alert severity levels"""
    SAFE = 0
    CAUTION = 1
    WARNING = 2
    DANGER = 3

class AlertSystem:
    """
    Multi-modal alert system for collision warnings
    Supports visual, audio, and hardware (GPIO) alerts
    """
    
    def __init__(self, sound_file: str = None, 
                 use_gpio: bool = False,
                 buzzer_pin: int = 18,
                 led_pin: int = 23,
                 cooldown_time: float = 3.0):
        """
        Initialize alert system
        
        Args:
            sound_file: Path to alert sound file
            use_gpio: Enable GPIO alerts (Raspberry Pi)
            buzzer_pin: GPIO pin for buzzer
            led_pin: GPIO pin for LED
            cooldown_time: Minimum time between alerts
        """
        self.sound_file = sound_file
        self.use_gpio = use_gpio and GPIO_AVAILABLE
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.cooldown_time = cooldown_time
        
        self.current_level = AlertLevel.SAFE
        self.last_alert_time = 0
        self.alert_active = False
        
        # Initialize audio
        self.audio_available = False
        if PYGAME_AVAILABLE and sound_file:
            try:
                pygame.mixer.init()
                self.alert_sound = pygame.mixer.Sound(sound_file)
                self.audio_available = True
                logger.info("Audio alerts initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio: {e}")
        
        # Initialize GPIO
        if self.use_gpio:
            self._setup_gpio()
        
        # Alert thread
        self.alert_thread = None
        self.stop_thread = False
        
        # Visual alert state
        self.flash_state = False
        self.flash_interval = 0.5  # seconds
        
    def _setup_gpio(self):
        """Setup GPIO pins for hardware alerts"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.buzzer_pin, GPIO.OUT)
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.output(self.led_pin, GPIO.LOW)
            logger.info(f"GPIO initialized: Buzzer={self.buzzer_pin}, LED={self.led_pin}")
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")
            self.use_gpio = False
    
    def trigger_alert(self, level: AlertLevel, 
                     distance: float = None, 
                     ttc: float = None):
        """
        Trigger an alert based on danger level
        
        Args:
            level: Alert severity level
            distance: Distance to obstacle (optional)
            ttc: Time to collision (optional)
        """
        current_time = time.time()
        
        # Check cooldown (except for DANGER level)
        if level != AlertLevel.DANGER:
            if current_time - self.last_alert_time < self.cooldown_time:
                return
        
        self.current_level = level
        self.last_alert_time = current_time
        
        # Log alert
        log_msg = f"Alert: {level.name}"
        if distance:
            log_msg += f" | Distance: {distance:.1f}m"
        if ttc:
            log_msg += f" | TTC: {ttc:.1f}s"
        logger.warning(log_msg)
        
        # Trigger appropriate alerts
        if level == AlertLevel.DANGER:
            self._trigger_danger_alert()
        elif level == AlertLevel.WARNING:
            self._trigger_warning_alert()
        elif level == AlertLevel.CAUTION:
            self._trigger_caution_alert()
        
        # Start alert thread if not running
        if level != AlertLevel.SAFE and not self.alert_active:
            self._start_alert_thread()
    
    def _trigger_danger_alert(self):
        """Maximum alert - continuous"""
        if self.audio_available:
            self.alert_sound.play(loops=-1)  # Loop continuously
        
        if self.use_gpio:
            GPIO.output(self.led_pin, GPIO.HIGH)
            # Buzzer will be controlled by thread
    
    def _trigger_warning_alert(self):
        """Warning alert - intermittent"""
        if self.audio_available:
            self.alert_sound.play()
        
        if self.use_gpio:
            GPIO.output(self.led_pin, GPIO.HIGH)
    
    def _trigger_caution_alert(self):
        """Caution alert - brief"""
        if self.audio_available:
            self.alert_sound.play()
        
        if self.use_gpio:
            # Quick LED flash
            GPIO.output(self.led_pin, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(self.led_pin, GPIO.LOW)
    
    def _start_alert_thread(self):
        """Start background thread for continuous alerts"""
        self.alert_active = True
        self.stop_thread = False
        self.alert_thread = threading.Thread(target=self._alert_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
    
    def _alert_loop(self):
        """Background alert loop for visual/hardware patterns"""
        while not self.stop_thread and self.current_level != AlertLevel.SAFE:
            if self.current_level == AlertLevel.DANGER:
                # Fast flashing for danger
                self.flash_state = not self.flash_state
                if self.use_gpio:
                    GPIO.output(self.buzzer_pin, GPIO.HIGH if self.flash_state else GPIO.LOW)
                time.sleep(0.1)
            
            elif self.current_level == AlertLevel.WARNING:
                # Slower flashing for warning
                self.flash_state = not self.flash_state
                if self.use_gpio:
                    GPIO.output(self.buzzer_pin, GPIO.HIGH)
                    time.sleep(0.2)
                    GPIO.output(self.buzzer_pin, GPIO.LOW)
                time.sleep(0.5)
            
            else:
                time.sleep(0.1)
        
        self.alert_active = False
        self._cleanup_alerts()
    
    def _cleanup_alerts(self):
        """Stop all alerts"""
        if self.audio_available:
            self.alert_sound.stop()
        
        if self.use_gpio:
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.output(self.led_pin, GPIO.LOW)
    
    def clear_alert(self):
        """Clear current alert"""
        self.current_level = AlertLevel.SAFE
        self.stop_thread = True
        
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=1.0)
        
        self._cleanup_alerts()
        logger.info("Alert cleared")
    
    def get_visual_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Add visual alert overlay to frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with alert overlay
        """
        if self.current_level == AlertLevel.SAFE:
            return frame
        
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        if self.current_level == AlertLevel.DANGER:
            # Red flashing border
            color = (0, 0, 255)
            thickness = 10
            message = "!!! BRAKE NOW - COLLISION IMMINENT !!!"
            font_scale = 1.5
            
            # Add semi-transparent red overlay
            if self.flash_state:
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        elif self.current_level == AlertLevel.WARNING:
            # Yellow border
            color = (0, 255, 255)
            thickness = 8
            message = "WARNING: Animal Detected - Reduce Speed"
            font_scale = 1.2
        
        else:  # CAUTION
            # Orange border
            color = (0, 165, 255)
            thickness = 5
            message = "CAUTION: Animal Ahead"
            font_scale = 1.0
        
        # Draw border
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, thickness)
        
        # Draw warning message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 
                                   font_scale, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 4
        
        # Text background
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
        
        return frame
    
    def cleanup(self):
        """Cleanup resources"""
        self.clear_alert()
        if self.use_gpio:
            GPIO.cleanup()
        if self.audio_available:
            pygame.mixer.quit()
        logger.info("Alert system cleaned up")