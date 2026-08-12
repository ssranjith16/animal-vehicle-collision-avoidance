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
                 led_pin : int = 23,
                 cooldown_time: float = 3.0):
        """
        Initialize alert system
        
        Arg:
            sound_file: Path to alert sound file
            use_gpio: Enable GPIO alerts (Raspberry Pi)
            buzzer_pin: GPIO pin for buzzer
            led_pin: GPIO pin for LED
            cooldown_time: Minimum time between alerts
        """
        self.sound_file = sound_file
        self.use_gpio= use_gpio and GPIO_AVAILABLE
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.cooldown_time = cooldown_time
        self.current_level = AlertLevel.SAFE
        self.last_alert_time = 0
        self.alert_active = False
        self.audio_available = False
        if PYGAME_AVAILABLE and sound_file:
            try:
                pygame.mixer.init()
                self.alert_sound = pygame.mixer.Sound(sound_file)
                self.audio_available = True
            except Exception as e:
                logger.error(f"Failed to initialize audio: {e}")
         if self.use_gpio:
            self._setup_gpio()
        self.alert_thread = None
        self.stop_thread = False
        self.flash_state = False
        self.flash_interval = 0.5
