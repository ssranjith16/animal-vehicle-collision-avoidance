"""
Configuration settings for Animal Collision Avoidance System
"""

import os
from pathlib import Path

class Config:
    # Project paths
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / 'models'
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    SOUNDS_DIR = BASE_DIR / 'sounds'
    
    # Create directories if they don't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Model settings
    MODEL_PATH = MODELS_DIR / 'yolov8n.pt'
    CUSTOM_MODEL_PATH = MODELS_DIR / 'custom_animal_detector.pt'
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Animal classes (COCO dataset IDs)
    # Add custom classes as needed
    ANIMAL_CLASSES = {
        16: 'dog',
        17: 'cat',
        18: 'horse',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        14: 'bird',
        15: 'cat'  # duplicate for completeness
    }
    
    # Indian highway specific animals (custom trained)
    INDIAN_ANIMALS = ['cow', 'buffalo', 'dog', 'nilgai', 'camel', 'elephant']
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Distance estimation parameters
    # These need calibration for your specific camera
    KNOWN_ANIMAL_HEIGHTS = {
        'cow': 1.4,      # meters
        'buffalo': 1.5,  # meters
        'dog': 0.6,      # meters
        'nilgai': 1.3,   # meters
        'camel': 2.0,    # meters
        'elephant': 2.5, # meters
        'default': 1.2   # meters
    }
    
    # Focal length (calculate using calibration)
    FOCAL_LENGTH = 625  # pixels
    
    # Safety parameters
    TWO_SECOND_RULE = 2.0  # seconds
    MIN_ALERT_DISTANCE = 30.0  # meters
    MAX_DETECTION_DISTANCE = 100.0  # meters
    
    # Alert settings
    ALERT_COOLDOWN = 3.0  # seconds between alerts
    ALERT_SOUND_PATH = SOUNDS_DIR / 'alert.wav'
    
    # Speed settings (km/h)
    DEFAULT_SPEED = 30.0
    MAX_SYSTEM_SPEED = 35.0  # As per thesis
    
    # Video settings
    VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
    SAVE_OUTPUT = True
    OUTPUT_VIDEO_PATH = DATA_DIR / 'output_videos'
    
    # Hardware settings (Raspberry Pi)
    USE_GPIO = False
    BUZZER_PIN = 18
    LED_PIN = 23
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = LOGS_DIR / 'system.log'
    SAVE_DETECTION_IMAGES = True
    DETECTION_IMAGES_DIR = DATA_DIR / 'detections'
    
    # Performance
    PROCESS_EVERY_N_FRAMES = 1  # Process every frame
    USE_GPU = True
    USE_HALF_PRECISION = True  # FP16 for faster inference

config = Config()