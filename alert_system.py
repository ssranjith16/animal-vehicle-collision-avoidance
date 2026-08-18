"""
Alert system for collision warnings.

Supports visual, optional audio, and optional Raspberry Pi GPIO alerts.
"""

import logging
import threading
import time
from enum import Enum
from typing import Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    GPIO = None

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    SAFE = 0
    CAUTION = 1
    WARNING = 2
    DANGER = 3


class AlertSystem:
    """Multi-modal collision warning alert system."""

    def __init__(
        self,
        sound_file: Optional[str] = None,
        use_gpio: bool = False,
        buzzer_pin: int = 18,
        led_pin: int = 23,
        cooldown_time: float = 3.0,
    ):
        self.sound_file = sound_file
        self.use_gpio = use_gpio and GPIO_AVAILABLE
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.cooldown_time = cooldown_time
        self.current_level = AlertLevel.SAFE
        self.last_alert_time = 0.0
        self.alert_active = False
        self.audio_available = False
        self.alert_sound = None
        self.alert_thread = None
        self.stop_thread = False
        self.flash_state = False
        self.flash_interval = 0.5

        if PYGAME_AVAILABLE and sound_file:
            try:
                pygame.mixer.init()
                self.alert_sound = pygame.mixer.Sound(sound_file)
                self.audio_available = True
            except Exception as exc:
                logger.warning("Failed to initialize audio: %s", exc)

        if self.use_gpio:
            self._setup_gpio()

    def _setup_gpio(self) -> None:
        """Initialize Raspberry Pi GPIO outputs when GPIO is enabled."""
        if not GPIO_AVAILABLE:
            logger.warning("GPIO is not available; hardware alerts disabled.")
            self.use_gpio = False
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.buzzer_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)

    def set_alert_level(self, level: AlertLevel) -> None:
        """Set the current alert severity."""
        self.current_level = level

    def trigger(self, level: AlertLevel) -> bool:
        """Trigger an alert if the cooldown period has elapsed.

        Returns True when an alert was triggered and False when suppressed
        by the cooldown period.
        """
        self.set_alert_level(level)
        now = time.monotonic()

        if level == AlertLevel.SAFE:
            self.stop_alert()
            return False

        if now - self.last_alert_time < self.cooldown_time:
            return False

        self.last_alert_time = now
        self.alert_active = True
        self._visual_alert(level)
        self._audio_alert(level)
        self._gpio_alert(level)
        return True

    def _visual_alert(self, level: AlertLevel) -> None:
        """Log the alert level for console/UI integrations."""
        logger.warning("Collision alert: %s", level.name)

    def _audio_alert(self, level: AlertLevel) -> None:
        """Play the configured alert sound when audio is available."""
        if not self.audio_available or self.alert_sound is None:
            return
        try:
            self.alert_sound.play()
        except Exception as exc:
            logger.warning("Unable to play alert sound: %s", exc)

    def _gpio_alert(self, level: AlertLevel) -> None:
        """Control Raspberry Pi buzzer/LED outputs when enabled."""
        if not self.use_gpio or not GPIO_AVAILABLE:
            return

        danger = level in (AlertLevel.WARNING, AlertLevel.DANGER)
        GPIO.output(self.led_pin, GPIO.HIGH)
        GPIO.output(self.buzzer_pin, GPIO.HIGH if danger else GPIO.LOW)

    def stop_alert(self) -> None:
        """Stop active audio/GPIO alerts and return to SAFE state."""
        self.alert_active = False
        self.current_level = AlertLevel.SAFE

        if self.audio_available and self.alert_sound is not None:
            try:
                self.alert_sound.stop()
            except Exception as exc:
                logger.debug("Unable to stop alert sound: %s", exc)

        if self.use_gpio and GPIO_AVAILABLE:
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.output(self.led_pin, GPIO.LOW)

    def cleanup(self) -> None:
        """Release audio and GPIO resources."""
        self.stop_alert()
        if self.use_gpio and GPIO_AVAILABLE:
            GPIO.cleanup([self.buzzer_pin, self.led_pin])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    alert_system = AlertSystem()
    alert_system.trigger(AlertLevel.CAUTION)
    alert_system.stop_alert()
    print("Alert system self-test completed successfully.")
