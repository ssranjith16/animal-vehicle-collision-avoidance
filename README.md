# AI-Powered Animal-Vehicle Collision Avoidance System

A computer-vision prototype for detecting animals on highways and generating collision-risk warnings using **YOLOv8**, monocular distance estimation, vehicle-speed monitoring, and multi-modal alerts.

## Features

- Real-time animal detection with YOLOv8
- Highway animal detection
- IoU-based object tracking
- Monocular distance estimation
- Time-to-Collision (TTC) calculation
- SAFE / CAUTION / WARNING / DANGER risk levels
- Visual and audio alerts
- Optional Raspberry Pi GPIO buzzer/LED integration
- Vehicle speed monitoring
- CSV/JSON session logging
- Camera calibration utilities
- Unit tests

## Project Structure

```text
animal-vehicle-collision-avoidance/
├── main.py
├── train.py
├── config.py
├── setup.py
├── requirements.txt
├── data/
│   ├── indian_highway_animals.yaml
│   └── custom_dataset.yaml
├── models/
│   ├── detector.py
│   └── train.py
├── utils/
│   ├── alert_system.py
│   ├── camera_calibration.py
│   ├── data_logger.py
│   ├── distance_estimator.py
│   └── speed_monitor.py
└── tests/
    ├── test_detector.py
    └── test_distance.py
```

## Installation

Python 3.8+ is recommended.

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Training

The project supports YOLOv8 training for animal detection.

```bash
python models/train.py --data dataset
```

## Distance Estimation

Distance is estimated using triangle similarity:

```text
distance = (focal_length × real_object_height) / pixel_object_height
```

TTC is calculated using estimated distance and vehicle speed.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Notes

Large datasets, generated videos, logs, virtual environments, caches, and model-weight binaries should not be committed to GitHub.

This project is a research/prototype system and is not a certified automotive safety system.
