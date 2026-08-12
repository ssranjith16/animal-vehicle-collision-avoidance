# An Intelligent Driver Assistance Framework for Animal–Vehicle Collision Avoidance Using Computer Vision

## Overview

This project presents a low-cost, vision-based driver assistance framework for detecting animals on roads and highways, estimating their approximate distance from a vehicle, and generating driver alerts to help reduce animal–vehicle collision risk.

The project is focused on Indian road/highway conditions, with cows as the primary target animal. A front-mounted camera captures road video, the system processes the video frames, detects the animal, estimates distance, and generates an alert when required.

## Problem Statement

Animals can suddenly enter or cross highways, while driver reaction time becomes limited at higher speeds. The project addresses this problem by automatically detecting animals in road scenes and providing an early warning to the driver.

## Objectives

- Design a vision-based animal detection system.
- Detect animals from video input.
- Estimate the approximate distance between the vehicle and detected animal.
- Generate driver alerts for collision prevention.
- Develop a low-cost approach suitable for road and highway environments.

## System Workflow

```text
Front-Mounted Camera
        ↓
   Video Capture
        ↓
   Frame Processing
        ↓
 HOG Feature Extraction
        ↓
Boosted Cascade Classifier
        ↓
   Animal Detection
        ↓
 Distance Estimation
        ↓
 Collision Risk Analysis
        ↓
     Driver Alert
```

## Dataset

A custom dataset was prepared for the project because the work focused on animal detection in Indian road conditions.

- Positive samples: **700** animal images
- Negative samples: **1,500** road/background images
- Primary target: **Cow**
- Images collected under different weather and lighting conditions

## Computer Vision Approach

### HOG Feature Extraction

Histogram of Oriented Gradients (HOG) is used to capture edge, shape, and contour information from animal images. HOG was selected because of its computational efficiency and suitability for real-time computer-vision detection.

### Boosted Cascade Classifier

A boosted cascade classifier based on Gentle AdaBoost is used for animal detection. Multiple weak classifiers are combined into a stronger classifier, while the cascade structure allows non-animal regions to be rejected early.

### Training Parameters

- Positive samples: 700
- Negative samples: 1,500
- Cascade stages: 20
- Feature type: HOG
- Reported training time: approximately 14 hours

## Distance Estimation

After an animal is detected, its position in the image is used for approximate distance estimation. The system uses camera calibration and a polynomial relationship to convert pixel measurements into real-world distance in meters.

The project report states an observed distance-estimation error of **less than 2%**.

## Driver Alert System

The repository also contains `alert_system.py`, which provides the alert component for the driver-assistance concept. It supports different alert severity levels and optional audio/GPIO hardware integration.

```text
SAFE → CAUTION → WARNING → DANGER
```

## Experimental Setup

- Front-mounted vehicle camera
- Video resolution: **640 × 480**
- Frame rate: **30 FPS**
- Tools: **OpenCV and Visual Studio**
- Tested at multiple vehicle speeds and road/weather conditions

## Reported Results

| Metric | Reported Result |
|---|---:|
| Sensitivity / True Positive Rate | **80.4%** |
| Specificity / True Negative Rate | **83.5%** |
| Overall detection accuracy | **82.5%** |
| Maximum detection distance | **20 m** |
| Effective collision avoidance | **Up to approximately 35 km/h** |
| Distance estimation error | **< 2%** |

These figures are the results reported in the project report/presentation and are not a newly reproduced benchmark in this repository.

## Repository Contents

```text
animal-vehicle-collision-avoidance/
├── README.md
├── alert_system.py
├── Identify the Animal_colab.ipynb
├── Identify_the_Animal_novalid_colab.ipynb
└── subm01.csv
```

## Limitations

- Detection is limited by lighting and road-scene conditions.
- Performance can decrease at higher vehicle speeds.
- Detection range is limited by the camera and detection setup.
- The reported system primarily focuses on cows.

## Future Work

- Use modern deep-learning object-detection models to improve robustness and accuracy.
- Extend detection to additional animal classes.
- Improve night-time detection using thermal or low-light cameras.
- Integrate the system with embedded automotive hardware.
- Combine animal, pedestrian, and vehicle detection for a broader road-safety system.

## Disclaimer

This project is an academic/research driver-assistance concept. It should not be treated as a certified automotive safety system or as a replacement for driver attention.
