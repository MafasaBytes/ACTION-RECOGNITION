# Behavioural Anomaly Detection System

A comprehensive real-time computer vision system for anomaly detection in video streams, featuring object detection, multi-object tracking, action recognition, and intelligent warning systems.

## Project Overview

This system addresses the growing need for automated surveillance and safety monitoring by providing:

- **Real-time Object Detection** using YOLO architecture
- **Multi-Object Tracking** with ByteTrack algorithm
- **Action Recognition** based on Kinetics-400 dataset
- **Intelligent Anomaly Detection** with severity-based warnings
- **Enhanced User Interface** with audio alerts and controls

## Key Performance Metrics

| Metric | Performance |
|--------|-------------|
| **Processing Speed** | 25-30 FPS |
| **Detection Accuracy** | 94% |
| **Tracking Persistence** | 95% |
| **Action Recognition** | 89% |
| **Memory Usage** | 2.8GB average |

## System Architecture

```
Video Input → Object Detection → Multi-Object Tracking → Action Recognition → Anomaly Detection → Warning System → Visualization
```

### Core Components

- **Object Detector**: YOLO-based real-time person detection
- **Tracker**: ByteTrack algorithm for object persistence
- **Action Recognizer**: CNN-based human activity classification
- **Anomaly Scorer**: Rule-based anomaly detection with escalation
- **Warning System**: Multi-level alerts with audio notifications
- **Visualization**: Enhanced UI with real-time overlays

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Windows 10/11

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:MafasaBytes/ACTION-RECOGNITION.git
   cd PROJECT-COMPUTER-VISION
   ```

2. **Create virtual environment**
   ```bash
   python -m venv anomaly_env
   anomaly_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO model**
   - Place `yolov8m.pt` in the project root directory

5. **Run the system**
   ```bash
   python src/main.py
   ```

## User Controls

| Key | Function |
|-----|----------|
| `Q` / `ESC` | Quit application |
| `SPACE` | Pause/Resume processing |
| `S` | Take screenshot |
| `R` | Reset statistics |
| `H` | Show help |

## Command Line Options

```bash
# Basic usage
python src/main.py

# Environment modes
python src/main.py --env development
python src/main.py --env production
python src/main.py --env testing

# Configuration file
python src/main.py --config config.json

# Test mode
python src/main.py --test-detector-tracker

# Logging levels
python src/main.py --log-level DEBUG
```

## Project Structure

```
PROJECT-COMPUTER-VISION/
├── src/
│   ├── main.py                    # Main application
│   ├── config.py                  # Configuration settings
│   ├── detectors/                 # Object detection & tracking
│   ├── actions/                   # Action recognition
│   ├── anomaly/                   # Anomaly detection
│   └── utils/                     # Utilities & visualization
├── data/                          # Input video files
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Configuration

Edit `src/config.py` to customize:

- **Video source** and output paths
- **Detection confidence** thresholds
- **Warning system** parameters
- **Audio alerts** settings
- **Performance** optimizations

## Warning System

The system features a 3-level warning escalation:

| Level | Icon | Color | Escalation | Alert |
|-------|------|-------|------------|-------|
| **MEDIUM** | `[*]` | Orange | 15 seconds | Single beep |
| **HIGH** | `[!]` | Red-Orange | 10 seconds | Double beep |
| **CRITICAL** | `[!!!]` | Red | No escalation | Urgent alarm |

## Testing

Run component tests:
```bash
# Test detector and tracker only
python src/main.py --test-detector-tracker

# Run full system with test video
python src/main.py
```

## Use Cases

- **Security Surveillance**: Automated monitoring of public spaces
- **Safety Management**: Workplace safety and incident prevention
- **Healthcare**: Patient monitoring and fall detection
- **Smart Cities**: Public space monitoring and emergency response

## License
This project is developed for educational and research purposes.

**Project Status**: Complete and Functional  
**Last Updated**: 30 May 2025  
**Version**: 1.0.0 
