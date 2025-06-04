import os
from src.utils.kinetics_classes import load_kinetics_400_classes, get_anomaly_relevant_classes

# Frame settings for action recognition
NUM_FRAMES = 32  # Number of frames for each action recognition clip
FRAME_STRIDE = 2 # Stride for sampling frames, if applicable

# Object Detection
DETECTOR_MODEL_NAME = "facebook/detr-resnet-101"
# Default confidence threshold for object detection, can be overridden by main cfg
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.5

# Action Recognition
ACTION_MODEL_NAME = "slowfast_r50"
# Default confidence threshold for an action to be considered, can be overridden by main cfg
# Note: Global Actions are maintained without confidence constraints
ACTION_PREDICTION_THRESHOLD = 0.1

# Action Stabilization Settings
ENABLE_ACTION_STABILIZATION = True
MIN_STABLE_CONFIDENCE = 0.0  # No minimum confidence constraint for Global Actions
STABILITY_THRESHOLD = 0.7    # Threshold for action stability score
ACTION_HISTORY_LENGTH = 8    # Number of recent predictions to consider

# Anomaly Scoring
# Confidence threshold for a mapped anomaly to be considered significant
ANOMALY_CONFIDENCE_THRESHOLD = 0.3

ANOMALY_LABELS = {
    0: "Normal",
    1: "Violence",
    2: "Falling",
    3: "Suspicious Object Interaction",
}

# Load Kinetics-400 class names
try:
    KINETICS_CLASSES = load_kinetics_400_classes()
except Exception as e:
    KINETICS_CLASSES = [f"kinetics_class_{i}" for i in range(400)]

# Load anomaly mapping
try:
    ANOMALY_MAP = get_anomaly_relevant_classes()
    # Create set of abnormal action IDs for consistent use across the system
    ABNORMAL_ACTION_IDS = set(ANOMALY_MAP.keys())
    print(f"Loaded {len(ANOMALY_MAP)} anomaly mappings")
except Exception as e:
    print(f"Error loading anomaly mappings: {e}")
    ANOMALY_MAP = {}
    ABNORMAL_ACTION_IDS = set()

# Video processing settings
DEFAULT_VIDEO_FPS = 30 # Used if FPS cannot be determined from video source

# Tracker settings (sensible defaults for ByteTrack)
TRACK_HIGH_THRESH = 0.6  # High detection threshold for matching
TRACK_LOW_THRESH = 0.1   # Low detection threshold for initializing new tracks
NEW_TRACK_THRESH = 0.7   # Threshold for considering a new track from low-score detections
TRACK_BUFFER = 30        # Number of frames to buffer a track if it's not detected
MATCH_THRESH = 0.8       # IoU threshold for matching tracks with detections 