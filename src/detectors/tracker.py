import numpy as np
import torch # For converting detections to tensor for ByteTrack
from bytetracker import BYTETracker
from typing import List, Dict, Tuple, Any

from src.config import (
    TRACK_HIGH_THRESH,
    TRACK_LOW_THRESH,
    NEW_TRACK_THRESH,
    TRACK_BUFFER,
    MATCH_THRESH,
    DEFAULT_VIDEO_FPS
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class Tracker:
    def __init__(self, cfg: Dict[str, Any]):
        self.track_high_thresh = cfg.get('track_high_thresh', TRACK_HIGH_THRESH)
        self.track_low_thresh = cfg.get('track_low_thresh', TRACK_LOW_THRESH)
        self.new_track_thresh = cfg.get('new_track_thresh', NEW_TRACK_THRESH)
        self.track_buffer = cfg.get('track_buffer', TRACK_BUFFER) # Number of frames to keep a track
        self.match_thresh = cfg.get('match_thresh', MATCH_THRESH) # IoU threshold for matching
        self.frame_rate = cfg.get('video_fps', DEFAULT_VIDEO_FPS)

        try:
            self.tracker = BYTETracker(
                track_thresh=self.track_high_thresh, # High confidence detection threshold
                track_buffer=self.track_buffer,      # Buffer frames for lost tracks
                match_thresh=self.match_thresh,      # Matching threshold for IoU
                frame_rate=self.frame_rate
            )
            logger.info("ByteTrack tracker initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing ByteTrack: {e}")
            raise

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        if not detections:
            logger.debug("No detections, updating tracker with empty array.")
            dummy_detections_for_tracker = np.empty((0, 6))  # x1,y1,x2,y2,score,class_id
            online_targets = self.tracker.update(torch.tensor(dummy_detections_for_tracker).float().cpu(), 
                                                 frame.shape[:2])
        else:
            # Create a mapping to store detection info by array index
            detection_info_map = {}
            formatted_detections = []

            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                score = det['confidence']
                class_id = det['class_id']  # Include class_id for ByteTrack
                
                # Store original detection info with index
                detection_info_map[i] = {
                    'class_name': det['class_name'],
                    'class_id': det['class_id'],
                    'original_confidence': score,
                    'original_bbox': [x1, y1, x2, y2]
                }
                
                # ByteTrack format: [x1, y1, x2, y2, score, class_id]
                formatted_detections.append([x1, y1, x2, y2, score, class_id])
            
            if formatted_detections:
                np_detections = np.array(formatted_detections, dtype=float)
                tensor_detections = torch.tensor(np_detections).float().cpu()
                logger.debug(f"Tracker:update - Input to ByteTrack: {tensor_detections.shape}")
                
                online_targets = self.tracker.update(tensor_detections, frame.shape[:2])
            else:
                online_targets = []

        tracked_objects = []
        if online_targets is not None and len(online_targets) > 0:
            logger.debug(f"ByteTracker returned {len(online_targets)} tracks")
            
            for i, t_obj in enumerate(online_targets):
                try:
                    # ByteTrack typically returns numpy arrays or STrack objects
                    if hasattr(t_obj, 'tlbr') and hasattr(t_obj, 'track_id'):
                        # STrack object
                        x1, y1, x2, y2 = map(int, t_obj.tlbr)
                        track_id = int(t_obj.track_id)
                        score = float(t_obj.score)
                        
                        # For STrack, we need to find the closest matching detection
                        best_match_class = "person"  # Default assumption for most tracking scenarios
                        best_match_id = 0
                        
                    elif isinstance(t_obj, np.ndarray):
                        # NumPy array format from ByteTrack
                        if len(t_obj) >= 6:  # [x1,y1,x2,y2,track_id,score,class_id]
                            x1, y1, x2, y2 = map(int, t_obj[0:4])
                            track_id = int(t_obj[4])
                            score = float(t_obj[5])
                            
                            # Try to get class_id if available
                            if len(t_obj) > 6:
                                class_id_from_track = int(t_obj[6])
                            else:
                                class_id_from_track = 0  # Default to first class
                            
                            # Find matching class name from original detections
                            best_match_class = "person"  # Default
                            best_match_id = class_id_from_track
                            
                            # Try to match by class_id first
                            for det in detections:
                                if det['class_id'] == class_id_from_track:
                                    best_match_class = det['class_name']
                                    best_match_id = det['class_id']
                                    break
                            
                            # If no class match found, use IoU matching as fallback
                            if best_match_class == "person" and class_id_from_track != 1:
                                best_iou = 0
                                track_bbox = [x1, y1, x2, y2]
                                for det in detections:
                                    det_bbox = det['bbox']
                                    iou = self._calculate_iou(track_bbox, det_bbox)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_match_class = det['class_name']
                                        best_match_id = det['class_id']
                        
                        elif len(t_obj) >= 5:  # Fallback for [x1,y1,x2,y2,track_id]
                            x1, y1, x2, y2 = map(int, t_obj[0:4])
                            track_id = int(t_obj[4])
                            score = 0.5  # Default score
                            best_match_class = "person"
                            best_match_id = 1
                        else:
                            logger.warning(f"Unexpected track array format: {t_obj}")
                            continue
                    else:
                        logger.warning(f"Unknown track object type: {type(t_obj)}")
                        continue
                    
                    tracked_objects.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': score,
                        'class_id': best_match_id,
                        'class_name': best_match_class,
                        'object_id': track_id
                    })
                    
                    logger.debug(f"Track {track_id}: {best_match_class} at [{x1},{y1},{x2},{y2}]")
                    
                except Exception as e:
                    logger.error(f"Error processing track object {i}: {e}")
                    continue
        
        logger.info(f"Tracker: {len(detections)} detections -> {len(tracked_objects)} tracks")
        return tracked_objects

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

if __name__ == '__main__':
    dummy_cfg = {
        'video_fps': 30,
        'track_high_thresh': 0.6,
        'track_buffer': 30,
        'match_thresh': 0.8,
    }
    tracker = Tracker(dummy_cfg)
    logger.info("Tracker initialized for testing.")

    # Simulate some frames and detections
    frame_height, frame_width = 480, 640
    dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Frame 1: Two detections
    detections_frame1 = [
        {'bbox': [50, 50, 100, 100], 'confidence': 0.9, 'class_id': 0, 'class_name': 'person'},
        {'bbox': [150, 150, 200, 200], 'confidence': 0.8, 'class_id': 2, 'class_name': 'car'}
    ]
    logger.info("\nProcessing Frame 1...")
    tracked_frame1 = tracker.update(detections_frame1, dummy_frame)
    logger.info(f"Frame 1 Tracks: {tracked_frame1}")
    assert len(tracked_frame1) == 2
    assert tracked_frame1[0]['object_id'] is not None

    # Frame 2: One detection moves, one new one
    detections_frame2 = [
        {'bbox': [60, 60, 110, 110], 'confidence': 0.92, 'class_id': 0, 'class_name': 'person'}, # Moved person
        {'bbox': [250, 250, 300, 300], 'confidence': 0.7, 'class_id': 0, 'class_name': 'person'} # New person
    ]
    logger.info("\nProcessing Frame 2...")
    tracked_frame2 = tracker.update(detections_frame2, dummy_frame)
    logger.info(f"Frame 2 Tracks: {tracked_frame2}")
    # Expected: first person track ID should be same, new track ID for second person
    # car from frame 1 might still be tracked or marked inactive depending on buffer
    person_track_ids_f2 = [t['object_id'] for t in tracked_frame2 if t['class_name'] == 'person']
    assert len(person_track_ids_f2) == 2
    assert tracked_frame1[0]['object_id'] == [t['object_id'] for t in tracked_frame2 if t['bbox'] == [60,60,110,110]][0]

    # Frame 3: No detections
    logger.info("\nProcessing Frame 3 (no detections)...")
    tracked_frame3 = tracker.update([], dummy_frame)
    logger.info(f"Frame 3 Tracks (after no detections): {tracked_frame3}")
    # Tracks from frame 2 should still be there if buffer is large enough

    logger.info("\nTracker test finished.")