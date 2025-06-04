import time
from typing import List, Dict, Any
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.actions.abnormal_action_config import (
    ABNORMAL_ACTION_SEVERITIES, 
    DURATION_THRESHOLD_WARN_LEVEL_1, 
    DURATION_THRESHOLD_WARN_LEVEL_2, 
    DURATION_THRESHOLD_WARN_LEVEL_3
)

from src.config import ANOMALY_MAP, ANOMALY_LABELS, ANOMALY_CONFIDENCE_THRESHOLD
from src.utils.logger import setup_logger
try:
    from src.utils.audio_alerts import play_warning_sound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio alerts not available")

logger = setup_logger(__name__)

class AnomalyScorer:
    def __init__(self, cfg: Dict[str, Any]):
        self.anomaly_map = ANOMALY_MAP
        self.anomaly_labels = ANOMALY_LABELS
        self.confidence_threshold = cfg.get('anomaly_conf_thresh', ANOMALY_CONFIDENCE_THRESHOLD)
        
        logger.info(f"AnomalyScorer initialized with {len(self.anomaly_map)} mapped actions")
        logger.debug(f"Anomaly confidence threshold: {self.confidence_threshold}")
        
        self.active_persistent_abnormalities = {}

    def score(self, tracks, actions, frame):
        anomaly_events = []
        current_time = time.time()
        actions_map = {action['track_id']: action for action in actions}
        active_track_ids_in_frame = set()

        for track in tracks:
            track_id = track['track_id']
            active_track_ids_in_frame.add(track_id)
            current_warning_level = 0

            if track_id in actions_map:
                action_info = actions_map[track_id]
                action_label = action_info.get('action_label', 'unknown')
                is_abnormal = action_info.get('is_abnormal', False)
                abnormal_duration = action_info.get('abnormal_duration', 0.0)
                base_severity = action_info.get('base_severity', 0)

                if is_abnormal:
                    if base_severity >= 3 and abnormal_duration >= DURATION_THRESHOLD_WARN_LEVEL_3:
                        current_warning_level = 3
                    elif base_severity >= 2 and abnormal_duration >= DURATION_THRESHOLD_WARN_LEVEL_2:
                        current_warning_level = 2
                    elif base_severity >= 1 and abnormal_duration >= DURATION_THRESHOLD_WARN_LEVEL_1:
                        current_warning_level = 1

                    if current_warning_level > 0:
                        description = f"Track {track_id}: {action_label} (Sev:{base_severity}) for {abnormal_duration:.1f}s - WARNING Level {current_warning_level}"
                        event = {
                            'track_id': track_id,
                            'anomaly_type': 'persistent_abnormal_action',
                            'description': description,
                            'action_label': action_label,
                            'duration': abnormal_duration,
                            'base_severity': base_severity,
                            'warning_level': current_warning_level
                        }
                        anomaly_events.append(event)
                        self.active_persistent_abnormalities[track_id] = (action_label, current_warning_level, current_time)
                    elif track_id in self.active_persistent_abnormalities:
                        del self.active_persistent_abnormalities[track_id]
                
                else:
                    if track_id in self.active_persistent_abnormalities:
                        del self.active_persistent_abnormalities[track_id]
            
            elif track_id in self.active_persistent_abnormalities:
                del self.active_persistent_abnormalities[track_id]
        for track_id_key in list(self.active_persistent_abnormalities.keys()):
            if track_id_key not in active_track_ids_in_frame:
                del self.active_persistent_abnormalities[track_id_key]
                
        return anomaly_events 

    def score_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        anomaly_scores = []
        
        for action in actions:
            action_id = action['action_class_id']
            action_name = action['action_name']
            action_conf = action['action_confidence']
            track_id = action.get('track_id', None)
            
            warning_level = 0
            warning_message = 'Activity: Normal'
            anomaly_score = 0.0
            
            if action_id in self.anomaly_map:
                anomaly_idx = self.anomaly_map[action_id]
                anomaly_type = self.anomaly_labels[anomaly_idx]
                anomaly_score = action_conf
                
                if anomaly_score >= self.confidence_threshold:
                    if anomaly_idx == 1:
                        warning_level = 3
                        warning_message = f'[!!!] VIOLENCE DETECTED: {action_name} ({action_conf:.2f})'
                        logger.warning(f"SEVERE: Violence detected - {action_name} with confidence {action_conf:.3f}")
                        
                        if AUDIO_AVAILABLE:
                            play_warning_sound(3, action_name)
                            
                    elif anomaly_idx == 2:
                        warning_level = 2
                        warning_message = f'[!] FALL DETECTED: {action_name} ({action_conf:.2f})'
                        logger.warning(f"HIGH: Fall detected - {action_name} with confidence {action_conf:.3f}")
                        
                        if AUDIO_AVAILABLE:
                            play_warning_sound(2, action_name)
                            
                    elif anomaly_idx == 3:
                        warning_level = 1
                        warning_message = f'[*] SUSPICIOUS ACTIVITY: {action_name} ({action_conf:.2f})'
                        logger.warning(f"MEDIUM: Suspicious activity - {action_name} with confidence {action_conf:.3f}")
                        
                        if AUDIO_AVAILABLE:
                            play_warning_sound(1, action_name)
                    
                    logger.info(f"Anomaly detected: {warning_message} (score: {anomaly_score:.3f})")
                else:
                    logger.debug(f"Potential anomaly below threshold: {action_name} (conf: {action_conf:.3f} < {self.confidence_threshold})")
            else:
                # For normal actions, still show them in the UI with a base score
                if action_conf > 0.5:  # Only show high-confidence normal actions
                    anomaly_score = action_conf * 0.1  # Give normal actions a low score for display
                    warning_message = f'Normal: {action_name} ({action_conf:.2f})'
                    logger.debug(f"Normal activity detected: {action_name} (conf: {action_conf:.3f})")
            
            # Only add to results if there's some meaningful score or it's an anomaly
            if anomaly_score > 0.05 or warning_level > 0:
                anomaly_scores.append({
                    'track_id': track_id if track_id is not None else -1,
                    'warning_level': warning_level,
                    'warning_message': warning_message,
                    'description': warning_message,  # For compatibility with visualization
                    'anomaly_score': anomaly_score,
                    'action_name': action_name,
                    'action_id': action_id
                })
        
        return anomaly_scores
    
    def aggregate_scores(self, anomaly_scores: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate anomaly scores across all detections/tracks.
        
        Returns a dictionary with overall anomaly scores for each type.
        """
        aggregated = {label: 0.0 for label in self.anomaly_labels.values()}
        
        for score in anomaly_scores:
            anomaly_type = score['anomaly_type']
            if anomaly_type in aggregated:
                # Take max score for each anomaly type
                aggregated[anomaly_type] = max(aggregated[anomaly_type], score['anomaly_score'])
        
        return aggregated

# Example usage
if __name__ == '__main__':
    # Test the scorer
    dummy_cfg = {'anomaly_conf_thresh': 0.3}
    scorer = AnomalyScorer(dummy_cfg)
    
    # Simulate some action predictions
    test_actions = [
        {'action_class_id': 259, 'action_name': 'punching person (boxing)', 
         'action_confidence': 0.85, 'track_id': 1},
        {'action_class_id': 147, 'action_name': 'gymnastics tumbling', 
         'action_confidence': 0.92, 'track_id': 2},
        {'action_class_id': 100, 'action_name': 'drinking', 
         'action_confidence': 0.75, 'track_id': 3},
    ]
    
    anomaly_results = scorer.score_actions(test_actions)
    
    print("\nAnomaly Scoring Results:")
    for result in anomaly_results:
        print(f"  Track {result['track_id']}: Level {result['warning_level']} - {result['warning_message']}")
    
    # Test aggregation
    aggregated = scorer.aggregate_scores(anomaly_results)
    print("\nAggregated Anomaly Scores:")
    for anomaly_type, score in aggregated.items():
        print(f"  {anomaly_type}: {score:.3f}") 