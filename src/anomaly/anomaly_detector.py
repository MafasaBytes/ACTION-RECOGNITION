import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BehavioralAnomalyDetector:
    
    def __init__(self, config: Dict):
        self.config = config
        self.track_history = {}
        self.zone_definitions = config.get('zones', {})
        self.crowd_threshold = config.get('crowd_threshold', 10)
        self.loitering_time_threshold = config.get('loitering_time_seconds', 30)
        self.speed_threshold = config.get('speed_threshold', 50)
        self.abandoned_object_time = config.get('abandoned_object_seconds', 60)
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        
        self.prev_frame = None
        self.motion_history = deque(maxlen=30)
        
    def analyze_frame(self, frame: np.ndarray, tracks: List[Dict], 
                     timestamp: datetime) -> List[Dict]:
        anomalies = []
        
        self._update_track_history(tracks, timestamp)
        
        crowd_anomaly = self._detect_crowd_density(tracks, frame.shape)
        if crowd_anomaly:
            anomalies.append(crowd_anomaly)
        
        loitering_anomalies = self._detect_loitering(timestamp)
        anomalies.extend(loitering_anomalies)
        
        speed_anomalies = self._detect_unusual_speed(timestamp)
        anomalies.extend(speed_anomalies)
        
        zone_anomalies = self._detect_zone_intrusion(tracks)
        anomalies.extend(zone_anomalies)
        
        motion_anomaly = self._analyze_motion_patterns(frame)
        if motion_anomaly:
            anomalies.append(motion_anomaly)
        
        abandoned_anomalies = self._detect_abandoned_objects(tracks, timestamp)
        anomalies.extend(abandoned_anomalies)
        
        return anomalies
    
    def _update_track_history(self, tracks: List[Dict], timestamp: datetime):
        current_track_ids = set()
        
        for track in tracks:
            track_id = track['object_id']
            current_track_ids.add(track_id)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'positions': deque(maxlen=100),
                    'timestamps': deque(maxlen=100),
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'class_name': track['class_name']
                }
            
            bbox = track['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            self.track_history[track_id]['positions'].append(center)
            self.track_history[track_id]['timestamps'].append(timestamp)
            self.track_history[track_id]['last_seen'] = timestamp
        
        cutoff_time = timestamp - timedelta(minutes=5)
        tracks_to_remove = [
            track_id for track_id, history in self.track_history.items()
            if history['last_seen'] < cutoff_time
        ]
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
    
    def _detect_crowd_density(self, tracks: List[Dict], frame_shape: Tuple) -> Optional[Dict]:
        person_count = sum(1 for track in tracks if track['class_name'] == 'person')
        
        if person_count > self.crowd_threshold:
            frame_area = frame_shape[0] * frame_shape[1]
            density = person_count / (frame_area / 1000000)
            
            return {
                'type': 'crowd_density',
                'severity': 'high' if person_count > self.crowd_threshold * 1.5 else 'medium',
                'description': f'High crowd density detected: {person_count} people',
                'confidence': min(0.9, person_count / self.crowd_threshold),
                'metadata': {
                    'person_count': person_count,
                    'density': density
                }
            }
        return None
    
    def _detect_loitering(self, current_time: datetime) -> List[Dict]:
        anomalies = []
        
        for track_id, history in self.track_history.items():
            if history['class_name'] != 'person':
                continue
                
            if len(history['positions']) < 10:
                continue
                
            time_in_area = (current_time - history['first_seen']).total_seconds()
            
            if time_in_area > self.loitering_time_threshold:
                positions = list(history['positions'])
                if len(positions) > 5:
                    x_coords = [pos[0] for pos in positions[-20:]]
                    y_coords = [pos[1] for pos in positions[-20:]]
                    
                    x_variance = np.var(x_coords)
                    y_variance = np.var(y_coords)
                    total_variance = x_variance + y_variance
                    
                    if total_variance < 1000:
                        anomalies.append({
                            'type': 'loitering',
                            'severity': 'medium',
                            'description': f'Person loitering for {time_in_area:.1f} seconds',
                            'confidence': min(0.8, time_in_area / (self.loitering_time_threshold * 2)),
                            'track_id': track_id,
                            'metadata': {
                                'duration': time_in_area,
                                'movement_variance': total_variance
                            }
                        })
        
        return anomalies
    
    def _detect_unusual_speed(self, current_time: datetime) -> List[Dict]:
        """Detect unusually fast movement (running, vehicles in pedestrian areas)"""
        anomalies = []
        
        for track_id, history in self.track_history.items():
            if len(history['positions']) < 5 or len(history['timestamps']) < 5:
                continue
            
            # Calculate speed over last few positions
            recent_positions = list(history['positions'])[-5:]
            recent_timestamps = list(history['timestamps'])[-5:]
            
            if len(recent_positions) >= 2:
                # Calculate distance and time
                start_pos = recent_positions[0]
                end_pos = recent_positions[-1]
                start_time = recent_timestamps[0]
                end_time = recent_timestamps[-1]
                
                distance = np.sqrt((end_pos[0] - start_pos[0])**2 + 
                                 (end_pos[1] - start_pos[1])**2)
                time_diff = (end_time - start_time).total_seconds()
                
                if time_diff > 0:
                    speed = distance / time_diff  # pixels per second
                    
                    if speed > self.speed_threshold:
                        anomalies.append({
                            'type': 'unusual_speed',
                            'severity': 'high' if speed > self.speed_threshold * 2 else 'medium',
                            'description': f'Fast movement detected: {speed:.1f} px/s',
                            'confidence': min(0.9, speed / (self.speed_threshold * 2)),
                            'track_id': track_id,
                            'metadata': {
                                'speed': speed,
                                'class_name': history['class_name']
                            }
                        })
        
        return anomalies
    
    def _detect_zone_intrusion(self, tracks: List[Dict]) -> List[Dict]:
        """Detect intrusion into restricted zones"""
        anomalies = []
        
        for zone_name, zone_config in self.zone_definitions.items():
            zone_polygon = zone_config.get('polygon', [])
            zone_type = zone_config.get('type', 'restricted')
            
            if not zone_polygon:
                continue
            
            for track in tracks:
                bbox = track['bbox']
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                
                # Check if center point is inside polygon
                if self._point_in_polygon(center, zone_polygon):
                    anomalies.append({
                        'type': 'zone_intrusion',
                        'severity': 'high',
                        'description': f'Intrusion detected in {zone_name}',
                        'confidence': 0.9,
                        'track_id': track['object_id'],
                        'metadata': {
                            'zone_name': zone_name,
                            'zone_type': zone_type,
                            'object_class': track['class_name']
                        }
                    })
        
        return anomalies
    
    def _analyze_motion_patterns(self, frame: np.ndarray) -> Optional[Dict]:
        """Analyze overall motion patterns in the scene"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Calculate motion intensity
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        self.motion_history.append(motion_ratio)
        
        # Detect sudden motion changes
        if len(self.motion_history) >= 10:
            recent_avg = np.mean(list(self.motion_history)[-5:])
            older_avg = np.mean(list(self.motion_history)[-10:-5])
            
            # Sudden increase in motion
            if recent_avg > older_avg * 3 and recent_avg > 0.1:
                return {
                    'type': 'sudden_motion',
                    'severity': 'medium',
                    'description': 'Sudden increase in scene motion detected',
                    'confidence': min(0.8, recent_avg / 0.2),
                    'metadata': {
                        'motion_ratio': recent_avg,
                        'motion_change': recent_avg / older_avg if older_avg > 0 else float('inf')
                    }
                }
        
        self.prev_frame = gray
        return None
    
    def _detect_abandoned_objects(self, tracks: List[Dict], current_time: datetime) -> List[Dict]:
        # Detect objects that have been stationary for too long
        anomalies = []
        for track_id, history in self.track_history.items():
            if history['class_name'] == 'person':
                continue
            time_stationary = (current_time - history['first_seen']).total_seconds()
            if time_stationary > self.abandoned_object_time:
                if len(history['positions']) > 10:
                    positions = list(history['positions'])
                    x_coords = [pos[0] for pos in positions]
                    y_coords = [pos[1] for pos in positions]
                    movement_range = (max(x_coords) - min(x_coords) + max(y_coords) - min(y_coords))
                    if movement_range < 50:
                        anomalies.append({
                            'type': 'abandoned_object',
                            'severity': 'high',
                            'description': f'Abandoned {history["class_name"]} detected',
                            'confidence': min(0.9, time_stationary / (self.abandoned_object_time * 2)),
                            'track_id': track_id,
                            'metadata': {
                                'object_class': history['class_name'],
                                'stationary_time': time_stationary,
                                'movement_range': movement_range
                            }
                        })
        return anomalies
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_statistics(self) -> Dict:
        """Get current statistics about tracked objects and anomalies"""
        return {
            'active_tracks': len(self.track_history),
            'track_classes': {
                class_name: sum(1 for h in self.track_history.values() 
                              if h['class_name'] == class_name)
                for class_name in set(h['class_name'] for h in self.track_history.values())
            },
            'average_motion': np.mean(self.motion_history) if self.motion_history else 0
        } 