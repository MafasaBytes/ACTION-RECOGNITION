import time
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
import threading
import queue

from src.utils.logger import setup_logger

try:
    from src.utils.audio_alerts import play_warning_sound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

logger = setup_logger(__name__)

class EnhancedWarningSystem:  
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        
        # Warning levels configuration
        self.warning_levels = {
            1: {
                'name': 'MEDIUM',
                'color': (0, 165, 255),      # Orange
                'icon': '[*]',
                'priority': 1,
                'audio_enabled': True,
                'flash_enabled': False,
                'escalation_time': 15.0      # Escalate to level 2 after 15 seconds
            },
            2: {
                'name': 'HIGH', 
                'color': (0, 100, 255),      # Red-Orange
                'icon': '[!]',
                'priority': 2,
                'audio_enabled': True,
                'flash_enabled': True,
                'escalation_time': 10.0      # Escalate to level 3 after 10 seconds
            },
            3: {
                'name': 'CRITICAL',
                'color': (0, 0, 255),        # Red
                'icon': '[!!!]',
                'priority': 3,
                'audio_enabled': True,
                'flash_enabled': True,
                'escalation_time': None      # No further escalation
            }
        }
        
        # Warning tracking
        self.active_warnings = {}  # track_id -> warning_info
        self.warning_history = defaultdict(list)  # track_id -> list of warnings
        self.last_audio_alert = {}  # track_id -> timestamp
        self.audio_cooldown = 5.0  # Minimum seconds between audio alerts for same track
        
        # Visual effects
        self.flash_start_time = {}  # track_id -> start_time for flashing
        self.flash_duration = 3.0  # Duration of flash effect in seconds
        
        # Escalation tracking
        self.escalation_timers = {}  # track_id -> {'start_time': time, 'current_level': level}
        
        # Audio queue for threaded audio playback
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self._start_audio_thread()
        
        logger.info("Enhanced Warning System initialized")
    
    def _start_audio_thread(self):
        if not AUDIO_AVAILABLE:
            return
            
        def audio_worker():
            while True:
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                    if audio_data is None:  # Shutdown signal
                        break
                    
                    level, action_name, track_id = audio_data
                    play_warning_sound(level, f"{action_name} (ID:{track_id})")
                    self.audio_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Audio alert error: {e}")
        
        self.audio_thread = threading.Thread(target=audio_worker, daemon=True)
        self.audio_thread.start()
    
    def process_warnings(self, anomaly_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        current_time = time.time()
        enhanced_warnings = []
        
        # Track which tracks have active warnings this frame
        current_frame_tracks = set()
        
        for score in anomaly_scores:
            track_id = score.get('track_id', 'Global')
            warning_level = score.get('warning_level', 0)
            
            if warning_level > 0:
                current_frame_tracks.add(track_id)
                enhanced_warning = self._process_single_warning(score, current_time)
                enhanced_warnings.append(enhanced_warning)
        
        # Clean up warnings for tracks no longer active
        self._cleanup_inactive_warnings(current_frame_tracks, current_time)
        
        # Sort warnings by priority (highest first)
        enhanced_warnings.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return enhanced_warnings
    
    def _process_single_warning(self, score: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        track_id = score.get('track_id', 'Global')
        warning_level = score.get('warning_level', 0)
        action_name = score.get('action_name', 'Unknown')
        description = score.get('description', 'Unknown Warning')
        
        # Check for escalation
        escalated_level = self._check_escalation(track_id, warning_level, current_time)
        
        # Update active warning
        warning_info = {
            'track_id': track_id,
            'original_level': warning_level,
            'current_level': escalated_level,
            'action_name': action_name,
            'description': description,
            'start_time': self.active_warnings.get(track_id, {}).get('start_time', current_time),
            'last_update': current_time,
            'escalation_count': self.active_warnings.get(track_id, {}).get('escalation_count', 0)
        }
        
        # Check if this is a new escalation
        if track_id in self.active_warnings:
            if escalated_level > self.active_warnings[track_id]['current_level']:
                warning_info['escalation_count'] += 1
                logger.warning(f"Warning escalated for Track {track_id}: Level {escalated_level}")
        
        self.active_warnings[track_id] = warning_info
        
        # Trigger audio alert if needed
        self._trigger_audio_alert(track_id, escalated_level, action_name, current_time)
        
        # Trigger flash effect for high priority warnings
        if escalated_level >= 2 and track_id not in self.flash_start_time:
            self.flash_start_time[track_id] = current_time
        
        # Create enhanced warning data
        level_config = self.warning_levels[escalated_level]
        enhanced_warning = {
            **score,  # Keep original data
            'warning_level': escalated_level,
            'severity_name': level_config['name'],
            'severity_color': level_config['color'],
            'severity_icon': level_config['icon'],
            'priority': level_config['priority'],
            'duration': current_time - warning_info['start_time'],
            'escalation_count': warning_info['escalation_count'],
            'is_escalated': escalated_level > warning_level,
            'flash_active': self._is_flash_active(track_id, current_time)
        }
        
        return enhanced_warning
    
    def _check_escalation(self, track_id: str, current_level: int, current_time: float) -> int:
        if track_id not in self.escalation_timers:
            self.escalation_timers[track_id] = {
                'start_time': current_time,
                'current_level': current_level
            }
            return current_level
        
        timer_info = self.escalation_timers[track_id]
        duration = current_time - timer_info['start_time']
        
        # Check for escalation based on duration
        for level in range(current_level, 4):  # Check levels up to 3
            if level in self.warning_levels:
                escalation_time = self.warning_levels[level].get('escalation_time')
                if escalation_time and duration >= escalation_time:
                    next_level = min(level + 1, 3)  # Cap at level 3
                    if next_level > timer_info['current_level']:
                        timer_info['current_level'] = next_level
                        logger.info(f"Warning escalated for Track {track_id}: {level} -> {next_level} after {duration:.1f}s")
                        return next_level
        
        return max(current_level, timer_info['current_level'])
    
    def _trigger_audio_alert(self, track_id: str, level: int, action_name: str, current_time: float):
        if not AUDIO_AVAILABLE:
            return
        
        level_config = self.warning_levels.get(level, {})
        if not level_config.get('audio_enabled', False):
            return
        
        # Check cooldown
        last_alert = self.last_audio_alert.get(track_id, 0)
        if current_time - last_alert < self.audio_cooldown:
            return
        
        # Queue audio alert
        try:
            self.audio_queue.put_nowait((level, action_name, track_id))
            self.last_audio_alert[track_id] = current_time
            logger.debug(f"Audio alert queued for Track {track_id}, Level {level}")
        except queue.Full:
            logger.warning("Audio queue full, skipping alert")
    
    def _is_flash_active(self, track_id: str, current_time: float) -> bool:
        if track_id not in self.flash_start_time:
            return False
        
        flash_elapsed = current_time - self.flash_start_time[track_id]
        if flash_elapsed > self.flash_duration:
            del self.flash_start_time[track_id]
            return False
        
        return True
    
    def _cleanup_inactive_warnings(self, active_tracks: set, current_time: float):
        inactive_tracks = set(self.active_warnings.keys()) - active_tracks
        
        for track_id in inactive_tracks:
            warning_info = self.active_warnings[track_id]
            
            # Move to history
            self.warning_history[track_id].append({
                **warning_info,
                'end_time': current_time,
                'total_duration': current_time - warning_info['start_time']
            })
            
            # Clean up tracking data
            del self.active_warnings[track_id]
            if track_id in self.escalation_timers:
                del self.escalation_timers[track_id]
            if track_id in self.flash_start_time:
                del self.flash_start_time[track_id]
            
            logger.info(f"Warning cleared for Track {track_id}")
    
    def get_warning_statistics(self) -> Dict[str, Any]:
        current_time = time.time()
        
        # Current warnings stats
        active_count = len(self.active_warnings)
        level_counts = defaultdict(int)
        total_duration = 0
        
        for warning in self.active_warnings.values():
            level_counts[warning['current_level']] += 1
            total_duration += current_time - warning['start_time']
        
        # Historical stats
        total_historical = sum(len(history) for history in self.warning_history.values())
        
        return {
            'active_warnings': active_count,
            'level_1_count': level_counts[1],
            'level_2_count': level_counts[2], 
            'level_3_count': level_counts[3],
            'total_active_duration': total_duration,
            'historical_warnings': total_historical,
            'tracks_with_warnings': len(self.warning_history)
        }
    
    def should_trigger_global_alert(self) -> bool:
        # Trigger global alert if any critical warnings or multiple high warnings
        critical_count = sum(1 for w in self.active_warnings.values() if w['current_level'] >= 3)
        high_count = sum(1 for w in self.active_warnings.values() if w['current_level'] >= 2)
        
        return critical_count > 0 or high_count >= 2
    
    def get_flash_intensity(self, current_time: float) -> float:
        if not any(self._is_flash_active(tid, current_time) for tid in self.flash_start_time):
            return 0.0
        
        # Pulsing flash effect
        return abs(np.sin(current_time * 4)) * 0.3 + 0.1
    
    def shutdown(self):
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_queue.put(None)  # Shutdown signal
            self.audio_thread.join(timeout=2.0)
        
        logger.info("Enhanced Warning System shutdown")

# helper function integration
def enhance_anomaly_scores(anomaly_scores: List[Dict[str, Any]], warning_system: EnhancedWarningSystem) -> List[Dict[str, Any]]:
    return warning_system.process_warnings(anomaly_scores) 