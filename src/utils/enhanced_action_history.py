import time
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

class ActionHistoryManager:
    """
    Advanced action history management for handling fast-disappearing actions.
    Implements multiple strategies to make transient actions more visible and analyzable.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Configuration
        self.max_history_length = config.get('max_action_history', 100)
        self.persistence_duration = config.get('action_persistence_duration', 3.0)  # seconds
        self.min_display_duration = config.get('min_action_display_duration', 1.5)  # seconds
        self.confidence_decay_rate = config.get('confidence_decay_rate', 0.95)
        self.grouping_time_window = config.get('action_grouping_window', 2.0)  # seconds
        self.min_action_confidence = config.get('min_action_confidence', 0.1)
        
        # Storage structures
        self.action_history: deque = deque(maxlen=self.max_history_length)
        self.persistent_actions: Dict[str, Dict] = {}  # Actions being displayed for min duration
        self.action_groups: defaultdict = defaultdict(list)  # Grouped similar actions
        self.track_action_buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # Temporal smoothing
        self.temporal_window = config.get('temporal_smoothing_window', 5)
        self.frame_action_buffer: deque = deque(maxlen=self.temporal_window)
        
        # Statistics
        self.action_frequencies: defaultdict = defaultdict(int)
        self.last_cleanup_time = time.time()
        
    def add_actions(self, actions: List[Dict[str, Any]], timestamp: Optional[float] = None) -> None:
        """Add new actions to the history with automatic timestamping."""
        if timestamp is None:
            timestamp = time.time()
            
        for action in actions:
            # Enhance action with metadata
            enhanced_action = {
                **action,
                'timestamp': timestamp,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'display_until': timestamp + self.min_display_duration,
                'original_confidence': action.get('action_confidence', 0.0),
                'current_confidence': action.get('action_confidence', 0.0),
                'frequency_score': 1,
                'status': 'active'
            }
            
            # Add to main history
            self.action_history.append(enhanced_action)
            
            # Update action frequencies
            action_name = action.get('action_name', 'unknown')
            self.action_frequencies[action_name] += 1
            enhanced_action['frequency_score'] = self.action_frequencies[action_name]
            
            # Add to track-specific buffer if it has a track_id
            track_id = action.get('track_id')
            if track_id is not None and track_id != -1:
                self.track_action_buffers[track_id].append(enhanced_action)
            
            # Add to temporal smoothing buffer
            self.frame_action_buffer.append(enhanced_action)
            
            # Group similar actions
            self._group_similar_action(enhanced_action)
            
    def _group_similar_action(self, action: Dict[str, Any]) -> None:
        """Group similar actions occurring within time window."""
        action_name = action.get('action_name', 'unknown')
        track_id = action.get('track_id', -1)
        group_key = f"{action_name}_{track_id}"
        
        # Clean old actions from group
        current_time = action['timestamp']
        cutoff_time = current_time - self.grouping_time_window
        
        self.action_groups[group_key] = [
            a for a in self.action_groups[group_key] 
            if a['timestamp'] >= cutoff_time
        ]
        
        # Add current action to group
        self.action_groups[group_key].append(action)
        
    def get_persistent_actions(self, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get actions that should be displayed persistently."""
        if current_time is None:
            current_time = time.time()
            
        persistent = []
        
        # Process actions that should be displayed for minimum duration
        for action_id, action_data in list(self.persistent_actions.items()):
            if current_time <= action_data['display_until']:
                # Decay confidence over time for visual indication
                time_since_last_seen = current_time - action_data['last_seen']
                decay_factor = self.confidence_decay_rate ** time_since_last_seen
                action_data['current_confidence'] = action_data['original_confidence'] * decay_factor
                action_data['status'] = 'persisting'
                persistent.append(action_data.copy())
            else:
                action_data['status'] = 'expired'
                del self.persistent_actions[action_id]
                
        # Add new actions from recent history
        for action in list(self.action_history)[-10:]:  # Check last 10 actions
            if action['timestamp'] >= current_time - self.persistence_duration:
                action_id = f"{action.get('action_name', 'unknown')}_{action.get('track_id', -1)}"
                
                if action_id not in self.persistent_actions:
                    self.persistent_actions[action_id] = action.copy()
                    persistent.append(action.copy())
                else:
                    # Update existing persistent action
                    existing = self.persistent_actions[action_id]
                    existing['last_seen'] = action['timestamp']
                    existing['current_confidence'] = max(
                        existing['current_confidence'], 
                        action['action_confidence']
                    )
                    existing['display_until'] = max(
                        existing['display_until'],
                        action['timestamp'] + self.min_display_duration
                    )
                    
        return persistent
        
    def get_temporal_smoothed_actions(self, window_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get temporally smoothed actions using majority voting."""
        if window_size is None:
            window_size = self.temporal_window
            
        if len(self.frame_action_buffer) < 2:
            return list(self.frame_action_buffer)
            
        # Group actions by name and track_id
        action_groups = defaultdict(list)
        for action in list(self.frame_action_buffer)[-window_size:]:
            key = f"{action.get('action_name', 'unknown')}_{action.get('track_id', -1)}"
            action_groups[key].append(action)
            
        smoothed_actions = []
        for group_key, actions in action_groups.items():
            if len(actions) >= max(1, window_size // 3):  # Require at least 1/3 consensus
                # Create smoothed action from group
                latest_action = max(actions, key=lambda x: x['timestamp'])
                avg_confidence = np.mean([a['action_confidence'] for a in actions])
                
                smoothed_action = latest_action.copy()
                smoothed_action['action_confidence'] = avg_confidence
                smoothed_action['smoothed'] = True
                smoothed_action['consensus_count'] = len(actions)
                smoothed_action['stability_score'] = len(actions) / window_size
                
                smoothed_actions.append(smoothed_action)
                
        return smoothed_actions
        
    def get_grouped_actions(self, min_group_size: int = 2) -> List[Dict[str, Any]]:
        """Get actions that have been grouped by similarity."""
        current_time = time.time()
        grouped_actions = []
        
        for group_key, actions in self.action_groups.items():
            if len(actions) >= min_group_size:
                # Create summary action for the group
                latest_action = max(actions, key=lambda x: x['timestamp'])
                avg_confidence = np.mean([a['action_confidence'] for a in actions])
                duration = current_time - min(a['timestamp'] for a in actions)
                
                group_summary = latest_action.copy()
                group_summary.update({
                    'action_confidence': avg_confidence,
                    'group_size': len(actions),
                    'group_duration': duration,
                    'group_frequency': len(actions) / max(0.1, duration),
                    'is_grouped': True,
                    'first_occurrence': min(a['timestamp'] for a in actions),
                    'last_occurrence': max(a['timestamp'] for a in actions)
                })
                
                grouped_actions.append(group_summary)
                
        return grouped_actions
        
    def get_high_frequency_actions(self, min_frequency: int = 3) -> List[Dict[str, Any]]:
        """Get actions that occur frequently."""
        frequent_actions = []
        
        for action_name, frequency in self.action_frequencies.items():
            if frequency >= min_frequency:
                # Find latest occurrence of this action
                latest_action = None
                for action in reversed(self.action_history):
                    if action.get('action_name') == action_name:
                        latest_action = action.copy()
                        break
                        
                if latest_action:
                    latest_action.update({
                        'is_frequent': True,
                        'total_frequency': frequency,
                        'frequency_rank': frequency
                    })
                    frequent_actions.append(latest_action)
                    
        # Sort by frequency
        frequent_actions.sort(key=lambda x: x['total_frequency'], reverse=True)
        return frequent_actions
        
    def get_recent_actions(self, time_window: float = 10.0) -> List[Dict[str, Any]]:
        """Get all actions within the specified time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_actions = [
            action for action in self.action_history
            if action['timestamp'] >= cutoff_time
        ]
        
        return sorted(recent_actions, key=lambda x: x['timestamp'], reverse=True)
        
    def get_track_action_summary(self, track_id: int) -> Dict[str, Any]:
        """Get action summary for a specific track."""
        if track_id not in self.track_action_buffers:
            return {}
            
        actions = list(self.track_action_buffers[track_id])
        if not actions:
            return {}
            
        action_counts = defaultdict(int)
        total_confidence = 0
        
        for action in actions:
            action_name = action.get('action_name', 'unknown')
            action_counts[action_name] += 1
            total_confidence += action.get('action_confidence', 0)
            
        most_common_action = max(action_counts.items(), key=lambda x: x[1])
        
        return {
            'track_id': track_id,
            'most_common_action': most_common_action[0],
            'action_frequency': most_common_action[1],
            'total_actions': len(actions),
            'avg_confidence': total_confidence / len(actions) if actions else 0,
            'action_diversity': len(action_counts),
            'latest_action': actions[-1] if actions else None,
            'action_timeline': actions
        }
        
    def cleanup_old_data(self, max_age: float = 300.0) -> None:
        """Clean up old data to prevent memory leaks."""
        current_time = time.time()
        
        # Only cleanup every 30 seconds
        if current_time - self.last_cleanup_time < 30:
            return
            
        cutoff_time = current_time - max_age
        
        # Clean action groups
        for group_key in list(self.action_groups.keys()):
            self.action_groups[group_key] = [
                a for a in self.action_groups[group_key]
                if a['timestamp'] >= cutoff_time
            ]
            if not self.action_groups[group_key]:
                del self.action_groups[group_key]
                
        # Clean track buffers (keep more recent data)
        track_cutoff = current_time - 60.0  # Keep 1 minute of track data
        for track_id in list(self.track_action_buffers.keys()):
            buffer = self.track_action_buffers[track_id]
            while buffer and buffer[0]['timestamp'] < track_cutoff:
                buffer.popleft()
            if not buffer:
                del self.track_action_buffers[track_id]
                
        self.last_cleanup_time = current_time
        
    def get_display_actions(self, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """
        Get actions for display using the specified strategy.
        
        Strategies:
        - 'persistent': Show actions for minimum duration
        - 'smoothed': Use temporal smoothing
        - 'grouped': Show grouped similar actions
        - 'frequent': Show high-frequency actions
        - 'mixed': Combine multiple strategies (recommended)
        """
        current_time = time.time()
        self.cleanup_old_data()
        
        if strategy == 'persistent':
            return self.get_persistent_actions(current_time)
        elif strategy == 'smoothed':
            return self.get_temporal_smoothed_actions()
        elif strategy == 'grouped':
            return self.get_grouped_actions()
        elif strategy == 'frequent':
            return self.get_high_frequency_actions()
        elif strategy == 'mixed':
            # Combine strategies for best results
            display_actions = []
            
            # Start with persistent actions (most important)
            persistent = self.get_persistent_actions(current_time)
            display_actions.extend(persistent)
            
            # Add smoothed actions if we don't have many persistent ones
            if len(persistent) < 3:
                smoothed = self.get_temporal_smoothed_actions()
                for action in smoothed:
                    # Avoid duplicates
                    if not any(
                        a.get('action_name') == action.get('action_name') and
                        a.get('track_id') == action.get('track_id')
                        for a in display_actions
                    ):
                        display_actions.append(action)
                        
            # Add frequent actions as lower priority
            if len(display_actions) < 5:
                frequent = self.get_high_frequency_actions()[:2]  # Limit to top 2
                for action in frequent:
                    if not any(
                        a.get('action_name') == action.get('action_name')
                        for a in display_actions
                    ):
                        display_actions.append(action)
                        
            return display_actions[:8]  # Limit total display actions
        else:
            return self.get_recent_actions(5.0)  # Fallback
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about action history."""
        current_time = time.time()
        
        return {
            'total_actions_recorded': len(self.action_history),
            'persistent_actions_count': len(self.persistent_actions),
            'active_tracks': len(self.track_action_buffers),
            'action_groups_count': len(self.action_groups),
            'unique_actions_seen': len(self.action_frequencies),
            'most_frequent_action': max(self.action_frequencies.items(), key=lambda x: x[1])[0] if self.action_frequencies else None,
            'actions_last_minute': len(self.get_recent_actions(60.0)),
            'actions_last_10_seconds': len(self.get_recent_actions(10.0)),
            'temporal_buffer_size': len(self.frame_action_buffer)
        } 