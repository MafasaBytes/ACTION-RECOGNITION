import cv2
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from .base_panel import BasePanel

class ActionHistoryPanel(BasePanel):
    """
    Enhanced panel for displaying action history with persistence, frequency, and temporal data.
    Shows not just current actions but also recent and frequent actions for better visibility.
    """
    
    def __init__(self, panel_id: str, color_scheme, panel_height: int = 250, max_items: int = 8):
        super().__init__(panel_id, color_scheme, panel_height)
        self.max_items = max_items
        self.header_height = 25
        self.item_height = 22
        self.indent = 8
        self.status_indicators = {
            'active': '●',      # Current action
            'persisting': '◐',  # Persisting action
            'frequent': '★',    # Frequent action
            'grouped': '▲',     # Grouped action
            'smoothed': '◆'     # Temporally smoothed
        }
        
    def render(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Render the action history panel.
        
        Expected data format:
        {
            'width': int,
            'action_history': List[Dict] - from ActionHistoryManager.get_display_actions(),
            'statistics': Dict - from ActionHistoryManager.get_statistics(),
            'strategy': str - display strategy being used
        }
        """
        width = data.get('width', 300)
        action_history = data.get('action_history', [])
        statistics = data.get('statistics', {})
        strategy = data.get('strategy', 'mixed')
        
        # Calculate required height based on content
        required_height = self._calculate_required_height(action_history, statistics)
        self.panel_height = min(required_height, 350)  # Cap at 350px
        
        panel = np.full((self.panel_height, width, 3), 
                       self.color_scheme.background_colors['panel_normal'], dtype=np.uint8)
        
        y_offset = 5
        
        # Header with strategy info
        header_text = f"ACTION HISTORY ({strategy.upper()})"
        cv2.putText(panel, header_text, (self.indent, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                   self.color_scheme.text_colors['header'], 1)
        
        # Statistics summary
        total_actions = statistics.get('total_actions_recorded', 0)
        persistent_count = statistics.get('persistent_actions_count', 0)
        stats_text = f"Total: {total_actions} | Persistent: {persistent_count}"
        cv2.putText(panel, stats_text, (self.indent, y_offset + 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, 
                   self.color_scheme.text_colors['info'], 1)
        
        y_offset += 45
        
        # Draw separator
        cv2.line(panel, (5, y_offset), (width - 5, y_offset), 
                self.color_scheme.ui_colors['divider'], 1)
        y_offset += 8
        
        # Display actions
        if not action_history:
            # No actions message
            cv2.putText(panel, "No recent actions detected", 
                       (self.indent, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       self.color_scheme.text_colors['info'], 1)
        else:
            # Show up to max_items actions
            displayed_count = 0
            for action in action_history[:self.max_items]:
                if y_offset + self.item_height > self.panel_height - 10:
                    break
                    
                y_offset = self._draw_action_item(panel, action, y_offset, width)
                displayed_count += 1
                
            # Show truncation message if needed
            if len(action_history) > displayed_count:
                remaining = len(action_history) - displayed_count
                cv2.putText(panel, f"... and {remaining} more", 
                           (self.indent, y_offset + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, 
                           self.color_scheme.text_colors['info'], 1)
        
        return panel
    
    def _calculate_required_height(self, action_history: List[Dict], statistics: Dict) -> int:
        """Calculate the required height for the panel based on content."""
        base_height = 60  # Header + stats + separator
        
        if not action_history:
            return base_height + 30  # Space for "no actions" message
        
        items_to_show = min(len(action_history), self.max_items)
        action_items_height = items_to_show * (self.item_height + 3)
        
        truncation_height = 20 if len(action_history) > self.max_items else 0
        
        return base_height + action_items_height + truncation_height + 10
    
    def _draw_action_item(self, panel: np.ndarray, action: Dict[str, Any], 
                         y_offset: int, width: int) -> int:
        """Draw a single action item with all its metadata."""
        current_time = time.time()
        
        # Extract action information
        action_name = action.get('action_name', 'Unknown Action')
        track_id = action.get('track_id', -1)
        confidence = action.get('current_confidence', action.get('action_confidence', 0.0))
        timestamp = action.get('timestamp', current_time)
        status = action.get('status', 'active')
        
        # Implement consistent color coordination: Green → Good, Orange → In-between, Red → Bad
        if confidence > 0.7:  # Red → Bad (High confidence threat)
            text_color = (100, 100, 255)  # Red
            conf_color = (100, 100, 255)  # Red
        elif confidence > 0.4:  # Orange → In-between (Medium confidence)
            text_color = (100, 165, 255)  # Orange  
            conf_color = (100, 165, 255)  # Orange
        else:  # Green → Good (Low confidence/normal)
            text_color = (100, 255, 100)  # Green
            conf_color = (100, 255, 100)  # Green
        
        # Determine status indicator based on action metadata
        if action.get('is_frequent'):
            status_indicator = self.status_indicators['frequent']
        elif action.get('is_grouped'):
            status_indicator = self.status_indicators['grouped']
        elif action.get('smoothed'):
            status_indicator = self.status_indicators['smoothed']
        elif status == 'persisting':
            status_indicator = self.status_indicators['persisting']
        else:
            status_indicator = self.status_indicators['active']
        
        # Draw status indicator
        cv2.putText(panel, status_indicator, (self.indent, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Truncate action name if too long
        max_name_length = 25
        display_name = action_name[:max_name_length] + "..." if len(action_name) > max_name_length else action_name
        
        # Draw action name (without ID)
        cv2.putText(panel, display_name, (self.indent + 20, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)
        
        # Draw confidence and additional info
        info_text = f"{confidence:.2f}"
        
        # Add frequency info if available
        if action.get('total_frequency'):
            info_text += f" ({action['total_frequency']}x)"
        elif action.get('group_size'):
            info_text += f" (g:{action['group_size']})"
        elif action.get('consensus_count'):
            info_text += f" (s:{action['consensus_count']})"
        
        cv2.putText(panel, info_text, (width - 80, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, conf_color, 1)
        
        # Draw time elapsed (for persisting actions) - removed track ID display
        if status == 'persisting':
            time_elapsed = current_time - timestamp
            if time_elapsed < 60:
                time_text = f"{time_elapsed:.1f}s"
            else:
                time_text = f"{time_elapsed/60:.1f}m"
            
            cv2.putText(panel, time_text, (self.indent + 20, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                       self.color_scheme.text_colors['info'], 1)
            
            # Draw decay bar for persisting actions
            bar_width = 60
            bar_height = 3
            bar_x = self.indent + 80
            bar_y = y_offset + 27
            
            # Calculate decay progress
            display_until = action.get('display_until', timestamp + 2.0)
            total_duration = display_until - timestamp
            remaining_duration = max(0, display_until - current_time)
            progress = remaining_duration / total_duration if total_duration > 0 else 0
            
            # Draw background bar
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         self.color_scheme.ui_colors['border_normal'], -1)
            
            # Draw progress bar with consistent color coordination
            if progress > 0:
                progress_width = int(bar_width * progress)
                # Use the same color coordination for the progress bar
                if confidence > 0.7:
                    bar_color = (100, 100, 255)  # Red
                elif confidence > 0.4:
                    bar_color = (100, 165, 255)  # Orange
                else:
                    bar_color = (100, 255, 100)  # Green
                cv2.rectangle(panel, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                             bar_color, -1)
        
        return y_offset + self.item_height + 3
    
    def get_required_height(self, data: Dict[str, Any]) -> int:
        """Calculate the required height for this panel."""
        action_history = data.get('action_history', [])
        statistics = data.get('statistics', {})
        return self._calculate_required_height(action_history, statistics) 