import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime
import psutil # Added psutil import

from .base_panel import BasePanel # Assuming base_panel.py is in the same directory

class SystemStatusPanel(BasePanel):
    """Displays general system status like FPS, object count, and current time."""
    def __init__(self, panel_id: str, color_scheme, panel_height: int = 195, **kwargs):
        super().__init__(panel_id, color_scheme, **kwargs)
        self.panel_height = panel_height

    def get_required_height(self, data: Any = None) -> int:
        return self.panel_height

    def _draw_progress_bar(self, surface: np.ndarray, y_pos: int, label: str, value: float, 
                             bar_width: int, bar_height: int, label_color: Tuple[int,int,int], value_color: Tuple[int,int,int], 
                             bar_color: Tuple[int,int,int], bar_bg_color: Tuple[int,int,int]):
        """Helper to draw a label, a progress bar, and the value text."""
        
        # Draw label
        cv2.putText(surface, label, (self.panel_padding, y_pos), self.font, 
                   self.font_scale, label_color, self.font_thickness, cv2.LINE_AA)
        
        # Calculate bar position
        bar_x = self.panel_padding + 80  # Adjusted to give more space for labels
        bar_y = y_pos - 12
        
        # Draw background bar
        cv2.rectangle(surface, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     bar_bg_color, -1)
        
        # Calculate fill width based on percentage
        fill_width = int((value / 100.0) * bar_width)
        
        # Determine bar color based on value using consistent color coordination
        # Green → Good (low usage), Orange → In-between (medium usage), Red → Bad (high usage)
        if value < 60:  # Green → Good performance
            actual_bar_color = (100, 255, 100)  # Green
        elif value < 80:  # Orange → In-between performance
            actual_bar_color = (100, 165, 255)  # Orange
        else:  # Red → Bad performance (high usage)
            actual_bar_color = (100, 100, 255)  # Red
        
        # Draw filled portion
        if fill_width > 0:
            cv2.rectangle(surface, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         actual_bar_color, -1)
        
        # Draw border
        cv2.rectangle(surface, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     label_color, 1)
        
        # Draw value text
        value_text = f"{value:.1f}%"
        cv2.putText(surface, value_text, (bar_x + bar_width + 10, y_pos), self.font, 
                   self.font_scale, value_color, self.font_thickness, cv2.LINE_AA)

    def render(self, data: Dict[str, Any]) -> np.ndarray:
        surface = np.zeros((self.panel_height, data['width'], 3), dtype=np.uint8)
        self._draw_panel_background(surface)
        
        y_pos = self.panel_padding + self.line_height
        
        # Header
        header_text = "SYSTEM PERFORMANCE" # Changed Header
        if self.color_scheme:
            header_color = self.color_scheme.text_colors['header_accent']
        else: # Fallback
            header_color = (200, 200, 150)
        cv2.putText(surface, header_text, (self.panel_padding, y_pos), self.font, 
                    self.header_font_scale, header_color, self.header_thickness, cv2.LINE_AA)
        y_pos += self.line_height + self.panel_padding // 2
        
        # Line separator
        if self.color_scheme: line_color = self.color_scheme.ui_colors['divider']
        else: line_color = (80,80,80)
        cv2.line(surface, (self.panel_padding, y_pos), (data['width'] - self.panel_padding, y_pos), line_color, 1)
        y_pos += self.line_height
        
        # Content
        # cpu_usage = data.get('cpu_usage', psutil.cpu_percent(interval=0.05)) # get live data below
        # ram_usage = data.get('ram_usage', psutil.virtual_memory().percent) # get live data below
        
        status_items_template = [
            # f"Time: {{time_now}}", # Removed Time from here
            f"FPS: {{avg_fps}}",
            f"CPU: {{cpu_usage}}%",
            f"RAM: {{ram_usage}}%",
            # f"Objects: {{object_count}}", # Objects count already removed
            f"Status: {{system_status}}"
        ]
        
        if self.color_scheme:
            default_text_color = self.color_scheme.text_colors['normal']
            value_color = self.color_scheme.text_colors['info']
            progress_bar_color = self.color_scheme.ui_colors.get('progress_bar', (100, 150, 200))
            progress_bar_bg_color = self.color_scheme.ui_colors.get('progress_bar_bg', (70, 70, 70))
            # Implement consistent color coordination: Green → Good, Orange → In-between, Red → Bad
            status_color_map = {
                "NORMAL": (100, 255, 100),      # Green → Good
                "WARNING": (100, 165, 255),     # Orange → In-between  
                "CRITICAL": (100, 100, 255),    # Red → Bad
                "SLOW": (100, 165, 255),        # Orange → In-between
                "INIT": (150,150,150)           # Gray → Neutral
            }
        else: # Fallback
            default_text_color = (180,180,180)
            value_color = (150,180,150)
            progress_bar_color = (100, 150, 200)
            progress_bar_bg_color = (70, 70, 70)
            status_color_map = { 
                "NORMAL": (100, 255, 100),      # Green → Good
                "WARNING": (100, 165, 255),     # Orange → In-between
                "CRITICAL": (100, 100, 255),    # Red → Bad
                "SLOW": (100, 165, 255),        # Orange → In-between
                "INIT": (150,150,150)           # Gray → Neutral
            }

        item_y_pos = y_pos
        label_x_offset = self.panel_padding
        value_x_offset = self.panel_padding + 100 # Original offset for values

        # Get live psutil data once
        live_cpu_usage = psutil.cpu_percent(interval=0.05) # Added interval for stability
        live_ram_usage = psutil.virtual_memory().percent

        # Update status_items to reflect live data for text display if needed, though progress bar uses live vars directly
        # This step is more for consistency if we were to display text for CPU/RAM instead of or alongside bars.
        status_items_updated = []
        for item_format_str in status_items_template:
            formatted_item = item_format_str.format(
                time_now=datetime.now().strftime('%H:%M:%S'), # Still needed for .format if string existed
                avg_fps=data.get('avg_fps', 'N/A'),
                cpu_usage=f"{live_cpu_usage:.1f}", 
                ram_usage=f"{live_ram_usage:.1f}", 
                object_count=data.get('object_count', 'N/A'), 
                system_status=data.get('system_status', 'NORMAL')
            )
            status_items_updated.append(formatted_item)
        status_items = status_items_updated

        for item_text in status_items:
            label, value_str = item_text.split(':', 1) if ':' in item_text else (item_text, "")
            value_str = value_str.strip()
            current_label = label.strip()

            if current_label == "CPU" or current_label == "RAM":
                bar_render_width = data['width'] - (self.panel_padding * 3) - 60 # Available width for bar and text after bar
                
                progress_value = 0.0
                if current_label == "CPU":
                    progress_value = live_cpu_usage
                elif current_label == "RAM":
                    progress_value = live_ram_usage
                
                self._draw_progress_bar(surface, item_y_pos, current_label + ":", progress_value, 
                                        bar_width=bar_render_width - 60, bar_height=self.line_height - 5, # Adjusted bar height
                                        label_color=default_text_color, value_color=value_color, 
                                        bar_color=progress_bar_color, bar_bg_color=progress_bar_bg_color)
                item_y_pos += self.line_height + self.panel_padding # Increased padding here
            else:
                current_value_color = value_color
                if current_label == "Status":
                    current_value_color = status_color_map.get(value_str.upper(), value_color)
                elif current_label == "Time": # Example of different color for time if desired
                    current_value_color = self.color_scheme.text_colors.get('highlight', value_color) if self.color_scheme else (200,200,200)

                cv2.putText(surface, label + ":", (label_x_offset, item_y_pos), self.font, 
                            self.font_scale, default_text_color, self.font_thickness, cv2.LINE_AA)
                cv2.putText(surface, value_str, (value_x_offset, item_y_pos), self.font, 
                            self.font_scale, current_value_color, self.font_thickness, cv2.LINE_AA)
                item_y_pos += self.line_height
            
            if item_y_pos > self.panel_height - self.panel_padding: # Avoid overflow
                break
        return surface

class AnomalyListPanel(BasePanel):
    """Displays a list of recent or critical anomalies/actions."""
    def __init__(self, panel_id: str, color_scheme, panel_height: int = 250, max_items: int = 5, **kwargs):
        super().__init__(panel_id, color_scheme, **kwargs)
        self.panel_height = panel_height
        self.max_items = max_items

    def get_required_height(self, data: Any = None) -> int:
        return self.panel_height

    def render(self, data: Dict[str, Any]) -> np.ndarray:
        surface = np.zeros((self.panel_height, data['width'], 3), dtype=np.uint8)
        anomalies = data.get('anomalies', []) # Expected to be a list of dicts
        panel_type = 'panel_danger' if any(a.get('is_critical', False) for a in anomalies) else 'panel_warning' if anomalies else 'panel_normal'
        self._draw_panel_background(surface, panel_type=panel_type)
        
        y_pos = self.panel_padding + self.line_height
        
        # Header
        header_text = "RECENT ALERTS"
        if self.color_scheme:
            header_color = self.color_scheme.text_colors['header_accent'] if panel_type == 'panel_normal' else self.color_scheme.text_colors['critical']
        else: # Fallback
            header_color = (200,150,150) if panel_type != 'panel_normal' else (200,200,150)
        cv2.putText(surface, header_text, (self.panel_padding, y_pos), self.font, 
                    self.header_font_scale, header_color, self.header_thickness, cv2.LINE_AA)
        y_pos += self.line_height + self.panel_padding // 2
        
        # Line separator
        if self.color_scheme: line_color = self.color_scheme.ui_colors['divider']
        else: line_color = (80,80,80)
        cv2.line(surface, (self.panel_padding, y_pos), (data['width'] - self.panel_padding, y_pos), line_color, 1)
        y_pos += self.line_height

        if not anomalies:
            no_alerts_text = "No active alerts."
            if self.color_scheme: text_color = self.color_scheme.text_colors['info']
            else: text_color = (150,180,150)
            cv2.putText(surface, no_alerts_text, (self.panel_padding, y_pos), self.font, 
                        self.font_scale, text_color, self.font_thickness, cv2.LINE_AA)
            return surface

        # Display anomalies (simplified for now)
        for i, anomaly in enumerate(anomalies[:self.max_items]):
            action_name = anomaly.get('action_type', 'Unknown Action')
            score = anomaly.get('score', 0.0)
            track_id = anomaly.get('track_id', 'N/A')
            
            # Remove ID display and just show action name and score
            text = f"{action_name} ({score:.2f})"
            truncated_text = self._truncate_text(text, data['width'] - (2 * self.panel_padding))

            # Implement consistent color coordination: Green → Good, Orange → In-between, Red → Bad
            if self.color_scheme:
                if score > 0.7:  # Red → Bad (High threat)
                    item_color = (100, 100, 255)  # Red
                elif score > 0.4:  # Orange → In-between (Medium threat)
                    item_color = (100, 165, 255)  # Orange
                else:  # Green → Good (Low/No threat)
                    item_color = (100, 255, 100)  # Green
            else: # Fallback
                item_color = (220,100,100) if score > 0.7 else (255,165,0) if score > 0.4 else (100,255,100)

            cv2.putText(surface, truncated_text, (self.panel_padding, y_pos), self.font, 
                        self.font_scale, item_color, self.font_thickness, cv2.LINE_AA)
            y_pos += self.line_height
            if y_pos > self.panel_height - self.panel_padding: # Avoid overflow
                break
        
        return surface

class GlobalDetectionsPanel(BasePanel):
    """Displays a summary of global object detections (class counts)."""
    def __init__(self, panel_id: str, color_scheme, panel_height: int = 180, max_classes_to_show: int = 5, **kwargs):
        super().__init__(panel_id, color_scheme, **kwargs)
        self.panel_height = panel_height
        self.max_classes_to_show = max_classes_to_show

    def get_required_height(self, data: Any = None) -> int:
        return self.panel_height

    def render(self, data: Dict[str, Any]) -> np.ndarray:
        surface = np.zeros((self.panel_height, data['width'], 3), dtype=np.uint8)
        self._draw_panel_background(surface)
        
        y_pos = self.panel_padding + self.line_height
        
        header_text = "GLOBAL DETECTIONS"
        if self.color_scheme: header_color = self.color_scheme.text_colors['header_accent']
        else: header_color = (200, 200, 150)
        cv2.putText(surface, header_text, (self.panel_padding, y_pos), self.font, 
                    self.header_font_scale, header_color, self.header_thickness, cv2.LINE_AA)
        y_pos += self.line_height + self.panel_padding // 2
        
        if self.color_scheme: line_color = self.color_scheme.ui_colors['divider']
        else: line_color = (80,80,80)
        cv2.line(surface, (self.panel_padding, y_pos), (data['width'] - self.panel_padding, y_pos), line_color, 1)
        y_pos += self.line_height

        class_counts = data.get('class_counts', {})
        total_objects = data.get('total_objects', 0)

        if not class_counts:
            no_detections_text = "No objects detected."
            if self.color_scheme: text_color = self.color_scheme.text_colors['info']
            else: text_color = (150,180,150)
            cv2.putText(surface, no_detections_text, (self.panel_padding, y_pos), self.font, 
                        self.font_scale, text_color, self.font_thickness, cv2.LINE_AA)
            return surface

        # Display total objects
        total_text = f"Total Objects: {total_objects}"
        if self.color_scheme: 
            label_color = self.color_scheme.text_colors['normal']
            value_color = self.color_scheme.text_colors['highlight']
        else: 
            label_color = (180,180,180)
            value_color = (220,220,220)

        cv2.putText(surface, total_text, (self.panel_padding, y_pos), self.font, self.font_scale, label_color, self.font_thickness, cv2.LINE_AA)
        y_pos += self.line_height + self.panel_padding // 4

        # Display top N classes by count
        sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
        
        for i, (class_name, count) in enumerate(sorted_classes[:self.max_classes_to_show]):
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            text = f"{i+1}. {class_name}: {count} ({percentage:.1f}%)"
            truncated_text = self._truncate_text(text, data['width'] - (2 * self.panel_padding))
            
            if self.color_scheme:
                item_color = self.color_scheme.get_class_color(class_name) 
            else: # Fallback
                item_color = (200,200,200) # Default color

            cv2.putText(surface, truncated_text, (self.panel_padding, y_pos), self.font, 
                        self.font_scale, item_color, self.font_thickness, cv2.LINE_AA)
            y_pos += self.line_height
            if y_pos > self.panel_height - self.panel_padding: break
        
        return surface

class DetectedActionsPanel(BasePanel):
    """Displays a list of detected actions with their scores and associated object class."""
    def __init__(self, panel_id: str, color_scheme, panel_height: int = 220, max_items: int = 5, **kwargs):
        super().__init__(panel_id, color_scheme, **kwargs)
        self.panel_height = panel_height
        self.max_items = max_items

    def get_required_height(self, data: Any = None) -> int:
        return self.panel_height

    def render(self, data: Dict[str, Any]) -> np.ndarray:
        surface = np.zeros((self.panel_height, data['width'], 3), dtype=np.uint8)
        self._draw_panel_background(surface, panel_type='panel_accent') # Use accent color
        
        y_pos = self.panel_padding + self.line_height
        
        header_text = "DETECTED ACTIONS"
        if self.color_scheme: header_color = self.color_scheme.text_colors['header_accent']
        else: header_color = (200, 200, 150)
        cv2.putText(surface, header_text, (self.panel_padding, y_pos), self.font, 
                    self.header_font_scale, header_color, self.header_thickness, cv2.LINE_AA)
        y_pos += self.line_height + self.panel_padding // 2
        
        if self.color_scheme: line_color = self.color_scheme.ui_colors['divider']
        else: line_color = (80,80,80)
        cv2.line(surface, (self.panel_padding, y_pos), (data['width'] - self.panel_padding, y_pos), line_color, 1)
        y_pos += self.line_height

        actions_data = data.get('actions_log', []) # Expects a list of dicts

        if not actions_data:
            no_actions_text = "No actions detected."
            if self.color_scheme: text_color = self.color_scheme.text_colors['info']
            else: text_color = (150,180,150)
            cv2.putText(surface, no_actions_text, (self.panel_padding, y_pos), self.font, 
                        self.font_scale, text_color, self.font_thickness, cv2.LINE_AA)
            return surface

        for i, action_item in enumerate(actions_data[:self.max_items]):
            track_id = action_item.get('track_id', 'N/A')
            action_type = action_item.get('action_type', 'Unknown Action')
            action_score = action_item.get('score', 0.0)
            class_name = action_item.get('class_name', '?')
            
            # Remove ID display and show action with class name
            text_line1 = f"{action_type[:20]} ({class_name[:10]})"
            text_line2 = f"  Confidence: {action_score:.2f}"
            
            truncated_text1 = self._truncate_text(text_line1, data['width'] - (2 * self.panel_padding))
            truncated_text2 = self._truncate_text(text_line2, data['width'] - (2 * self.panel_padding) - self.panel_padding) # Indent second line

            # Implement consistent color coordination: Green → Good, Orange → In-between, Red → Bad
            if self.color_scheme:
                if action_score > 0.7:  # Red → Bad (High confidence threat)
                    color_line1 = (100, 100, 255)  # Red
                    color_line2_action = (100, 100, 255)  # Red
                elif action_score > 0.4:  # Orange → In-between (Medium confidence)
                    color_line1 = (100, 165, 255)  # Orange
                    color_line2_action = (100, 165, 255)  # Orange
                else:  # Green → Good (Low confidence/normal)
                    color_line1 = (100, 255, 100)  # Green
                    color_line2_action = (100, 255, 100)  # Green
            else: # Fallback
                color_line1 = (180,180,180)
                color_line2_action = (220,100,100) if action_score > 0.7 else (255,165,0) if action_score > 0.4 else (100,255,100)

            cv2.putText(surface, truncated_text1, (self.panel_padding, y_pos), self.font, 
                        self.font_scale, color_line1, self.font_thickness, cv2.LINE_AA)
            y_pos += self.line_height

            if y_pos > self.panel_height - self.panel_padding - self.line_height: break # Check before drawing second line

            cv2.putText(surface, truncated_text2, (self.panel_padding + self.panel_padding//2, y_pos), self.font, 
                        self.font_scale, color_line2_action, self.font_thickness, cv2.LINE_AA)

            y_pos += self.line_height + self.panel_padding // 4 # Extra padding between entries
            if y_pos > self.panel_height - self.panel_padding: break
        
        return surface

# Example Usage (for testing these panels directly)
if __name__ == '__main__':
    # Mock color scheme
    class MockColorScheme:
        def __init__(self):
            self.background_colors = {
                'panel_normal': (50, 50, 50), 
                'panel_warning': (120, 100, 40),
                'panel_danger': (150, 50, 50)
            }
            self.text_colors = {
                'header_accent': (200, 200, 150),
                'critical': (255, 180, 180),
                'warning': (255, 230, 150),
                'info': (180, 220, 255),
                'normal': (200, 200, 200)
            }
            self.ui_colors = {'divider': (100, 100, 100), 'border_normal': (80,80,80)}

    mock_scheme = MockColorScheme()
    panel_width = 380 # Example width for sidebar panels

    # Test SystemStatusPanel
    status_panel = SystemStatusPanel("status", mock_scheme)
    status_data = {
        'width': panel_width,
        'avg_fps': "28.5", 
        'object_count': 5, 
        'detection_count': 12,
        'cpu_usage': 73.7, # Test data
        'ram_usage': 71.6,  # Test data
        'system_status': "SLOW" # Test data
    }
    status_surface = status_panel.render(status_data)
    cv2.imshow("System Status Panel", status_surface)

    # Test AnomalyListPanel (empty)
    anomaly_panel_empty = AnomalyListPanel("anomalies_empty", mock_scheme)
    anomaly_data_empty = {'width': panel_width, 'anomalies': []}
    anomaly_surface_empty = anomaly_panel_empty.render(anomaly_data_empty)
    cv2.imshow("Anomaly List Panel (Empty)", anomaly_surface_empty)

    # Test AnomalyListPanel (with data)
    anomaly_panel = AnomalyListPanel("anomalies", mock_scheme)
    anomalies_sample = [
        {'track_id': 101, 'action_type': 'Suspicious Loitering', 'score': 0.65, 'is_critical': False},
        {'track_id': 102, 'action_type': 'Running Fast', 'score': 0.88, 'is_critical': True},
        {'track_id': 103, 'action_type': 'Unusual Object Interaction', 'score': 0.72, 'is_critical': True},
        {'track_id': 104, 'action_type': 'Normal Walking', 'score': 0.20, 'is_critical': False},
    ]
    anomaly_data = {'width': panel_width, 'anomalies': anomalies_sample}
    anomaly_surface = anomaly_panel.render(anomaly_data)
    cv2.imshow("Anomaly List Panel (With Data)", anomaly_surface)

    # Test GlobalDetectionsPanel
    detections_panel = GlobalDetectionsPanel("detections", mock_scheme)
    detections_data = {
        'width': panel_width,
        'class_counts': {'person': 7, 'car': 2, 'bicycle': 1, 'dog': 3, 'cat': 1, 'bird': 0},
        'total_objects': 14
    }
    detections_surface = detections_panel.render(detections_data)
    cv2.imshow("Global Detections Panel", detections_surface)

    # Test DetectedActionsPanel
    actions_panel = DetectedActionsPanel("actions_log", mock_scheme)
    actions_log_data_sample = [
        {'track_id': 'T1', 'action_type': 'Walking Slowly', 'score': 0.45, 'class_name': 'person'},
        {'track_id': 'T2', 'action_type': 'Brandishing a Big Knife', 'score': 0.92, 'class_name': 'person'},
        {'track_id': 'T3', 'action_type': 'Standing Still', 'score': 0.20, 'class_name': 'person'},
        {'track_id': 'C1', 'action_type': 'Driving Normally', 'score': 0.15, 'class_name': 'car'},
        {'track_id': 'T2', 'action_type': 'Running Fast', 'score': 0.78, 'class_name': 'person'},
    ]
    actions_log_data = {
        'width': panel_width,
        'actions_log': actions_log_data_sample
    }
    actions_surface = actions_panel.render(actions_log_data)
    cv2.imshow("Detected Actions Panel", actions_surface)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 