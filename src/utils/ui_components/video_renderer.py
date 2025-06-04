import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict

class VideoPaneRenderer:
    """Handles processing and rendering of the video feed, including object tracks and labels."""

    def __init__(self, color_scheme=None, font_scale: float = 0.55, font_thickness: int = 1, 
                 line_height: int = 22, panel_padding: int = 10, object_label_bg_alpha: float = 0.6):
        self.color_scheme = color_scheme # Instance of ColorScheme
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.line_height = line_height
        self.panel_padding = panel_padding
        self.object_label_bg_alpha = object_label_bg_alpha

    def _apply_alert_overlay(self, frame: np.ndarray):
        """Apply static red alert overlay for critical situations."""
        # This logic is taken from the original EnhancedVisualizer
        alert_overlay_color = self.color_scheme.ui_colors.get('video_alert_overlay', (0,0,40)) # Darker red for less distraction
        alert_overlay_alpha = self.color_scheme.ui_params.get('video_alert_overlay_alpha', 0.1)
        
        alert_overlay = np.full(frame.shape, alert_overlay_color, dtype=frame.dtype)
        cv2.addWeighted(frame, 1.0 - alert_overlay_alpha, alert_overlay, alert_overlay_alpha, 0, frame)

    def _draw_object_labels(self, frame_for_text: np.ndarray, overlay_for_bg: np.ndarray, 
                              x1: int, y1: int, y2: int, 
                              track_id: str, class_name: str, confidence: float, 
                              action_type: Optional[str], action_score: Optional[float],
                              box_color: Tuple[int, int, int], is_high_risk: bool):
        """Draw semantic object labels without IDs and with consistent color coordination."""
        
        # Create labels without ID display
        labels = []
        labels.append(f"{class_name}")  # Just class name, no ID
        labels.append(f"Conf: {confidence:.2f}")  # Confidence
        if action_type: 
            labels.append(f"{action_type}")  # Action type
        if action_score is not None:
            labels.append(f"Score: {action_score:.2f}")  # Action score
        
        # Calculate space needed for labels
        text_height_one_line = cv2.getTextSize("Tg", self.font, self.font_scale, self.font_thickness)[0][1] + 5
        label_block_height = len(labels) * (text_height_one_line + 4) # +4 for spacing between lines
        
        # Position for labels
        label_y_start = y1 - label_block_height - 10
        if label_y_start < text_height_one_line: # If too high (goes off screen or overlaps top), position below box
            label_y_start = y2 + 5

        # Calculate label width based on longest label
        max_text_width = 0
        for label_text in labels:
            (text_w, _), _ = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)
            if text_w > max_text_width:
                max_text_width = text_w
        
        bg_width = max_text_width + 2 * self.panel_padding
        current_bg_y = label_y_start

        # Use appropriate background color
        if self.color_scheme:
            bg_color = self.color_scheme.background_colors['panel_danger'] if is_high_risk else self.color_scheme.background_colors['panel_accent']
            label_bg_border_color = self.color_scheme.ui_colors.get('label_bg_border', box_color)
        else: # Fallback
            bg_color = (100,0,0) if is_high_risk else (30,30,30)
            label_bg_border_color = box_color

        # Draw background for the entire label block on the overlay
        bg_rect_y2 = label_y_start + label_block_height + self.panel_padding // 2
        cv2.rectangle(overlay_for_bg, (x1, label_y_start), (x1 + bg_width, bg_rect_y2), bg_color, -1)
        cv2.rectangle(overlay_for_bg, (x1, label_y_start), (x1 + bg_width, bg_rect_y2), label_bg_border_color, 1)

        # Draw labels with consistent color coordination on the frame_for_text
        text_y = label_y_start + text_height_one_line + self.panel_padding // 2
        for i, label_text in enumerate(labels):
            if i == 0:  # Class name
                if is_high_risk:
                    color = (100, 100, 255)  # Red for high risk classes
                else:
                    color = (220, 220, 220)  # White for normal classes
            elif i == 1:  # Confidence - use consistent color coordination
                if confidence > 0.7:  # Green → Good (High confidence)
                    color = (100, 255, 100)  # Green
                elif confidence > 0.4:  # Orange → In-between (Medium confidence)
                    color = (100, 165, 255)  # Orange
                else:  # Red → Bad (Low confidence)
                    color = (100, 100, 255)  # Red
            elif i == 2 and action_type:  # Action Type - use action score for color
                if action_score is not None:
                    if action_score > 0.7:  # Red → Bad (High threat action)
                        color = (100, 100, 255)  # Red
                    elif action_score > 0.4:  # Orange → In-between (Medium threat)
                        color = (100, 165, 255)  # Orange
                    else:  # Green → Good (Low threat action)
                        color = (100, 255, 100)  # Green
                else:
                    color = (180, 180, 180)  # Gray for unknown
            elif i == 3 and action_score is not None:  # Action Score
                if action_score > 0.7:  # Red → Bad (High threat)
                    color = (100, 100, 255)  # Red
                elif action_score > 0.4:  # Orange → In-between (Medium threat)
                    color = (100, 165, 255)  # Orange
                else:  # Green → Good (Low threat)
                    color = (100, 255, 100)  # Green
            else:
                color = (220, 220, 220)  # Default white

            cv2.putText(frame_for_text, label_text, (x1 + self.panel_padding, text_y), 
                        self.font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)
            text_y += text_height_one_line + 4 # Move to next line position

    def _draw_object_tracks(self, frame_for_text_and_boxes: np.ndarray, 
                              overlay_for_label_bg: np.ndarray, 
                              tracks: List[Dict]):
        """Draw object tracks with semantic color coding."""
        if self.color_scheme is None: # Fallback if no color scheme
            high_risk_objects = {'knife', 'scissors', 'gun', 'weapon'} # Simplified

        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = str(track.get('object_id', track.get('track_id', -1)))
            class_name = track.get('class_name', 'Unknown')
            confidence = track.get('confidence', 0.0)
            action_type = track.get('action_type', None) # Get action_type
            action_score = track.get('action_score', None) # Get action_score
            
            if self.color_scheme:
                box_color = self.color_scheme.get_class_color(class_name)
                is_high_risk = class_name.lower() in self.color_scheme.high_risk_objects
            else: # Fallback
                box_color = (0,255,0) # Default green
                is_high_risk = class_name.lower() in high_risk_objects
            
            thickness = 3 if is_high_risk else 2
            cv2.rectangle(frame_for_text_and_boxes, (x1, y1), (x2, y2), box_color, thickness)
            
            self._draw_object_labels(frame_for_text_and_boxes, overlay_for_label_bg, 
                                     x1, y1, y2, track_id, class_name, confidence, 
                                     action_type, action_score,
                                     box_color, is_high_risk)

    def render_video_pane(self, frame: np.ndarray, tracks: List[Dict], 
                            max_width: int, max_height: int, 
                            abnormal_detected: bool) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Process and resize video frame, returning the rendered pane and its final rect (x,y,w,h) on the dashboard."""
        
        # Work on a copy for drawing video-specific elements
        video_content_frame = frame.copy()
        # Create a separate overlay for object label backgrounds (will be blended with alpha)
        label_background_overlay = np.zeros_like(video_content_frame, dtype=np.uint8)

        if abnormal_detected and self.color_scheme:
            self._apply_alert_overlay(video_content_frame) # Modifies video_content_frame in place
        
        self._draw_object_tracks(video_content_frame, label_background_overlay, tracks)
        
        # Blend the object label backgrounds onto the video_content_frame
        cv2.addWeighted(label_background_overlay, self.object_label_bg_alpha, 
                        video_content_frame, 1.0, 0, video_content_frame)
        
        # Calculate resize dimensions maintaining aspect ratio
        h, w = video_content_frame.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        
        new_width = max_width
        new_height = int(new_width / aspect_ratio) if aspect_ratio > 0 else max_height

        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        
        if new_width <= 0 or new_height <=0: # Safety for invalid dims
             final_pane = np.zeros((max_height, max_width, 3), dtype=np.uint8)
             final_rect_on_dashboard = (0,0,max_width, max_height)
             return final_pane, final_rect_on_dashboard

        resized_pane = cv2.resize(video_content_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create a final pane of max_width x max_height and center the resized_pane in it
        # This ensures the video pane is always the same size for the layout manager
        final_pane = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        if self.color_scheme:
            final_pane[:] = self.color_scheme.ui_colors.get('video_bg', (10,10,10)) # Dark background
        else:
            final_pane[:] = (10,10,10)

        offset_x = (max_width - new_width) // 2
        offset_y = (max_height - new_height) // 2
        
        final_pane[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_pane
        
        # The rect returned is relative to the video *area* on the dashboard, not the dashboard itself.
        # The main visualizer will add the video area's base x, y.
        actual_video_rect_in_pane = (offset_x, offset_y, new_width, new_height)

        return final_pane, actual_video_rect_in_pane


# Example Usage (for testing this module directly)
if __name__ == '__main__':
    # Mock color scheme for testing
    class MockColorScheme:
        def __init__(self):
            self.background_colors = {
                'panel_danger': (150, 30, 30),
                'panel_accent': (60, 60, 60),
            }
            self.text_colors = {
                'critical': (255, 200, 200),
                'highlight': (230, 230, 230),
                'info_accent': (180, 180, 220),
                'success': (200, 255, 200),
                'warning': (255, 230, 180)
            }
            self.ui_colors = {
                'video_alert_overlay': (0,0,40),
                'label_bg_border': (150,150,150),
                'video_bg': (10,10,10)
            }
            self.ui_params = {
                'video_alert_overlay_alpha': 0.15
            }
            self.high_risk_objects = {'knife', 'gun'}
        def get_class_color(self, class_name):
            if class_name == 'person': return (0, 255, 0)
            if class_name == 'knife': return (255, 0, 0)
            return (200, 200, 200)
        # Add mock get_action_severity_color for testing
        def get_action_severity_color(self, score: float):
            if score > 0.75: return self.text_colors['critical']
            if score > 0.3: return (255, 180, 0) # Mock 'suspicious'
            return (0, 200, 0) # Mock 'normal'

    mock_scheme = MockColorScheme()
    renderer = VideoPaneRenderer(color_scheme=mock_scheme)
    
    # Create a sample frame
    sample_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(sample_frame, "Sample Video Content", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (200,200,200), 3)

    # Sample tracks
    sample_tracks = [
        {'bbox': [100, 100, 200, 300], 'object_id': 1, 'class_name': 'person', 'confidence': 0.95, 'action_type': 'Walking', 'action_score': 0.3},
        {'bbox': [250, 150, 350, 400], 'object_id': 2, 'class_name': 'knife', 'confidence': 0.88, 'action_type': 'Threatening', 'action_score': 0.9},
        {'bbox': [400, 200, 450, 300], 'object_id': 3, 'class_name': 'bottle', 'confidence': 0.70, 'action_type': 'Idle', 'action_score': 0.1},
    ]

    video_area_w, video_area_h = 980, 820 # Example dimensions from layout manager

    # Test normal rendering
    rendered_pane_normal, video_rect_normal = renderer.render_video_pane(sample_frame.copy(), sample_tracks, video_area_w, video_area_h, False)
    cv2.imshow("Rendered Video Pane (Normal)", rendered_pane_normal)
    print(f"Video Pane (Normal) Dimensions: {rendered_pane_normal.shape}, Actual Video Rect in Pane: {video_rect_normal}")

    # Test alert rendering
    rendered_pane_alert, video_rect_alert = renderer.render_video_pane(sample_frame.copy(), sample_tracks, video_area_w, video_area_h, True)
    cv2.imshow("Rendered Video Pane (Alert)", rendered_pane_alert)
    print(f"Video Pane (Alert) Dimensions: {rendered_pane_alert.shape}, Actual Video Rect in Pane: {video_rect_alert}")

    cv2.waitKey(0)
    cv2.destroyAllWindows() 