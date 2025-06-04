import cv2
import numpy as np
from collections import Counter
import colorsys
import time
import psutil
from datetime import datetime

class PerformanceTracker:
    """Track system performance metrics for display"""
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.last_update = time.time()
        
    def update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            
    def update_system_metrics(self):
        """Update system performance metrics"""
        current_time = time.time()
        # Update more frequently for better real-time monitoring (every 0.5 seconds)
        if current_time - self.last_update >= 0.5:
            # Use interval=None for immediate reading instead of blocking
            self.cpu_percent = psutil.cpu_percent(interval=None)
            self.memory_percent = psutil.virtual_memory().percent
            self.last_update = current_time

def get_class_color(class_name):
    """Generate consistent colors for each object class using hash-based HSV color generation."""
    class_colors = {
    'person': (0, 255, 0),
    'backpack': (255, 165, 0),
    'suitcase': (255, 20, 147),
    'handbag': (255, 105, 180),
    'skis': (0, 191, 255),
    'cell phone': (255, 0, 255),
    'skateboard': (50, 205, 50),
    'bottle': (255, 69, 0),
    'cup': (255, 140, 0),
    'knife': (255, 0, 0),
    'scissors': (220, 20, 60),
    'car': (0, 0, 255),
    'truck': (0, 100, 255),
    'motorcycle': (138, 43, 226),
    'bicycle': (75, 0, 130),
    }
    
    if class_name in class_colors:
        return class_colors[class_name]
    
    hash_val = hash(class_name) % 360
    rgb = colorsys.hsv_to_rgb(hash_val / 360.0, 0.8, 0.9)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

def is_abnormal_behavior(actions):
    """Check if any of the detected actions are abnormal."""
    from src.config import ABNORMAL_ACTION_IDS
    
    for action in actions:
        action_id = action.get('action_class_id', -1)
        if action_id in ABNORMAL_ACTION_IDS:
            return True, action
    return False, None

def draw_results(frame, tracks, actions, anomaly_scores):
    global _performance_tracker
    
    # Update performance tracking
    _performance_tracker.update_fps()
    _performance_tracker.update_system_metrics()
    
    display_frame = frame.copy()
    overlay = display_frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Define colors
    text_color_normal = (255, 255, 255)  # White
    text_color_action = (0, 255, 255)    # Yellow for actions
    text_color_warning = (0, 0, 255)     # Red for warnings
    text_color_detection = (0, 255, 0)   # Green for detection stats
    text_color_abnormal = (0, 0, 255)    # Red for abnormal actions

    bg_color = (0, 0, 0)
    alpha = 0.7

    line_height_estimate = cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1]
    line_height = int(line_height_estimate * 2.0)

    # Check for abnormal behavior and create screen flash effect
    abnormal_detected, abnormal_action = is_abnormal_behavior(actions)
    has_warnings = any(score.get('warning_level', 0) > 0 for score in anomaly_scores)
    
    # Static red overlay for alerts (no pulsing)
    if abnormal_detected or has_warnings:
        # Create static red overlay for screen flash
        red_overlay = np.zeros_like(display_frame)
        red_overlay[:] = (0, 0, 255)  # Red color
        cv2.addWeighted(display_frame, 0.9, red_overlay, 0.1, 0, display_frame)

    # Collect global detection statistics
    class_counts = Counter()
    total_detections = len(tracks)
    
    for track in tracks:
        class_name = track.get('class_name', 'Unknown')
        class_counts[class_name] += 1

    # Get top 5 detected classes
    top_5_classes = class_counts.most_common(5)

    # Draw enhanced global detection statistics panel with rounded corners
    detection_y_start = 30
    detection_x_start = frame.shape[1] - 350  # Increased width for better layout
    
    detection_labels = ["GLOBAL DETECTIONS"]
    detection_labels.append(f"Total Objects: {total_detections}")
    detection_labels.append("Top 5 Classes")
    
    for i, (class_name, count) in enumerate(top_5_classes):
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        detection_labels.append(f"  {i+1}. {class_name}: {count} ({percentage:.1f}%)")
    
    # Draw detection statistics background with enhanced styling
    max_text_width = 0
    for label in detection_labels:
        (text_width, _), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_width)
    
    detection_bg_height = len(detection_labels) * line_height + 20
    detection_bg_width = max_text_width + 30
    
    # Ensure the detection panel fits within the frame
    if detection_x_start + detection_bg_width > frame.shape[1]:
        detection_x_start = frame.shape[1] - detection_bg_width - 10
    
    # Enhanced panel with gradient and rounded corners
    panel_overlay = overlay.copy()
    cv2.rectangle(panel_overlay, (detection_x_start - 10, detection_y_start - 10), 
                 (detection_x_start + detection_bg_width, detection_y_start + detection_bg_height), 
                 (50, 50, 50), -1)  # Darker background
    cv2.addWeighted(panel_overlay, 0.8, overlay, 0.2, 0, overlay)
    
    # Border with rounded effect
    cv2.rectangle(overlay, (detection_x_start - 10, detection_y_start - 10), 
                 (detection_x_start + detection_bg_width, detection_y_start + detection_bg_height), 
                 (120, 120, 120), 2)
    
    # Draw detection statistics text with enhanced formatting
    for i, label in enumerate(detection_labels):
        text_y = detection_y_start + (i * line_height)
        if i == 0:  # Header with larger font
            cv2.putText(display_frame, label, (detection_x_start, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        elif i == 1:  # Total count
            cv2.putText(display_frame, label, (detection_x_start, text_y), 
                       font, font_scale, text_color_detection, font_thickness, cv2.LINE_AA)
        elif i == 2:  # Separator
            cv2.putText(display_frame, label, (detection_x_start, text_y), 
                       font, font_scale, text_color_normal, font_thickness, cv2.LINE_AA)
        else:  # Class statistics
            cv2.putText(display_frame, label, (detection_x_start, text_y), 
                       font, font_scale, text_color_detection, font_thickness, cv2.LINE_AA)

    # Draw tracked objects with CLASS-SPECIFIC COLORS
    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        track_id = track.get('object_id', track.get('track_id', -1))
        class_name = track.get('class_name', 'Unknown')
        confidence = track.get('confidence', 0.0)

        # Get class-specific color
        current_box_color = get_class_color(class_name)
        current_text_color = text_color_normal

        # Create labels for this track
        labels_to_draw = []
        labels_to_draw.append(f"ID:{track_id}")
        labels_to_draw.append(f"{class_name}:{confidence:.2f}")

        # Draw CLASS-COLORED bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), current_box_color, 2)
        
        # Calculate text position
        total_text_block_height = len(labels_to_draw) * line_height
        start_text_y = y1 - 10 - total_text_block_height + line_height_estimate
        if start_text_y < line_height_estimate:
            start_text_y = y2 + line_height_estimate + 5

        # Draw labels with background
        for i, label_text in enumerate(labels_to_draw):
            text_y_baseline = start_text_y + (i * line_height)
            
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            bg_x1 = x1
            bg_y1 = text_y_baseline - text_height - baseline // 2
            bg_x2 = x1 + text_width
            bg_y2 = text_y_baseline + baseline // 2

            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            cv2.putText(display_frame, label_text, (x1, text_y_baseline), 
                        font, font_scale, current_text_color, font_thickness, cv2.LINE_AA)

        # Draw enhanced global actions with confidence bars and status (always show)
    action_y_start = 30
    action_x_start = 10
    
    # Create action info box with enhanced abnormal detection
    action_labels = ["DETECTED ACTIONS"]
    
    if not actions:
        action_labels.append("No actions detected")
    
    for action in actions:
        action_name = action.get('action_name', 'Unknown Action')
        action_conf = action.get('action_confidence', 0.0)
        action_id = action.get('action_class_id', -1)
        
        # Check if this action is abnormal
        from src.config import ABNORMAL_ACTION_IDS
        is_abnormal = action_id in ABNORMAL_ACTION_IDS
        status = "[NOK] ABNORMAL" if is_abnormal else "[OK] NORMAL"
        
        if action.get('track_id') is None:
            # Global action
            action_labels.append(f"Global: {status} - {action_name}")
            # Add confidence visualization
            action_labels.append(f"  Confidence: {action_conf:.1%}")
        else:
            # Track-specific action
            track_id = action.get('track_id')
            action_labels.append(f"ID{track_id}: {status} - {action_name}")
            action_labels.append(f"  Confidence: {action_conf:.1%}")
    
    # Draw action info background with fixed dimensions
    # Use fixed dimensions to prevent panel resizing based on content
    action_bg_width = 350  # Fixed width
    action_bg_height = 120  # Fixed height
    
    # Enhanced background with transparency
    action_overlay = overlay.copy()
    bg_color_action = (50, 0, 0) if abnormal_detected else (0, 30, 0)  # Dark red/green tint
    cv2.rectangle(action_overlay, (action_x_start - 10, action_y_start - 10), 
                 (action_x_start + action_bg_width, action_y_start + action_bg_height), 
                 bg_color_action, -1)
    cv2.addWeighted(action_overlay, 0.8, overlay, 0.2, 0, overlay)
    
    # Border
    border_color = (100, 100, 100)
    cv2.rectangle(overlay, (action_x_start - 10, action_y_start - 10), 
                 (action_x_start + action_bg_width, action_y_start + action_bg_height), 
                 border_color, 2)
    
    # Draw action text with confidence bars within fixed bounds
    conf_bar_drawn = False
    max_action_lines = (action_bg_height - 40) // line_height  # Calculate max lines that fit
    for i, label in enumerate(action_labels):
        if i >= max_action_lines:  # Stop if we exceed the fixed panel height
            break
            
        text_y = action_y_start + (i * line_height)
        
        # Truncate text if it's too long for fixed width
        if len(label) > 45:  # Approximate character limit for fixed width
            label = label[:42] + "..."
            
        if i == 0:  # Header
            cv2.putText(display_frame, label, (action_x_start, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color_normal, 2, cv2.LINE_AA)
        elif "Confidence:" in label:  # Confidence line - draw bar
            if not conf_bar_drawn and actions:  # Only draw one confidence bar per action
                action_conf = actions[0].get('action_confidence', 0.0)
                conf_color = text_color_abnormal if abnormal_detected else text_color_action
                
                # Draw confidence bar
                bar_x = action_x_start + 20
                bar_y = text_y - 8
                bar_width = 150
                bar_height = 8
                draw_confidence_bar(display_frame, bar_x, bar_y, bar_width, bar_height, 
                                  action_conf, "Action", conf_color)
                conf_bar_drawn = True
        else:  # Action items
            # Use red text for abnormal actions
            if "[NOK] ABNORMAL" in label:
                color = text_color_abnormal
            else:
                color = text_color_action
            cv2.putText(display_frame, label, (action_x_start, text_y), 
                       font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Draw enhanced anomaly warnings with better styling (always show)
    # Calculate consistent position regardless of actions
    action_panel_height = (len(action_labels) * line_height + 20) if 'action_labels' in locals() else 80
    warning_y_start = 30 + action_panel_height + 20  # Fixed spacing after action panel
    warning_x_start = 10
    
    warning_labels = ["ANOMALY WARNINGS"]
    active_warnings = 0
    
    if anomaly_scores:
        for score in anomaly_scores:
            warning_level = score.get('warning_level', 0)
            description = score.get('description', 'Unknown Warning')
            track_id = score.get('track_id', 'Global')
            
            if warning_level > 0:
                active_warnings += 1
                if warning_level >= 3:
                    severity_icon = "[!!!]"
                elif warning_level >= 2:
                    severity_icon = "[!]"
                else:
                    severity_icon = "[*]"
                warning_labels.append(f"{severity_icon} ID{track_id}: {description}")
    
    if active_warnings == 0:
        warning_labels.append("No warnings")
    
    # Always show warning panel with fixed dimensions (no resizing)
    # Use fixed dimensions to prevent panel resizing based on content
    warning_bg_width = 350  # Fixed width
    warning_bg_height = 150  # Fixed height
    
    # Enhanced warning panel
    warning_overlay = overlay.copy()
    cv2.rectangle(warning_overlay, (warning_x_start - 10, warning_y_start - 10), 
                 (warning_x_start + warning_bg_width, warning_y_start + warning_bg_height), 
                 (0, 0, 80), -1)  # Dark red background
    cv2.addWeighted(warning_overlay, 0.8, overlay, 0.2, 0, overlay)
    
    # Static red border for warnings
    border_color = (0, 0, 255) if active_warnings > 0 else (100, 100, 100)
    cv2.rectangle(overlay, (warning_x_start - 10, warning_y_start - 10), 
                 (warning_x_start + warning_bg_width, warning_y_start + warning_bg_height), 
                 border_color, 3)
    
    # Draw warning text within fixed bounds
    max_lines = (warning_bg_height - 40) // line_height  # Calculate max lines that fit
    for i, label in enumerate(warning_labels):
        if i >= max_lines:  # Stop if we exceed the fixed panel height
            break
            
        text_y = warning_y_start + (i * line_height)
        
        # Truncate text if it's too long for fixed width
        if len(label) > 45:  # Approximate character limit for fixed width
            label = label[:42] + "..."
            
        if i == 0:  # Header
            cv2.putText(display_frame, label, (warning_x_start, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color_normal, 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, label, (warning_x_start, text_y), 
                       font, font_scale, text_color_warning, font_thickness, cv2.LINE_AA)

    # Add enhanced global behavior status indicator
    behavior_status_y = frame.shape[0] - 60
    behavior_status_x = 10
    
    if abnormal_detected or has_warnings:
        status_text = "ABNORMAL BEHAVIOR DETECTED"
        status_color = text_color_abnormal
        # Static red background
        cv2.rectangle(overlay, (behavior_status_x - 10, behavior_status_y - 30), 
                     (behavior_status_x + 420, behavior_status_y + 15), 
                     (0, 0, 100), -1)
        
        # Static red border
        cv2.rectangle(overlay, (behavior_status_x - 10, behavior_status_y - 30), 
                     (behavior_status_x + 420, behavior_status_y + 15), 
                     (0, 0, 255), 3)
    else:
        status_text = "[OK] NORMAL BEHAVIOR"
        status_color = text_color_detection
        cv2.rectangle(overlay, (behavior_status_x - 10, behavior_status_y - 30), 
                     (behavior_status_x + 220, behavior_status_y + 15), 
                     (0, 50, 0), -1)
        cv2.rectangle(overlay, (behavior_status_x - 10, behavior_status_y - 30), 
                     (behavior_status_x + 220, behavior_status_y + 15), 
                     (0, 150, 0), 2)
    
    cv2.putText(display_frame, status_text, (behavior_status_x, behavior_status_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    # Blend overlay with original frame
    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
    
    # Add system performance panel (drawn on top)
    draw_system_info_panel(display_frame, _performance_tracker)

    return display_frame

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=10):
    """Draw rectangle with rounded corners"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_confidence_bar(img, x, y, width, height, confidence, label, color):
    """Draw a confidence bar with percentage"""
    # Background bar
    cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Confidence fill
    fill_width = int(width * confidence)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
    # Text
    text = f"{label}: {confidence:.1%}"
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_system_info_panel(img, performance_tracker):
    """Draw comprehensive system information panel"""
    panel_width = 280
    panel_height = 140
    panel_x = img.shape[1] - panel_width - 10
    panel_y = img.shape[0] - panel_height - 10
    
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    # Border
    cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                  (100, 100, 100), 2)
    
    # Title
    cv2.putText(img, "SYSTEM PERFORMANCE", (panel_x + 10, panel_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, f"Time: {timestamp}", (panel_x + 10, panel_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # FPS with color coding
    fps_color = (0, 255, 0) if performance_tracker.current_fps > 25 else (0, 255, 255) if performance_tracker.current_fps > 15 else (0, 0, 255)
    cv2.putText(img, f"FPS: {performance_tracker.current_fps:.1f}", (panel_x + 10, panel_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_color, 1)
    
    # CPU Usage Bar
    cpu_color = (0, 255, 0) if performance_tracker.cpu_percent < 70 else (0, 255, 255) if performance_tracker.cpu_percent < 90 else (0, 0, 255)
    draw_confidence_bar(img, panel_x + 10, panel_y + 75, 120, 12, 
                       performance_tracker.cpu_percent / 100.0, "CPU", cpu_color)
    
    # Memory Usage Bar
    mem_color = (0, 255, 0) if performance_tracker.memory_percent < 70 else (0, 255, 255) if performance_tracker.memory_percent < 90 else (0, 0, 255)
    draw_confidence_bar(img, panel_x + 140, panel_y + 75, 120, 12, 
                       performance_tracker.memory_percent / 100.0, "RAM", mem_color)
    
    # Status indicator
    overall_status = "OPTIMAL" if performance_tracker.current_fps > 25 and performance_tracker.cpu_percent < 70 else "MODERATE" if performance_tracker.current_fps > 15 else "SLOW"
    status_color = (0, 255, 0) if overall_status == "OPTIMAL" else (0, 255, 255) if overall_status == "MODERATE" else (0, 0, 255)
    cv2.putText(img, f"Status: {overall_status}", (panel_x + 10, panel_y + 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

# Global performance tracker instance
_performance_tracker = PerformanceTracker() 