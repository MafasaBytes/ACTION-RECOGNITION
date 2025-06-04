"""
Color Schemes for Computer Vision Anomaly Detection Pipeline

This module provides multiple color coordination strategies.
"""

import colorsys
import cv2
import numpy as np

class ColorScheme:
    """Base class for color schemes"""
    
    def __init__(self, name):
        self.name = name
        self.text_colors = {}
        self.background_colors = {}
        self.class_colors = {}
        self.status_colors = {}
        self.ui_colors = {}
    
    def get_class_color(self, class_name):
        """Get color for object class"""
        return self.class_colors.get(class_name, self._generate_fallback_color(class_name))
    
    def _generate_fallback_color(self, class_name):
        """Generate consistent fallback color for unknown classes"""
        hash_val = hash(class_name) % 360
        rgb = colorsys.hsv_to_rgb(hash_val / 360.0, 0.6, 0.8)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

class SemanticColorScheme(ColorScheme):
    """Color scheme based on semantic meaning and risk levels"""
    
    def __init__(self):
        super().__init__("Semantic Risk-Based")
        
        self.text_colors = {
            'normal': (240, 240, 240),
            'info': (100, 200, 255),
            'success': (100, 255, 150),
            'warning': (255, 200, 100),
            'danger': (255, 100, 100),
            'critical': (255, 50, 50),
            'header': (255, 255, 255),
        }
        
        self.background_colors = {
            'panel_normal': (40, 40, 40),
            'panel_info': (20, 40, 60),
            'panel_success': (20, 50, 30),
            'panel_warning': (60, 45, 20),
            'panel_danger': (60, 20, 20),
            'overlay_flash': (80, 20, 20),
        }
        
        self.class_colors = {
            'knife': (255, 50, 50),
            'scissors': (255, 100, 50),
            'gun': (255, 0, 0),
            'person': (100, 255, 200),
            'car': (100, 150, 255),
            'truck': (50, 100, 255),
            'motorcycle': (150, 100, 255),
            'bicycle': (200, 150, 255),
            'backpack': (255, 180, 100),
            'handbag': (255, 150, 150),
            'suitcase': (255, 200, 150),
            'skateboard': (150, 255, 100),
            'skis': (100, 255, 150),
            'cell phone': (200, 100, 255),
            'laptop': (150, 50, 255),
            'bottle': (255, 255, 100),
            'cup': (255, 200, 50),
        }
        
        self.status_colors = {
            'optimal': (100, 255, 150),
            'good': (150, 255, 100),
            'moderate': (255, 200, 100),
            'poor': (255, 150, 100),
            'critical': (255, 100, 100),
            'abnormal_pulse': (255, 50, 50),
        }
        
        self.ui_colors = {
            'border_normal': (120, 120, 120),
            'border_active': (100, 200, 255),
            'border_warning': (255, 200, 100),
            'border_danger': (255, 100, 100),
            'confidence_bg': (60, 60, 60),
            'confidence_border': (180, 180, 180),
        }

class HighContrastScheme(ColorScheme):
    """High contrast scheme optimized for readability"""
    
    def __init__(self):
        super().__init__("High Contrast")
        
        self.text_colors = {
            'normal': (255, 255, 255),
            'info': (0, 255, 255),
            'success': (0, 255, 0),
            'warning': (255, 255, 0),
            'danger': (255, 0, 0),
            'critical': (255, 0, 255),
            'header': (255, 255, 255),
        }
        
        self.background_colors = {
            'panel_normal': (0, 0, 0),
            'panel_info': (0, 0, 80),
            'panel_success': (0, 80, 0),
            'panel_warning': (80, 80, 0),
            'panel_danger': (80, 0, 0),
            'overlay_flash': (100, 0, 0),
        }
        
        self.class_colors = {
            'person': (0, 255, 0),
            'car': (0, 0, 255),
            'truck': (0, 100, 255),
            'motorcycle': (100, 0, 255),
            'bicycle': (0, 255, 255),
            'knife': (255, 0, 0),
            'scissors': (255, 100, 0),
            'backpack': (255, 255, 0),
            'handbag': (255, 0, 255),
            'suitcase': (255, 150, 0),
            'cell phone': (150, 0, 255),
            'bottle': (255, 255, 100),
            'cup': (255, 200, 0),
            'skateboard': (100, 255, 0),
            'skis': (0, 255, 150),
        }
        
        self.status_colors = {
            'optimal': (0, 255, 0),
            'good': (100, 255, 0),
            'moderate': (255, 255, 0),
            'poor': (255, 100, 0),
            'critical': (255, 0, 0),
            'abnormal_pulse': (255, 0, 255),
        }
        
        self.ui_colors = {
            'border_normal': (255, 255, 255),
            'border_active': (0, 255, 255),
            'border_warning': (255, 255, 0),
            'border_danger': (255, 0, 0),
            'confidence_bg': (0, 0, 0),
            'confidence_border': (255, 255, 255),
        }

class ProfessionalDashboardScheme(ColorScheme):
    """Professional monitoring dashboard aesthetic"""
    
    def __init__(self):
        super().__init__("Professional Dashboard")
        
        self.text_colors = {
            'normal': (220, 220, 220),
            'info': (120, 180, 255),
            'success': (120, 200, 140),
            'warning': (255, 180, 80),
            'danger': (255, 120, 120),
            'critical': (255, 80, 80),
            'header': (240, 240, 240),
        }
        
        self.background_colors = {
            'panel_normal': (45, 45, 50),
            'panel_info': (40, 50, 65),
            'panel_success': (40, 55, 45),
            'panel_warning': (65, 55, 40),
            'panel_danger': (65, 45, 45),
            'overlay_flash': (80, 40, 40),
        }
        
        self.class_colors = {
            'person': (120, 200, 180),
            'car': (120, 160, 220),
            'truck': (100, 140, 200),
            'motorcycle': (160, 120, 220),
            'bicycle': (140, 180, 200),
            'knife': (220, 100, 100),
            'scissors': (220, 140, 100),
            'backpack': (200, 160, 120),
            'handbag': (200, 140, 160),
            'suitcase': (180, 160, 140),
            'cell phone': (160, 140, 200),
            'bottle': (200, 200, 140),
            'cup': (180, 180, 120),
            'skateboard': (140, 200, 120),
            'skis': (120, 200, 160),
        }
        
        self.status_colors = {
            'optimal': (120, 200, 140),
            'good': (140, 200, 120),
            'moderate': (200, 180, 120),
            'poor': (200, 140, 120),
            'critical': (200, 120, 120),
            'abnormal_pulse': (220, 100, 100),
        }
        
        self.ui_colors = {
            'border_normal': (100, 100, 110),
            'border_active': (120, 160, 200),
            'border_warning': (200, 160, 100),
            'border_danger': (200, 120, 120),
            'confidence_bg': (55, 55, 60),
            'confidence_border': (140, 140, 150),
        }

class ThreatLevelScheme(ColorScheme):
    """Threat level color coding"""
    
    def __init__(self):
        super().__init__("Threat Level System")
        
        # Military-inspired text colors
        self.text_colors = {
            'normal': (200, 255, 200),      # Light green (all clear)
            'info': (200, 200, 255),        # Light blue (info)
            'success': (100, 255, 100),     # Green (secure)
            'warning': (255, 255, 100),     # Yellow (elevated)
            'danger': (255, 150, 0),        # Orange (high)
            'critical': (255, 0, 0),        # Red (severe)
            'header': (255, 255, 255),      # White
        }
        
        # Threat level backgrounds
        self.background_colors = {
            'panel_normal': (20, 40, 20),       # Dark green
            'panel_info': (20, 20, 40),         # Dark blue
            'panel_success': (10, 50, 10),      # Darker green
            'panel_warning': (50, 50, 10),      # Dark yellow
            'panel_danger': (50, 30, 10),       # Dark orange
            'overlay_flash': (100, 0, 0),       # Bright red flash
        }
        
        # Threat-based object classification
        self.class_colors = {
            # CRITICAL THREATS - Red family
            'knife': (255, 0, 0),              # Pure red
            'scissors': (255, 50, 0),          # Red-orange
            'gun': (200, 0, 0),                # Dark red
            
            # HIGH INTEREST - Orange family
            'person': (255, 150, 0),           # Orange (persons of interest)
            
            # VEHICLES - Yellow to orange (potential threats)
            'car': (255, 200, 0),              # Yellow-orange
            'truck': (255, 150, 0),            # Orange
            'motorcycle': (255, 100, 0),       # Red-orange
            'bicycle': (255, 255, 0),          # Yellow
            
            # SUSPICIOUS ITEMS - Yellow family
            'backpack': (255, 255, 100),       # Light yellow
            'suitcase': (255, 200, 100),       # Yellow-orange
            'handbag': (200, 200, 100),        # Muted yellow
            
            # LOW THREAT - Green family
            'bottle': (150, 255, 150),         # Light green
            'cup': (100, 255, 100),            # Green
            'cell phone': (100, 200, 100),     # Muted green
            'skateboard': (50, 255, 50),       # Bright green
            'skis': (100, 255, 200),           # Green-cyan
        }
        
        # Threat level status colors
        self.status_colors = {
            'optimal': (100, 255, 100),        # Green (secure)
            'good': (150, 255, 100),           # Light green (low)
            'moderate': (255, 255, 100),       # Yellow (elevated)
            'poor': (255, 150, 0),             # Orange (high)
            'critical': (255, 0, 0),           # Red (severe)
            'abnormal_pulse': (255, 50, 50),   # Bright red pulse
        }
        
        # Military-style UI colors
        self.ui_colors = {
            'border_normal': (100, 150, 100),     # Green border
            'border_active': (255, 255, 100),     # Yellow border
            'border_warning': (255, 150, 0),      # Orange border
            'border_danger': (255, 0, 0),         # Red border
            'confidence_bg': (40, 40, 40),        # Dark background
            'confidence_border': (150, 150, 150), # Gray border
        }

class AccessibilityFirstScheme(ColorScheme):
    """Colorblind-friendly scheme with pattern support"""
    
    def __init__(self):
        super().__init__("Accessibility First")
        
        # Colorblind-safe text colors
        self.text_colors = {
            'normal': (255, 255, 255),      # White
            'info': (100, 200, 255),        # Blue (safe for all)
            'success': (0, 150, 0),         # Dark green (safe)
            'warning': (255, 140, 0),       # Orange (distinguishable)
            'danger': (200, 0, 0),          # Dark red (safe)
            'critical': (150, 0, 150),      # Purple (alternative to red)
            'header': (255, 255, 255),      # White
        }
        
        # Accessible backgrounds
        self.background_colors = {
            'panel_normal': (40, 40, 40),       # Dark gray
            'panel_info': (20, 40, 80),         # Dark blue
            'panel_success': (0, 60, 0),        # Dark green
            'panel_warning': (80, 50, 0),       # Dark orange
            'panel_danger': (80, 0, 0),         # Dark red
            'overlay_flash': (100, 0, 100),     # Purple flash (alternative)
        }
        
        # Colorblind-safe object colors with high contrast
        self.class_colors = {
            'person': (0, 150, 150),           # Teal (safe)
            'car': (0, 100, 200),              # Blue
            'truck': (0, 50, 150),             # Dark blue
            'motorcycle': (100, 0, 150),       # Purple
            'bicycle': (0, 200, 200),          # Cyan
            'knife': (200, 0, 0),              # Red
            'scissors': (150, 75, 0),          # Brown-orange
            'backpack': (150, 150, 0),         # Olive
            'handbag': (150, 0, 150),          # Purple
            'suitcase': (100, 100, 0),         # Dark yellow
            'cell phone': (75, 0, 150),        # Dark purple
            'bottle': (200, 200, 0),           # Yellow
            'cup': (150, 150, 0),              # Olive
            'skateboard': (0, 150, 0),         # Green
            'skis': (0, 150, 100),             # Teal-green
        }
        
        # Accessible status colors
        self.status_colors = {
            'optimal': (0, 150, 0),            # Green
            'good': (50, 150, 0),              # Yellow-green
            'moderate': (150, 150, 0),         # Yellow
            'poor': (150, 75, 0),              # Orange
            'critical': (150, 0, 0),           # Red
            'abnormal_pulse': (150, 0, 150),   # Purple pulse
        }
        
        # Accessible UI colors
        self.ui_colors = {
            'border_normal': (150, 150, 150),     # Gray
            'border_active': (100, 150, 255),     # Light blue
            'border_warning': (200, 150, 0),      # Orange
            'border_danger': (200, 0, 0),         # Red
            'confidence_bg': (50, 50, 50),        # Dark gray
            'confidence_border': (200, 200, 200), # Light gray
        }
        
        # Pattern support for additional differentiation
        self.patterns = {
            'critical': 'dashed',      # Dashed lines for critical items
            'warning': 'dotted',       # Dotted lines for warnings
            'normal': 'solid',         # Solid lines for normal
        }

# Available color schemes
AVAILABLE_SCHEMES = {
    'semantic': SemanticColorScheme(),
    'high_contrast': HighContrastScheme(),
    'professional': ProfessionalDashboardScheme(),
    'threat_level': ThreatLevelScheme(),
    'accessibility': AccessibilityFirstScheme(),
}

def get_color_scheme(scheme_name='semantic'):
    """Get a color scheme by name"""
    return AVAILABLE_SCHEMES.get(scheme_name, AVAILABLE_SCHEMES['semantic'])

def draw_pattern_line(img, pt1, pt2, color, thickness, pattern='solid'):
    """Draw line with different patterns for accessibility"""
    if pattern == 'solid':
        cv2.line(img, pt1, pt2, color, thickness)
    elif pattern == 'dashed':
        # Draw dashed line
        length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        dash_length = 10
        num_dashes = int(length / (dash_length * 2))
        
        for i in range(num_dashes):
            start_ratio = (i * 2 * dash_length) / length
            end_ratio = ((i * 2 + 1) * dash_length) / length
            
            if end_ratio > 1.0:
                end_ratio = 1.0
            
            start_pt = (
                int(pt1[0] + start_ratio * (pt2[0] - pt1[0])),
                int(pt1[1] + start_ratio * (pt2[1] - pt1[1]))
            )
            end_pt = (
                int(pt1[0] + end_ratio * (pt2[0] - pt1[0])),
                int(pt1[1] + end_ratio * (pt2[1] - pt1[1]))
            )
            
            cv2.line(img, start_pt, end_pt, color, thickness)
    
    elif pattern == 'dotted':
        # Draw dotted line
        length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        dot_spacing = 8
        num_dots = int(length / dot_spacing)
        
        for i in range(num_dots):
            ratio = (i * dot_spacing) / length
            if ratio > 1.0:
                break
                
            dot_pt = (
                int(pt1[0] + ratio * (pt2[0] - pt1[0])),
                int(pt1[1] + ratio * (pt2[1] - pt1[1]))
            )
            
            cv2.circle(img, dot_pt, thickness, color, -1)

def draw_pattern_rectangle(img, pt1, pt2, color, thickness, pattern='solid'):
    """Draw rectangle with different patterns"""
    if pattern == 'solid':
        cv2.rectangle(img, pt1, pt2, color, thickness)
    else:
        # Draw each side with pattern
        draw_pattern_line(img, pt1, (pt2[0], pt1[1]), color, thickness, pattern)  # Top
        draw_pattern_line(img, (pt2[0], pt1[1]), pt2, color, thickness, pattern)  # Right
        draw_pattern_line(img, pt2, (pt1[0], pt2[1]), color, thickness, pattern)  # Bottom
        draw_pattern_line(img, (pt1[0], pt2[1]), pt1, color, thickness, pattern)  # Left 