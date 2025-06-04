"""
Unified Color Management System
Consolidates color schemes from color_schemes.py and enhanced_visualization.py
"""

import colorsys
from typing import Dict, Tuple, Any
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass 
class ColorPalette:
    """Standardized color palette with consistent naming"""
    # Base colors using consistent coordination: Green → Good, Orange → Medium, Red → Bad
    GREEN_GOOD: Tuple[int, int, int] = (100, 255, 100)        # Good performance/low risk
    ORANGE_MEDIUM: Tuple[int, int, int] = (100, 165, 255)     # Medium performance/moderate risk  
    RED_BAD: Tuple[int, int, int] = (100, 100, 255)           # Poor performance/high risk
    WHITE_NEUTRAL: Tuple[int, int, int] = (220, 220, 220)     # Neutral information
    GRAY_DISABLED: Tuple[int, int, int] = (150, 150, 150)     # Disabled/inactive elements
    
    # Extended palette for UI elements
    DARK_PANEL: Tuple[int, int, int] = (45, 45, 45)           # Panel backgrounds
    DARK_BG: Tuple[int, int, int] = (30, 30, 30)              # Main background
    BLUE_INFO: Tuple[int, int, int] = (120, 180, 255)         # Information
    YELLOW_WARNING: Tuple[int, int, int] = (255, 180, 80)     # Warnings

class UnifiedColorScheme:
    """
    Unified color scheme that consolidates functionality from both 
    enhanced_visualization.py and color_schemes.py
    """
    
    def __init__(self, scheme_name: str = 'professional'):
        self.name = scheme_name
        self.palette = ColorPalette()
        
        # Core color categories
        self.background_colors: Dict[str, Tuple[int, int, int]] = {}
        self.text_colors: Dict[str, Tuple[int, int, int]] = {}
        self.ui_colors: Dict[str, Tuple[int, int, int]] = {}
        self.class_colors: Dict[str, Tuple[int, int, int]] = {}
        self.ui_params: Dict[str, Any] = {}
        
        # Risk-based object classification
        self.high_risk_objects = {'knife', 'gun', 'weapon', 'scissors'}
        
        self._load_scheme()
        
    def _load_scheme(self):
        """Load the specified color scheme"""
        if self.name == 'professional':
            self._load_professional_scheme()
        elif self.name == 'high_contrast':
            self._load_high_contrast_scheme()
        else:
            self._load_professional_scheme()  # Default fallback
            
    def _load_professional_scheme(self):
        """Professional monitoring dashboard aesthetic"""
        p = self.palette
        
        self.background_colors = {
            'dashboard': (30, 30, 30),
            'panel_normal': (45, 45, 45),
            'panel_accent': (60, 60, 60),
            'panel_warning': (120, 100, 40),
            'panel_danger': (130, 50, 50),
        }
        
        self.text_colors = {
            'header': (220, 220, 220), 
            'header_accent': (200, 200, 160),
            'normal': (190, 190, 190),
            'highlight': (230, 230, 230),
            'info': (170, 200, 230),
            'info_accent': (200, 220, 255),
            'warning': (255, 210, 130),
            'critical': (255, 170, 170),
            'success': (170, 230, 170),
        }
        
        self.ui_colors = {
            'border_normal': (70, 70, 70),
            'border_active': (100, 150, 200),
            'border_warning': (200, 150, 50),
            'border_danger': (220, 80, 80),
            'divider': (80, 80, 80),
            'video_bg': (15, 15, 15),
            'video_alert_overlay': (0, 0, 30),
            'label_bg_border': (120, 120, 120),
        }
        
        self.ui_params = {
            'video_alert_overlay_alpha': 0.12,
            'object_label_bg_alpha': 0.65,
            'anomaly_display_threshold': 0.3,
            'critical_threshold': 0.75
        }
        
        # Standard class colors
        self.class_colors = {
            # High risk objects - Red variants
            'knife': p.RED_BAD,
            'gun': (255, 0, 0),            # Pure red for maximum danger
            'weapon': p.RED_BAD,
            'scissors': (220, 20, 60),     # Dark red
            
            # People - Green (good/safe)
            'person': p.GREEN_GOOD,
            
            # Vehicles - Blue family
            'car': (0, 0, 255),
            'truck': (0, 100, 255),
            'motorcycle': (138, 43, 226),
            'bicycle': (75, 0, 130),
            
            # Personal items - Orange/Yellow family
            'backpack': p.ORANGE_MEDIUM,
            'handbag': (255, 150, 150),
            'suitcase': (255, 200, 150),
            'bottle': (255, 255, 100),
            'cup': (255, 200, 50),
            
            # Electronics - Purple family
            'cell phone': (200, 100, 255),
            'laptop': (150, 50, 255),
            
            # Default
            'default': p.WHITE_NEUTRAL,
        }
        
    def _load_high_contrast_scheme(self):
        """High contrast scheme for accessibility"""
        self.background_colors = {
            'dashboard': (0, 0, 0),
            'panel_normal': (0, 0, 0),
            'panel_accent': (0, 0, 80),
            'panel_warning': (80, 80, 0),
            'panel_danger': (80, 0, 0),
        }
        
        self.text_colors = {
            'header': (255, 255, 255),
            'header_accent': (255, 255, 255),
            'normal': (255, 255, 255),
            'highlight': (255, 255, 255),
            'info': (0, 255, 255),
            'warning': (255, 255, 0),
            'critical': (255, 0, 0),
            'success': (0, 255, 0),
        }
        
        self.ui_colors = {
            'border_normal': (255, 255, 255),
            'border_active': (0, 255, 255),
            'border_warning': (255, 255, 0),
            'border_danger': (255, 0, 0),
            'divider': (255, 255, 255),
            'video_bg': (0, 0, 0),
            'video_alert_overlay': (100, 0, 0),
        }
        
        # High contrast class colors
        self.class_colors = {
            'person': (0, 255, 0),
            'knife': (255, 0, 0),
            'gun': (255, 0, 0),
            'car': (0, 0, 255),
            'default': (255, 255, 255),
        }
        
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for object class with fallback generation"""
        color = self.class_colors.get(class_name.lower())
        if color is not None:
            return color
            
        # Generate consistent fallback color using hash
        return self._generate_fallback_color(class_name)
        
    def _generate_fallback_color(self, class_name: str) -> Tuple[int, int, int]:
        """Generate consistent color for unknown classes"""
        hash_val = hash(class_name) % 360
        rgb = colorsys.hsv_to_rgb(hash_val / 360.0, 0.6, 0.8)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        
    def get_action_severity_color(self, score: float) -> Tuple[int, int, int]:
        """
        Get color based on action severity using consistent coordination:
        Green → Good (low threat), Orange → Medium, Red → Bad (high threat)
        """
        if score > 0.7:  # Red → Bad (High threat)
            return self.palette.RED_BAD
        elif score > 0.4:  # Orange → In-between (Medium threat)
            return self.palette.ORANGE_MEDIUM
        else:  # Green → Good (Low/No threat)
            return self.palette.GREEN_GOOD
            
    def get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Get color based on confidence level using consistent coordination:
        Green → Good (high confidence), Orange → Medium, Red → Bad (low confidence)
        """
        if confidence > 0.7:  # Green → Good (High confidence)
            return self.palette.GREEN_GOOD
        elif confidence > 0.4:  # Orange → In-between (Medium confidence)
            return self.palette.ORANGE_MEDIUM
        else:  # Red → Bad (Low confidence)
            return self.palette.RED_BAD
            
    def get_performance_color(self, percentage: float) -> Tuple[int, int, int]:
        """
        Get color based on performance percentage using consistent coordination:
        Green → Good (low usage), Orange → Medium, Red → Bad (high usage)
        """
        if percentage < 60:  # Green → Good performance
            return self.palette.GREEN_GOOD
        elif percentage < 80:  # Orange → In-between performance
            return self.palette.ORANGE_MEDIUM
        else:  # Red → Bad performance (high usage)
            return self.palette.RED_BAD
            
    def get_status_color(self, status: str) -> Tuple[int, int, int]:
        """Get color for system status"""
        status_map = {
            "NORMAL": self.palette.GREEN_GOOD,      # Green → Good
            "WARNING": self.palette.ORANGE_MEDIUM,  # Orange → In-between  
            "CRITICAL": self.palette.RED_BAD,       # Red → Bad
            "SLOW": self.palette.ORANGE_MEDIUM,     # Orange → In-between
            "INIT": self.palette.GRAY_DISABLED      # Gray → Neutral
        }
        return status_map.get(status.upper(), self.palette.WHITE_NEUTRAL)

class ColorManager:
    """
    Central color management service for the entire application
    Replaces multiple color handling systems with unified approach
    """
    
    def __init__(self, scheme_name: str = 'professional'):
        self.current_scheme = UnifiedColorScheme(scheme_name)
        logger.info(f"ColorManager initialized with scheme: {scheme_name}")
        
    def set_scheme(self, scheme_name: str):
        """Change the color scheme"""
        self.current_scheme = UnifiedColorScheme(scheme_name)
        logger.info(f"Color scheme changed to: {scheme_name}")
        
    def get_scheme(self) -> UnifiedColorScheme:
        """Get the current color scheme"""
        return self.current_scheme
        
    def get_available_schemes(self) -> list:
        """Get list of available color schemes"""
        return ['professional', 'high_contrast']

# Global color manager instance
_color_manager: ColorManager = None

def get_color_manager(scheme_name: str = 'professional') -> ColorManager:
    """Get global color manager instance"""
    global _color_manager
    if _color_manager is None:
        _color_manager = ColorManager(scheme_name)
    return _color_manager

def get_color_scheme() -> UnifiedColorScheme:
    """Get the current color scheme"""
    return get_color_manager().get_scheme() 