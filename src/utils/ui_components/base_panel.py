import cv2
import numpy as np
from typing import List, Tuple, Any, Dict

class BasePanel:
    """Base class for all sidebar UI panels."""

    def __init__(self, panel_id: str, color_scheme, 
                 font_scale: float = 0.5, font_thickness: int = 1, 
                 header_font_scale: float = 0.6, header_thickness: int = 2,
                 line_height: int = 22, panel_padding: int = 10):
        self.panel_id = panel_id
        self.color_scheme = color_scheme
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.header_font_scale = header_font_scale
        self.header_thickness = header_thickness
        self.line_height = line_height
        self.panel_padding = panel_padding

    def _draw_panel_background(self, surface: np.ndarray, panel_type: str = 'panel_normal'):
        """Draws the standard background and border for the panel."""
        if self.color_scheme:
            bg_color = self.color_scheme.background_colors.get(panel_type, self.color_scheme.background_colors['panel_normal'])
            border_color = self.color_scheme.ui_colors.get('border_normal', (100, 100, 100))
        else: # Fallback
            bg_color = (40,40,40)
            border_color = (80,80,80)
        
        cv2.rectangle(surface, (0, 0), (surface.shape[1], surface.shape[0]), bg_color, -1)
        cv2.rectangle(surface, (0, 0), (surface.shape[1], surface.shape[0]), border_color, 2)

    def _truncate_text(self, text: str, max_width: int) -> str:
        """Truncates text if it exceeds the max_width in pixels."""
        text_width = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0][0]
        if text_width <= max_width:
            return text

        # Estimate average char width and truncate (simple method)
        avg_char_width = text_width / len(text) if len(text) > 0 else 10
        max_chars = int(max_width / avg_char_width) - 3 # -3 for "..."
        return text[:max_chars] + "..." if max_chars > 0 else "..."

    def render(self, data: Any) -> np.ndarray:
        """Abstract method to render the panel content. 
        Subclasses must implement this and return a cv2.Mat (surface) for the panel.
        The 'data' argument will vary depending on the panel type.
        The returned surface should be of the panel's fixed height and width.
        """
        raise NotImplementedError("Each panel must implement its own render method.")

    def get_required_height(self, data: Any) -> int:
        """ Estimates or returns the fixed height required by this panel. """
        # Default implementation, subclasses should override if height is dynamic or fixed differently.
        # For many panels, this might be a fixed value set during init.
        return 100 # Default placeholder


if __name__ == '__main__':
    # Mock color scheme for testing BasePanel
    class MockColorScheme:
        def __init__(self):
            self.background_colors = {'panel_normal': (50, 50, 50)}
            self.ui_colors = {'border_normal': (100, 100, 100)}

    mock_scheme = MockColorScheme()
    base_panel = BasePanel("test_panel", mock_scheme)
    
    # Test background drawing
    panel_surface = np.zeros((150, 300, 3), dtype=np.uint8) # Example surface
    base_panel._draw_panel_background(panel_surface)
    cv2.imshow("Base Panel Background Test", panel_surface)
    
    # Test text truncation
    truncated = base_panel._truncate_text("This is a very long string that should be truncated", 200)
    print(f"Truncated text: {truncated}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows() 