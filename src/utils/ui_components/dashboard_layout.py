import cv2
import numpy as np
from typing import Tuple, Dict
from datetime import datetime

class DashboardLayoutManager:
    """Manages the overall dashboard canvas, dimensions, and layout zones."""

    def __init__(self, dashboard_width: int = 1400, dashboard_height: int = 900,
                 video_width_ratio: float = 0.70, sidebar_width_ratio: float = 0.28,
                 top_bar_height: int = 60, bottom_margin: int = 20,
                 panel_margin: int = 15, color_scheme=None):
        self.dashboard_width = dashboard_width
        self.dashboard_height = dashboard_height
        self.video_width_ratio = video_width_ratio
        self.sidebar_width_ratio = sidebar_width_ratio
        self.top_bar_height = top_bar_height
        self.bottom_margin = bottom_margin
        self.panel_margin = panel_margin
        self.color_scheme = color_scheme

        self.video_area_width: int = 0
        self.sidebar_area_width: int = 0
        self.video_area_height: int = 0
        self.sidebar_area_x: int = 0
        self.sidebar_area_y: int = 0
        self.sidebar_content_height: int = 0

        self._calculate_layout_dimensions()

    def _calculate_layout_dimensions(self):
        """Calculates the primary layout dimensions based on initial parameters."""
        self.video_area_width = int(self.dashboard_width * self.video_width_ratio)
        self.sidebar_area_width = int(self.dashboard_width * self.sidebar_width_ratio)
        self.video_area_height = self.dashboard_height - self.top_bar_height - self.bottom_margin
        
        self.sidebar_area_x = self.video_area_width + (self.dashboard_width - self.video_area_width - self.sidebar_area_width) // 2
        self.sidebar_area_y = self.top_bar_height + self.panel_margin
        self.sidebar_content_height = self.dashboard_height - self.top_bar_height - self.bottom_margin - (2 * self.panel_margin)

    def create_dashboard_canvas(self) -> np.ndarray:
        """Creates a new blank dashboard canvas."""
        return np.zeros((self.dashboard_height, self.dashboard_width, 3), dtype=np.uint8)

    def get_video_area_rect(self) -> Tuple[int, int, int, int]:
        """Returns the rectangle (x, y, width, height) for the video display area."""
        return 0, self.top_bar_height, self.video_area_width, self.video_area_height

    def get_sidebar_rect(self) -> Tuple[int, int, int, int]:
        """Returns the rectangle (x, y, width, height) for the sidebar area."""
        return self.sidebar_area_x, self.sidebar_area_y, self.sidebar_area_width, self.sidebar_content_height
        
    def get_title_bar_rect(self) -> Tuple[int, int, int, int]:
        """Returns the rectangle (x, y, width, height) for the title bar area."""
        return 0, 0, self.dashboard_width, self.top_bar_height

    def draw_base_layout(self, dashboard_canvas: np.ndarray, abnormal_detected: bool, has_warnings: bool):
        """Draws the base layout elements like title bar background and borders."""
        if self.color_scheme is None:
            title_bg_color = (50, 50, 50)
            title_text_color = (220, 220, 220)
            title_border_color = (100, 100, 100)
            status_text = "ANOMALY DETECTION SYSTEM - INITIALIZING"
        elif abnormal_detected or has_warnings:
            title_bg_color = self.color_scheme.background_colors['panel_danger']
            title_text_color = self.color_scheme.text_colors['critical']
            title_border_color = self.color_scheme.ui_colors['border_danger']
            status_text = "ANOMALY DETECTION SYSTEM - ALERT ACTIVE"
        else:
            title_bg_color = self.color_scheme.background_colors['panel_normal']
            title_text_color = self.color_scheme.text_colors['header']
            title_border_color = self.color_scheme.ui_colors['border_normal']
            status_text = "ANOMALY DETECTION SYSTEM - MONITORING"

        title_x, title_y, title_w, title_h = self.get_title_bar_rect()
        cv2.rectangle(dashboard_canvas, (title_x, title_y), (title_x + title_w, title_y + title_h), title_bg_color, -1)
        cv2.rectangle(dashboard_canvas, (title_x, title_y), (title_x + title_w, title_y + title_h), title_border_color, 2)

        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x_pos = (title_w - text_size[0]) // 2
        text_y_pos = title_y + int(title_h * 0.65)
        cv2.putText(dashboard_canvas, status_text, (text_x_pos, text_y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_text_color, 2, cv2.LINE_AA)

        timestamp_text = datetime.now().strftime("%a %H:%M:%S")
        timestamp_font_scale = 0.55
        timestamp_thickness = 1
        timestamp_color = self.color_scheme.text_colors.get('info', (200,200,100)) if self.color_scheme else (200,200,100)
        
        ts_text_size = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, timestamp_font_scale, timestamp_thickness)[0]
        ts_x_pos = title_w - ts_text_size[0] - 15
        ts_y_pos = title_y + int(title_h * 0.65)
        
        cv2.putText(dashboard_canvas, timestamp_text, (ts_x_pos, ts_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, timestamp_font_scale, timestamp_color, timestamp_thickness, cv2.LINE_AA)

    def draw_video_border(self, canvas: np.ndarray, video_rect: Tuple[int, int, int, int], abnormal_detected: bool, has_warnings: bool):
        """Draws a border around the actual video placement within the video area."""
        vx, vy, vw, vh = video_rect
        if self.color_scheme is None:
            border_color = (100,100,100)
            thickness = 2
        elif abnormal_detected or has_warnings:
            border_color = self.color_scheme.ui_colors.get('border_danger', (0,0,255))
            thickness = 4
        else:
            border_color = self.color_scheme.ui_colors.get('border_active', (200,200,200))
            thickness = 2
        
        cv2.rectangle(canvas, (vx - thickness//2, vy - thickness//2), 
                     (vx + vw + thickness//2, vy + vh + thickness//2), border_color, thickness)


if __name__ == '__main__':
    class MockColorScheme:
        def __init__(self):
            self.background_colors = {
                'panel_danger': (30, 30, 150),
                'panel_normal': (50, 50, 50),
            }
            self.text_colors = {
                'critical': (200, 200, 255),
                'header': (220, 220, 220),
                'info': (200, 200, 100)
            }
            self.ui_colors = {
                'border_danger': (0, 0, 255),
                'border_normal': (100, 100, 100),
                'border_active': (200, 200, 200)
            }

    mock_scheme = MockColorScheme()
    layout_manager = DashboardLayoutManager(color_scheme=mock_scheme)
    dashboard = layout_manager.create_dashboard_canvas()
    
    layout_manager.draw_base_layout(dashboard, abnormal_detected=False, has_warnings=False)
    
    example_video_x = (layout_manager.video_area_width - 640) // 2
    example_video_y = layout_manager.top_bar_height + (layout_manager.video_area_height - 480) // 2
    layout_manager.draw_video_border(dashboard, (example_video_x, example_video_y, 640, 480), abnormal_detected=False, has_warnings=False)
    
    cv2.imshow("Dashboard Layout Test (Normal)", dashboard)

    dashboard_alert = layout_manager.create_dashboard_canvas()
    layout_manager.draw_base_layout(dashboard_alert, abnormal_detected=True, has_warnings=True)
    layout_manager.draw_video_border(dashboard_alert, (example_video_x, example_video_y, 640, 480), abnormal_detected=True, has_warnings=True)
    cv2.imshow("Dashboard Layout Test (Alert)", dashboard_alert)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows() 