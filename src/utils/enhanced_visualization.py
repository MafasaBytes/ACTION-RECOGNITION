import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any
from collections import deque

# New UI Component Imports
from .ui_components.dashboard_layout import DashboardLayoutManager
from .ui_components.video_renderer import VideoPaneRenderer
from .ui_components.sidebar_panels import SystemStatusPanel, AnomalyListPanel, GlobalDetectionsPanel, DetectedActionsPanel
from .ui_components.action_history_panel import ActionHistoryPanel
# from .ui_components.action_log_panel import ActionLogPanel # Example for future panel
# from .ui_components.statistics_panel import StatisticsPanel # Example for future panel

# --- ColorScheme Class (remains here for now or can be moved) ---
class ColorScheme:
    def __init__(self, scheme_name: str = 'professional'):
        self.name = scheme_name
        self.background_colors: Dict[str, Tuple[int, int, int]] = {}
        self.text_colors: Dict[str, Tuple[int, int, int]] = {}
        self.ui_colors: Dict[str, Tuple[int, int, int]] = {}
        self.class_colors: Dict[str, Tuple[int, int, int]] = {}
        self.action_colors: Dict[str, Tuple[int, int, int]] = {}
        self.ui_params: Dict[str, Any] = {}
        self.high_risk_objects = {'knife', 'gun', 'weapon'} # Example, can be configured

        self._load_scheme()

    def _load_scheme(self):
        # Default Professional Scheme
        if self.name == 'professional':
            self.background_colors = {
                'dashboard': (30, 30, 30),
                'panel_normal': (45, 45, 45),
                'panel_accent': (60, 60, 60),
                'panel_warning': (120, 100, 40), # Dark Yellow/Orange
                'panel_danger': (130, 50, 50),   # Dark Red
            }
            self.text_colors = {
                'header': (220, 220, 220), 
                'header_accent': (200, 200, 160),
                'normal': (190, 190, 190),
                'highlight': (230, 230, 230),
                'info': (170, 200, 230), # Light Blueish
                'info_accent': (200, 220, 255),
                'warning': (255, 210, 130), # Lighter Orange
                'critical': (255, 170, 170), # Lighter Red
                'success': (170, 230, 170), # Light Green
            }
            self.ui_colors = {
                'border_normal': (70, 70, 70),
                'border_active': (100, 150, 200),
                'border_warning': (200, 150, 50),
                'border_danger': (220, 80, 80),
                'divider': (80, 80, 80),
                'video_bg': (15,15,15),
                'video_alert_overlay': (0,0,30), # Very Dark Red for video tint
                'label_bg_border': (120,120,120),
            }
            self.ui_params = {
                'video_alert_overlay_alpha': 0.12,
                'object_label_bg_alpha': 0.65,
                'anomaly_display_threshold': 0.3, # Min score to show in anomaly list
                'critical_threshold': 0.75 # Score above which an anomaly is critical
            }
        # Add other schemes like 'high_contrast', 'dark_mode' etc.
        else: # Fallback to a basic scheme (e.g., name='default')
            self._load_default_fallback()

        # Common class and action colors (can be overridden by schemes too)
        # These provide a baseline if not specified in a theme.
        base_class_colors = {
            'person': (0, 220, 0), 'car': (200, 0, 0), 'bicycle': (0, 0, 220), 
            'default': (200, 200, 200), 'knife': (255,0,0), 'gun': (255,0,0), 'weapon':(255,0,0)
        }
        self.class_colors = {**base_class_colors, **self.class_colors} # Scheme specific can override
        
        base_action_colors = {
            'normal': (0, 200, 0), 'suspicious': (255, 180, 0), 'critical': (220, 0, 0),
            'default': (200, 200, 200)
        }
        self.action_colors = {**base_action_colors, **self.action_colors}

    def _load_default_fallback(self):
        self.background_colors = {'dashboard': (20,20,20), 'panel_normal': (40,40,40), 'panel_accent':(55,55,55), 'panel_warning':(100,80,30), 'panel_danger':(100,40,40)}
        self.text_colors = {'header': (200,200,200), 'header_accent':(180,180,150), 'normal':(180,180,180), 'highlight':(220,220,220), 'info':(150,180,200), 'info_accent':(180,200,230), 'warning':(230,190,100), 'critical':(230,150,150), 'success':(150,200,150)}
        self.ui_colors = {'border_normal': (60,60,60), 'border_active': (80,130,180), 'border_warning':(180,130,40), 'border_danger':(200,70,70), 'divider':(70,70,70), 'video_bg':(10,10,10), 'video_alert_overlay': (0,0,25), 'label_bg_border':(100,100,100)}
        self.ui_params = {'video_alert_overlay_alpha': 0.1, 'object_label_bg_alpha': 0.6, 'anomaly_display_threshold': 0.3, 'critical_threshold': 0.75}

    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        return self.class_colors.get(class_name.lower(), self.class_colors['default'])

    def get_action_severity_color(self, score: float) -> Tuple[int, int, int]:
        # Implement consistent color coordination: Green → Good, Orange → In-between, Red → Bad
        if score > 0.7:  # Red → Bad (High threat)
            return (100, 100, 255)  # Red
        elif score > 0.4:  # Orange → In-between (Medium threat)
            return (100, 165, 255)  # Orange
        else:  # Green → Good (Low/No threat)
            return (100, 255, 100)  # Green

# --- EnhancedVisualizer Class (Refactored) ---
class EnhancedVisualizer:
    def __init__(self, color_scheme_name: str = 'professional', 
                 dashboard_width: int = 1400, dashboard_height: int = 900):
        
        self.color_scheme = ColorScheme(color_scheme_name)
        
        # Initialize Layout Manager
        self.layout_manager = DashboardLayoutManager(
            dashboard_width=dashboard_width, 
            dashboard_height=dashboard_height,
            color_scheme=self.color_scheme
        )
        
        # Initialize Video Pane Renderer
        self.video_renderer = VideoPaneRenderer(
            color_scheme=self.color_scheme,
            object_label_bg_alpha=self.color_scheme.ui_params.get('object_label_bg_alpha', 0.6)
        )
        
        # Initialize Sidebar Panels
        self.system_status_panel = SystemStatusPanel(
            "system_status", self.color_scheme, panel_height=195 # Adjusted height
        )
        self.anomaly_list_panel = AnomalyListPanel(
            "anomaly_list", self.color_scheme, panel_height=175, max_items=5 # Adjusted height and items
        )
        self.global_detections_panel = GlobalDetectionsPanel(
            "global_detections", self.color_scheme, panel_height=175, max_classes_to_show=4 # Adjusted height and items
        )
        self.detected_actions_panel = DetectedActionsPanel(
            "detected_actions", self.color_scheme, panel_height=200, max_items=3 # Adjusted height and items
        )
        self.action_history_panel = ActionHistoryPanel(
            "action_history", self.color_scheme, panel_height=200, max_items=3 # Adjusted height and items
        )
        # Add other panels here (e.g., action log, statistics)
        # self.action_log_panel = ActionLogPanel(...) 

        self.sidebar_panels = [
            # self.system_status_panel, # Moved to the bottom
            self.global_detections_panel, 
            self.detected_actions_panel, 
            self.anomaly_list_panel, 
            self.system_status_panel, # Moved to the bottom
            self.action_history_panel,
            # self.action_log_panel 
        ]

        self.frame_times = deque(maxlen=60)
        self.current_fps = 0.0
        self.object_count_history = deque(maxlen=30)
        self.log_interval = 1.0
        self.last_log_time = time.time()
        self._last_frame_time = None

    def _update_performance_metrics(self):
        current_time = time.perf_counter()
        if self._last_frame_time is not None:
            frame_time = current_time - self._last_frame_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 1:
                valid_frame_times = [ft for ft in self.frame_times if ft > 0]
                if valid_frame_times:
                    self.current_fps = len(valid_frame_times) / sum(valid_frame_times)
                else:
                    self.current_fps = 0
        self._last_frame_time = current_time
    
    def _get_displayable_anomalies(self, actions, anomaly_scores, tracks):
        """Prepares anomalies for display, linking them with tracks and actions."""
        display_anomalies = []
        track_info_map = {str(t.get('object_id', t.get('track_id'))): t for t in tracks}

        if not isinstance(anomaly_scores, list):
            anomaly_scores = [] 

        for i, score_data in enumerate(anomaly_scores):
            track_id_str = "N/A"
            action_type = "Unknown Action"
            score = 0
            class_name = "Unknown"

            if isinstance(score_data, dict):
                track_id = score_data.get('track_id')
                score = score_data.get('anomaly_score', 0) 
                action_type = score_data.get('action_name', 'Unknown Action')
                
                track_id_str = str(track_id) if track_id is not None else "N/A"

                if track_id_str != "N/A" and track_id_str != "-1" and track_id_str in track_info_map:
                    class_name = track_info_map[track_id_str].get('class_name', 'Object')
                elif track_id_str == "-1":
                    class_name = "Global"
            
            elif isinstance(score_data, (float, int)) : 
                score = score_data
                if i < len(actions) and isinstance(actions[i], dict):
                    action_detail = actions[i]
                    track_id_val = action_detail.get('track_id')
                    track_id_str = str(track_id_val) if track_id_val is not None else "N/A"
                    action_type = action_detail.get('action_name', action_type)
                    if track_id_str in track_info_map and track_id_str != "N/A":
                        class_name = track_info_map[track_id_str].get('class_name', class_name)
            
            if score > self.color_scheme.ui_params.get('anomaly_display_threshold', 0.3):
                display_anomalies.append({
                    'track_id': track_id_str,
                    'action_type': action_type,
                    'score': score,
                    'class_name': class_name,
                    'is_critical': score > self.color_scheme.ui_params.get('critical_threshold', 0.75)
                })
        
        display_anomalies.sort(key=lambda x: x['score'], reverse=True)
        return display_anomalies

    def draw_enhanced_results(self, frame: np.ndarray, tracks: List[Dict], 
                                actions: List[Any], anomaly_scores: List[Any], 
                                action_history_manager=None) -> np.ndarray:
        self._update_performance_metrics() 

        dashboard_canvas = self.layout_manager.create_dashboard_canvas()
        dashboard_canvas[:] = self.color_scheme.background_colors['dashboard']
        
        active_anomalies = self._get_displayable_anomalies(actions, anomaly_scores, tracks)
        is_system_alert_active = any(a['is_critical'] for a in active_anomalies)
        has_warnings = any(self.color_scheme.ui_params.get('anomaly_display_threshold',0.3) < a['score'] <= self.color_scheme.ui_params.get('critical_threshold', 0.75) for a in active_anomalies)

        self.layout_manager.draw_base_layout(dashboard_canvas, is_system_alert_active, has_warnings)

        video_area_x, video_area_y, video_area_w, video_area_h = self.layout_manager.get_video_area_rect()
        
        all_actions_map_for_video = {}
        if isinstance(anomaly_scores, list):
            for score_item in anomaly_scores:
                if isinstance(score_item, dict):
                    track_id = score_item.get('track_id')
                    if track_id is not None and track_id != -1 and track_id != "N/A":
                        all_actions_map_for_video[str(track_id)] = {
                            'action_type': score_item.get('action_name', 'Unknown Action'),
                            'score': score_item.get('anomaly_score', 0)
                        }
        
        enriched_tracks_for_video = []
        for track_item in tracks:
            enriched_track = track_item.copy()
            track_id_str = str(track_item.get('object_id', track_item.get('track_id', '')))
            
            if track_id_str and track_id_str in all_actions_map_for_video:
                action_data = all_actions_map_for_video[track_id_str]
                enriched_track['action_type'] = action_data['action_type']
                enriched_track['action_score'] = action_data['score']
            enriched_tracks_for_video.append(enriched_track)
            
        rendered_video_pane, actual_video_rect_in_pane = self.video_renderer.render_video_pane(
            frame, enriched_tracks_for_video, video_area_w, video_area_h, is_system_alert_active
        )
        
        pane_h, pane_w = rendered_video_pane.shape[:2]
        dashboard_canvas[video_area_y : video_area_y + pane_h, 
                         video_area_x : video_area_x + pane_w] = rendered_video_pane
        
        abs_video_x = video_area_x + actual_video_rect_in_pane[0]
        abs_video_y = video_area_y + actual_video_rect_in_pane[1]
        self.layout_manager.draw_video_border(dashboard_canvas, 
                                              (abs_video_x, abs_video_y, actual_video_rect_in_pane[2], actual_video_rect_in_pane[3]),
                                              is_system_alert_active, has_warnings)

        sidebar_x, sidebar_y, sidebar_w, sidebar_h = self.layout_manager.get_sidebar_rect()
        current_y_offset = 0
        
        self.object_count_history.append(len(tracks))
        avg_object_count = sum(self.object_count_history) / len(self.object_count_history) if self.object_count_history else 0
        detection_count = sum(1 for t in tracks if t.get('is_new', True)) 

        system_status = "NORMAL"
        if is_system_alert_active:
            system_status = "CRITICAL"
        elif has_warnings:
            system_status = "WARNING"
        
        low_fps_threshold = 15 
        if self.current_fps > 0 and self.current_fps < low_fps_threshold and system_status == "NORMAL":
            system_status = "SLOW"
        elif self.current_fps == 0 and system_status == "NORMAL":
            system_status = "INIT"

        status_data = {
            'width': sidebar_w,
            'avg_fps': f"{self.current_fps:.1f}",
            'object_count': len(tracks), 
            'detection_count': detection_count,
            'active_alerts': len(active_anomalies),
            'cpu_usage': -1,
            'ram_usage': -1,
            'system_status': system_status
        }
        
        anomaly_panel_data = {
            'width': sidebar_w,
            'anomalies': active_anomalies
        }

        class_counts = {}
        for track in tracks:
            class_name = track.get('class_name', 'Unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        global_detections_data = {
            'width': sidebar_w,
            'class_counts': class_counts,
            'total_objects': len(tracks)
        }

        all_actions_log = []
        raw_actions_input = actions
        
        track_id_to_class_name = {str(t.get('object_id', t.get('track_id'))): t.get('class_name', 'Unknown') for t in enriched_tracks_for_video}

        for action_data_item in anomaly_scores:
            if isinstance(action_data_item, dict):
                track_id_val = action_data_item.get('track_id', "N/A")
                track_id_str = str(track_id_val)

                action_type = action_data_item.get('action_name', 'Unknown Action')
                score = action_data_item.get('anomaly_score', 0.0)
                warning_level = action_data_item.get('warning_level', 0)
                
                if track_id_val == -1:
                    class_name = "Global"
                else:
                    class_name = track_id_to_class_name.get(track_id_str, 'Unknown')
                
                all_actions_log.append({
                    'track_id': track_id_str,
                    'action_type': action_type,
                    'score': score,
                    'class_name': class_name,
                    'warning_level': warning_level
                })
        
        for raw_action in actions:
            if isinstance(raw_action, dict):
                track_id_val = raw_action.get('track_id')
                track_id_str = str(track_id_val) if track_id_val is not None else "N/A"
                action_type = raw_action.get('action_name', 'Unknown Action')
                confidence = raw_action.get('action_confidence', 0.0)
                
                existing_action = next((a for a in all_actions_log 
                                      if a['track_id'] == track_id_str and a['action_type'] == action_type), None)
                
                if not existing_action:
                    if track_id_val == -1:
                        class_name = "Global"
                    else:
                        class_name = track_id_to_class_name.get(track_id_str, 'Unknown')
                    
                    all_actions_log.append({
                        'track_id': track_id_str,
                        'action_type': action_type,
                        'score': confidence * 0.1,
                        'class_name': class_name,
                        'warning_level': 0
                    })
        
        all_actions_log.sort(key=lambda x: x['score'], reverse=True)

        detected_actions_data = {
            'width': sidebar_w,
            'actions_log': all_actions_log
        }

        action_history_data = {
            'width': sidebar_w,
            'action_history': [],
            'statistics': {},
            'strategy': 'basic'
        }
        
        if action_history_manager:
            action_history_data['action_history'] = action_history_manager.get_display_actions()
            action_history_data['statistics'] = action_history_manager.get_statistics()
            action_history_data['strategy'] = getattr(action_history_manager, 'current_strategy', 'mixed')

        panel_data_map = {
            "system_status": status_data,
            "anomaly_list": anomaly_panel_data,
            "global_detections": global_detections_data, 
            "detected_actions": detected_actions_data,
            "action_history": action_history_data,
        }

        for panel in self.sidebar_panels:
            panel_id = panel.panel_id
            current_panel_data = panel_data_map.get(panel_id, {'width': sidebar_w})
            panel_render_height = panel.get_required_height(current_panel_data)
            
            if current_y_offset + panel_render_height <= sidebar_h:
                panel_surface = panel.render(current_panel_data)
                ps_h, ps_w = panel_surface.shape[:2]
                dashboard_canvas[
                    sidebar_y + current_y_offset : sidebar_y + current_y_offset + ps_h, 
                    sidebar_x : sidebar_x + ps_w
                ] = panel_surface
                current_y_offset += ps_h + self.layout_manager.panel_margin
            else:
                pass 
        
        return dashboard_canvas

# --- Example Usage (for testing EnhancedVisualizer directly) ---
if __name__ == '__main__':
    test_visualizer = EnhancedVisualizer(color_scheme_name='professional') 
    
    sample_video_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(sample_video_frame, "VIDEO FEED SIMULATION", (400, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)

    # Initialize psutil here for the test case if needed, or rely on panel's internal call
    # For testing, the panel will call psutil.cpu_percent() and psutil.virtual_memory().percent directly
    # if cpu_usage and ram_usage are not in the passed data.

    sample_tracks_data = [
        {'bbox': [100, 150, 220, 350], 'object_id': 'T1', 'class_name': 'person', 'confidence': 0.92, 'is_new': True},
        {'bbox': [300, 200, 400, 450], 'object_id': 'T2', 'class_name': 'knife', 'confidence': 0.85, 'is_new': True},
        {'bbox': [500, 100, 580, 280], 'object_id': 'T3', 'class_name': 'person', 'confidence': 0.78, 'is_new': False},
    ]
    sample_actions_data = [
        {'track_id': 'T1', 'action_type': 'Walking', 'score': 0.4},
        {'track_id': 'T2', 'action_type': 'Brandishing Weapon', 'score': 0.90},
        {'track_id': 'T3', 'action_type': 'Standing Still', 'score': 0.25},
        {'action_type': 'Global Anomaly - Too Dark', 'score': 0.6}
    ]
    sample_anomaly_scores_data = [
        {'track_id': 'T1', 'score': 0.4, 'action_type': 'Walking'},
        {'track_id': 'T2', 'score': 0.90, 'action_type': 'Brandishing Weapon'},
        {'track_id': 'T3', 'score': 0.25, 'action_type': 'Standing Still'},
        {'score': 0.6, 'action_type': 'Global Anomaly - Too Dark'} 
    ]

    frame_counter = 0
    test_visualizer._last_frame_time = time.perf_counter() # Initialize for first FPS calc

    while True:
        frame_counter += 1
        time.sleep(1/30) # Simulate ~30 FPS processing delay

        if frame_counter % 60 == 0: 
            sample_tracks_data[0]['bbox'][0] = (sample_tracks_data[0]['bbox'][0] + 20) % 800
            new_score_person1 = np.random.uniform(0.1, 0.5)
            new_score_knife = np.random.uniform(0.6, 0.95)
            sample_actions_data[0]['score'] = new_score_person1
            sample_anomaly_scores_data[0]['score'] = new_score_person1
            sample_actions_data[1]['score'] = new_score_knife
            sample_anomaly_scores_data[1]['score'] = new_score_knife
            if np.random.rand() > 0.7:
                 new_anomaly = {'score': np.random.uniform(0.4,0.8), 'action_type':'New Random Event', 'track_id':f'RX{frame_counter}'}
                 sample_anomaly_scores_data.append(new_anomaly)
                 if len(sample_anomaly_scores_data) > 6:
                     sample_anomaly_scores_data.pop(3) 

        current_frame_sim = sample_video_frame.copy()
        cv2.circle(current_frame_sim, (100 + (frame_counter*2 % 1000), 100), 20, (0,200,0), -1)
        
        dashboard_output_frame = test_visualizer.draw_enhanced_results(
            current_frame_sim, 
            sample_tracks_data, 
            sample_actions_data, 
            sample_anomaly_scores_data
        )
        
        cv2.imshow("Refactored Enhanced Dashboard", dashboard_output_frame)
        
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'): 
            current_scheme_name = test_visualizer.color_scheme.name
            new_scheme = 'default' if current_scheme_name == 'professional' else 'professional'
            print(f"Switching to color scheme: {new_scheme}")
            # Re-init visualizer for new scheme. Dependent objects get new scheme too.
            dashboard_width = test_visualizer.layout_manager.dashboard_width
            dashboard_height = test_visualizer.layout_manager.dashboard_height
            test_visualizer = EnhancedVisualizer(color_scheme_name=new_scheme, dashboard_width=dashboard_width, dashboard_height=dashboard_height)
            test_visualizer._last_frame_time = time.perf_counter() # Re-init for FPS

    cv2.destroyAllWindows() 