import cv2
import time
from typing import Dict, Callable, Any

class ControlsManager:
    """Manages keyboard shortcuts and controls for the application"""
    
    def __init__(self):
        self.controls = {
            ord('q'): self.quit_application,
            ord('h'): self.show_help,
            ord('p'): self.toggle_pause,
            ord('r'): self.reset_stats,
            ord('s'): self.take_screenshot,
            ord('a'): self.toggle_audio,
            ord('f'): self.toggle_fps_display,
            ord('d'): self.toggle_debug_mode,
            ord('m'): self.mute_audio,
            27: self.quit_application,  # ESC key
        }
        
        self.state = {
            'paused': False,
            'audio_enabled': True,
            'fps_display': True,
            'debug_mode': False,
            'audio_muted': False,
            'show_help': False,
            'screenshot_count': 0,
            'last_screenshot_time': 0
        }
        
        self.help_text = [
            "=== COMPUTER VISION SYSTEM CONTROLS ===",
            "",
            "[Q] / [ESC] - Quit Application",
            "[H] - Toggle Help Display",
            "[P] - Pause/Resume Processing",
            "[R] - Reset Statistics",
            "[S] - Take Screenshot",
            "[A] - Toggle Audio Alerts",
            "[M] - Mute/Unmute Audio",
            "[F] - Toggle FPS Display",
            "[D] - Toggle Debug Mode",
            "",
            "=== STATUS INDICATORS ===",
            "Green Boxes: Normal Objects",
            "Red Screen Flash: Abnormal Behavior",
            "Audio Alerts: Different levels for severity",
            "",
            "Press [H] again to hide this help"
        ]
    
    def process_key(self, key: int) -> Dict[str, Any]:
        """Process keyboard input and return action results"""
        if key == -1:
            return {'action': 'none'}
            
        if key in self.controls:
            return self.controls[key]()
        
        return {'action': 'unknown', 'key': key}
    
    def quit_application(self) -> Dict[str, Any]:
        """Handle quit application"""
        return {'action': 'quit', 'message': 'Application quit requested'}
    
    def show_help(self) -> Dict[str, Any]:
        """Toggle help display"""
        self.state['show_help'] = not self.state['show_help']
        return {
            'action': 'toggle_help',
            'show_help': self.state['show_help'],
            'message': 'Help display toggled'
        }
    
    def toggle_pause(self) -> Dict[str, Any]:
        """Toggle pause/resume"""
        self.state['paused'] = not self.state['paused']
        status = "paused" if self.state['paused'] else "resumed"
        return {
            'action': 'pause',
            'paused': self.state['paused'],
            'message': f'Processing {status}'
        }
    
    def reset_stats(self) -> Dict[str, Any]:
        """Reset statistics"""
        return {
            'action': 'reset_stats',
            'message': 'Statistics reset'
        }
    
    def take_screenshot(self) -> Dict[str, Any]:
        """Take screenshot with rate limiting"""
        current_time = time.time()
        if current_time - self.state['last_screenshot_time'] < 1.0:
            return {
                'action': 'screenshot_cooldown',
                'message': 'Screenshot cooldown active'
            }
        
        self.state['last_screenshot_time'] = current_time
        self.state['screenshot_count'] += 1
        
        return {
            'action': 'screenshot',
            'count': self.state['screenshot_count'],
            'message': f'Screenshot taken ({self.state["screenshot_count"]})'
        }
    
    def toggle_audio(self) -> Dict[str, Any]:
        """Toggle audio alerts"""
        self.state['audio_enabled'] = not self.state['audio_enabled']
        status = "enabled" if self.state['audio_enabled'] else "disabled"
        return {
            'action': 'toggle_audio',
            'audio_enabled': self.state['audio_enabled'],
            'message': f'Audio alerts {status}'
        }
    
    def toggle_fps_display(self) -> Dict[str, Any]:
        """Toggle FPS display"""
        self.state['fps_display'] = not self.state['fps_display']
        status = "shown" if self.state['fps_display'] else "hidden"
        return {
            'action': 'toggle_fps',
            'fps_display': self.state['fps_display'],
            'message': f'FPS display {status}'
        }
    
    def toggle_debug_mode(self) -> Dict[str, Any]:
        """Toggle debug mode"""
        self.state['debug_mode'] = not self.state['debug_mode']
        status = "enabled" if self.state['debug_mode'] else "disabled"
        return {
            'action': 'toggle_debug',
            'debug_mode': self.state['debug_mode'],
            'message': f'Debug mode {status}'
        }
    
    def mute_audio(self) -> Dict[str, Any]:
        """Mute/unmute audio"""
        self.state['audio_muted'] = not self.state['audio_muted']
        status = "muted" if self.state['audio_muted'] else "unmuted"
        return {
            'action': 'mute_audio',
            'audio_muted': self.state['audio_muted'],
            'message': f'Audio {status}'
        }
    
    def draw_help_overlay(self, frame):
        """Draw help overlay on frame"""
        if not self.state['show_help']:
            return frame
            
        overlay = frame.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 25
        
        max_width = 0
        for line in self.help_text:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_width = max(max_width, text_width)
        
        box_width = max_width + 40
        box_height = len(self.help_text) * line_height + 40
        
        frame_height, frame_width = frame.shape[:2]
        box_x = (frame_width - box_width) // 2
        box_y = (frame_height - box_height) // 2
        
        cv2.rectangle(overlay, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.rectangle(frame, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     (255, 255, 255), 2)
        
        for i, line in enumerate(self.help_text):
            text_x = box_x + 20
            text_y = box_y + 30 + (i * line_height)
            
            if line.startswith("==="):
                color = (0, 255, 255)
                cv2.putText(frame, line, (text_x, text_y), 
                           font, 0.6, color, 2, cv2.LINE_AA)
            elif line.startswith("["):
                color = (0, 255, 0)
                cv2.putText(frame, line, (text_x, text_y), 
                           font, font_scale, color, font_thickness, cv2.LINE_AA)
            elif line.strip() == "":
                continue
            else:
                color = (255, 255, 255)
                cv2.putText(frame, line, (text_x, text_y), 
                           font, font_scale, color, font_thickness, cv2.LINE_AA)
        
        return frame
    
    def draw_status_bar(self, frame):
        """Draw status bar with current control states"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        
        status_items = []
        
        if self.state['paused']:
            status_items.append("[PAUSED]")
        if not self.state['audio_enabled']:
            status_items.append("[AUDIO OFF]")
        if self.state['audio_muted']:
            status_items.append("[MUTED]")
        if self.state['debug_mode']:
            status_items.append("[DEBUG]")
        
        if status_items:
            status_text = " ".join(status_items)
            (text_width, text_height), _ = cv2.getTextSize(status_text, font, font_scale, 1)
            
            status_x = frame.shape[1] - text_width - 10
            status_y = 25
            
            cv2.rectangle(frame, (status_x - 5, status_y - text_height - 5), 
                         (status_x + text_width + 5, status_y + 5), 
                         (50, 50, 50), -1)
            
            cv2.putText(frame, status_text, (status_x, status_y), 
                       font, font_scale, (255, 255, 0), 1, cv2.LINE_AA)
        
        return frame
    
    def get_state(self) -> Dict[str, Any]:
        """Get current control state"""
        return self.state.copy()

# Global controls manager instance
_controls_manager = ControlsManager()

def process_keyboard_input(key: int) -> Dict[str, Any]:
    """Process keyboard input and return action results"""
    return _controls_manager.process_key(key)

def draw_controls_overlay(frame):
    """Draw control overlays on frame"""
    frame = _controls_manager.draw_help_overlay(frame)
    frame = _controls_manager.draw_status_bar(frame)
    return frame

def get_controls_state() -> Dict[str, Any]:
    """Get current controls state"""
    return _controls_manager.get_state()

if __name__ == "__main__":
    # Test the controls system
    print("Testing controls system...")
    print("Available controls:")
    for key, func in _controls_manager.controls.items():
        if key == 27:
            print(f"ESC: {func.__name__}")
        else:
            print(f"{chr(key).upper()}: {func.__name__}")
    
    # Test some functions
    result = _controls_manager.show_help()
    print(f"\nTest result: {result}")
    
    state = _controls_manager.get_state()
    print(f"Current state: {state}") 