import threading
import time
import numpy as np
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Audio alerts disabled.")

class AudioAlerts:
    """Manages audio alerts for different warning levels"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and PYGAME_AVAILABLE
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # Minimum seconds between alerts
        
        if self.enabled:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.generate_alert_sounds()
            except Exception as e:
                print(f"Warning: Could not initialize audio system: {e}")
                self.enabled = False
    
    def generate_alert_sounds(self):
        """Generate different alert sounds for different warning levels"""
        if not self.enabled:
            return
            
        # Generate beep sounds with different patterns
        sample_rate = 22050
        
        # Level 1: Single beep (Suspicious activity)
        self.level1_sound = self._generate_beep(800, 0.2, sample_rate)
        
        # Level 2: Double beep (Falling)
        beep1 = self._generate_beep(1000, 0.15, sample_rate)
        silence = np.zeros(int(0.1 * sample_rate))
        beep2 = self._generate_beep(1000, 0.15, sample_rate)
        self.level2_sound = np.concatenate([beep1, silence, beep2])
        
        # Level 3: Triple urgent beep (Violence)
        beep1 = self._generate_beep(1200, 0.1, sample_rate)
        silence = np.zeros(int(0.05 * sample_rate))
        beep2 = self._generate_beep(1200, 0.1, sample_rate)
        beep3 = self._generate_beep(1200, 0.1, sample_rate)
        self.level3_sound = np.concatenate([beep1, silence, beep2, silence, beep3])
    
    def _generate_beep(self, frequency, duration, sample_rate):
        """Generate a beep sound at specified frequency and duration"""
        frames = int(duration * sample_rate)
        arr = np.linspace(0, duration, frames)
        wave = np.sin(2 * np.pi * frequency * arr) * 0.3  # Lower volume
        
        # Apply envelope to avoid clicks
        envelope_frames = int(0.01 * sample_rate)  # 10ms fade in/out
        wave[:envelope_frames] *= np.linspace(0, 1, envelope_frames)
        wave[-envelope_frames:] *= np.linspace(1, 0, envelope_frames)
        
        # Convert to 16-bit integers
        wave = (wave * 32767).astype(np.int16)
        
        # Make stereo
        stereo_wave = np.zeros((frames, 2), dtype=np.int16)
        stereo_wave[:, 0] = wave
        stereo_wave[:, 1] = wave
        
        return stereo_wave
    
    def play_alert(self, warning_level, action_name="Unknown"):
        """Play appropriate alert sound based on warning level"""
        if not self.enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return  # Still in cooldown period
            
        self.last_alert_time = current_time
        
        # Play alert in separate thread to avoid blocking
        thread = threading.Thread(target=self._play_sound_async, 
                                 args=(warning_level, action_name))
        thread.daemon = True
        thread.start()
    
    def _play_sound_async(self, warning_level, action_name):
        """Play sound asynchronously"""
        try:
            if warning_level >= 3:
                sound_array = self.level3_sound
            elif warning_level >= 2:
                sound_array = self.level2_sound
            elif warning_level >= 1:
                sound_array = self.level1_sound
            else:
                return
            sound = pygame.sndarray.make_sound(sound_array)
            sound.play()
            while pygame.mixer.get_busy():
                time.sleep(0.1)
        except Exception as e:
            pass
    
    def test_alerts(self):
        """Test all alert levels"""
        if not self.enabled:
            print("Audio alerts disabled")
            return
            
        print("Testing audio alerts...")
        for level in [1, 2, 3]:
            print(f"Playing level {level} alert...")
            self.play_alert(level, f"Test Level {level}")
            time.sleep(2)

# Global audio alerts instance
_audio_alerts = AudioAlerts()

def play_warning_sound(warning_level, action_name="Unknown"):
    """Convenient function to play warning sound"""
    _audio_alerts.play_alert(warning_level, action_name)

def test_audio_system():
    """Test the audio system"""
    _audio_alerts.test_alerts()

if __name__ == "__main__":
    # Test the audio system
    test_audio_system() 