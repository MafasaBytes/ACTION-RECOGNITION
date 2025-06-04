"""
Centralized Configuration Management
Consolidates configuration from main.py and config.py into a clean, validated system.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class VideoConfig:
    """Video processing configuration"""
    source: str = 'data/sample.mp4'
    output_path: str = 'src/recordings/output_sample.mp4'
    fps: Optional[int] = 60
    
@dataclass 
class DetectionConfig:
    """Object detection configuration"""
    model_name: str = "facebook/detr-resnet-101"
    confidence_threshold: float = 0.7
    
@dataclass
class ActionConfig:
    """Action recognition configuration"""
    model_name: str = "slowfast_r50"
    confidence_threshold: float = 0.1
    num_frames: int = 32
    frame_stride: int = 2
    enable_stabilization: bool = True
    min_stable_confidence: float = 0.3
    stability_threshold: float = 0.7
    history_length: int = 8
    
@dataclass
class AnomalyConfig:
    """Anomaly detection configuration"""
    confidence_threshold: float = 0.1
    
@dataclass
class TrackingConfig:
    """Object tracking configuration"""
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 30
    match_thresh: float = 0.8
    
@dataclass
class UIConfig:
    """User interface configuration"""
    enhanced_visualization: bool = True
    color_scheme: str = 'professional'
    enable_controls: bool = True
    show_performance: bool = True
    dashboard_width: int = 1400
    dashboard_height: int = 900
    
@dataclass
class AudioConfig:
    """Audio alerts configuration"""
    enabled: bool = True
    
@dataclass
class ActionHistoryConfig:
    """Action history management configuration"""
    max_history: int = 200
    persistence_duration: float = 4.0
    min_display_duration: float = 2.0
    confidence_decay_rate: float = 0.98
    grouping_window: float = 3.0
    temporal_smoothing_window: int = 7
    display_strategy: str = 'mixed'
    
@dataclass
class SystemConfig:
    """Complete system configuration"""
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    action_history: ActionHistoryConfig = field(default_factory=ActionHistoryConfig)
    
    # Directories
    screenshot_dir: str = 'src/screenshots'
    output_dir: str = 'src/recordings'
    
class ConfigManager:
    """Centralized configuration management with validation and environment support"""
    
    def __init__(self, config_file: Optional[str] = None, environment: str = 'development'):
        self.environment = environment
        self.config_file = config_file
        self._config: Optional[SystemConfig] = None
        self._legacy_config: Optional[Dict[str, Any]] = None
        
    def load_config(self) -> SystemConfig:
        """Load and validate configuration from various sources"""
        if self._config is not None:
            return self._config
            
        # Start with default configuration
        config = SystemConfig()
        
        # Load from file if specified
        if self.config_file and Path(self.config_file).exists():
            config = self._load_from_file(config)
            
        # Override with environment variables
        config = self._load_from_environment(config)
        
        # Validate configuration
        self._validate_config(config)
        
        # Create directories if they don't exist
        self._ensure_directories(config)
        
        self._config = config
        logger.info(f"Configuration loaded for environment: {self.environment}")
        return config
    
    def get_legacy_config(self) -> Dict[str, Any]:
        """Convert to legacy format for backwards compatibility"""
        if self._legacy_config is not None:
            return self._legacy_config
            
        config = self.load_config()
        
        # Convert to legacy format for existing code
        legacy = {
            # Video settings
            'video_source': config.video.source,
            'output_video_path': config.video.output_path,
            'video_fps': config.video.fps,
            
            # Detection settings
            'detector_conf': config.detection.confidence_threshold,
            'detector_model_name': config.detection.model_name,
            
            # Action recognition settings  
            'action_rec_conf_thresh': config.action.confidence_threshold,
            'enable_action_stabilization': config.action.enable_stabilization,
            'min_stable_confidence': config.action.min_stable_confidence,
            'stability_threshold': config.action.stability_threshold,
            'action_history_length': config.action.history_length,
            
            # Anomaly settings
            'anomaly_conf_thresh': config.anomaly.confidence_threshold,
            
            # UI settings
            'enhanced_visualization': config.ui.enhanced_visualization,
            'color_scheme': config.ui.color_scheme,
            'enable_controls': config.ui.enable_controls,
            'show_performance': config.ui.show_performance,
            
            # Audio settings
            'audio_enabled': config.audio.enabled,
            
            # Directories
            'screenshot_dir': config.screenshot_dir,
            
            # Action History Management
            'max_action_history': config.action_history.max_history,
            'action_persistence_duration': config.action_history.persistence_duration,
            'min_action_display_duration': config.action_history.min_display_duration,
            'confidence_decay_rate': config.action_history.confidence_decay_rate,
            'action_grouping_window': config.action_history.grouping_window,
            'temporal_smoothing_window': config.action_history.temporal_smoothing_window,
            'action_display_strategy': config.action_history.display_strategy,
        }
        
        self._legacy_config = legacy
        return legacy
    
    def _load_from_file(self, config: SystemConfig) -> SystemConfig:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                
            # Update config with file values (simplified for now)
            if 'video' in file_config:
                for key, value in file_config['video'].items():
                    if hasattr(config.video, key):
                        setattr(config.video, key, value)
                        
            # Add similar logic for other sections...
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_file}: {e}")
            
        return config
    
    def _load_from_environment(self, config: SystemConfig) -> SystemConfig:
        """Load configuration from environment variables"""
        env_mappings = {
            'CVAD_VIDEO_SOURCE': ('video', 'source'),
            'CVAD_DETECTOR_CONF': ('detection', 'confidence_threshold'),
            'CVAD_ACTION_CONF': ('action', 'confidence_threshold'),
            'CVAD_COLOR_SCHEME': ('ui', 'color_scheme'),
            'CVAD_AUDIO_ENABLED': ('audio', 'enabled'),
        }
        
        for env_var, (section, attr) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(config, section)
                current_value = getattr(section_obj, attr)
                
                # Type conversion based on current value type
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, int):
                    value = int(value)
                    
                setattr(section_obj, attr, value)
                logger.info(f"Config override from env {env_var}: {section}.{attr} = {value}")
                
        return config
    
    def _validate_config(self, config: SystemConfig):
        """Validate configuration values"""
        # Validate confidence thresholds
        assert 0.0 <= config.detection.confidence_threshold <= 1.0, "Detection confidence must be 0-1"
        assert 0.0 <= config.action.confidence_threshold <= 1.0, "Action confidence must be 0-1"
        assert 0.0 <= config.anomaly.confidence_threshold <= 1.0, "Anomaly confidence must be 0-1"
        
        # Validate video source exists (if not a device index)
        if not config.video.source.isdigit() and not Path(config.video.source).exists():
            logger.warning(f"Video source {config.video.source} does not exist")
            
        # Validate display strategy
        valid_strategies = ['recent', 'persistent', 'frequent', 'mixed']
        assert config.action_history.display_strategy in valid_strategies, \
            f"Display strategy must be one of {valid_strategies}"
        
        logger.info("Configuration validation passed")
    
    def _ensure_directories(self, config: SystemConfig):
        """Create necessary directories"""
        directories = [
            config.screenshot_dir,
            config.output_dir,
            Path(config.video.output_path).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def save_config(self, filename: Optional[str] = None):
        """Save current configuration to file"""
        if not self._config:
            logger.warning("No configuration loaded to save")
            return
            
        filename = filename or f"config_{self.environment}.json"
        
        # Convert to dict for JSON serialization (simplified)
        config_dict = {
            'video': {
                'source': self._config.video.source,
                'output_path': self._config.video.output_path,
                'fps': self._config.video.fps,
            },
            'detection': {
                'confidence_threshold': self._config.detection.confidence_threshold,
                'model_name': self._config.detection.model_name,
            },
            # Add other sections...
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        logger.info(f"Configuration saved to {filename}")

# Global instance for easy access
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_file: Optional[str] = None, environment: str = 'development') -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file, environment)
    return _config_manager

def get_config() -> SystemConfig:
    """Get the current system configuration"""
    return get_config_manager().load_config()

def get_legacy_config() -> Dict[str, Any]:
    """Get configuration in legacy format for backwards compatibility"""
    return get_config_manager().get_legacy_config() 