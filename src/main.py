import os
import sys
import argparse
from typing import Dict, Any

# Env setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import modules
from src.core.config_manager import ConfigManager, get_config_manager
from src.core.color_manager import get_color_manager
from src.services.video_processing_service import VideoProcessingService
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_development_config() -> Dict[str, Any]:
    """Create development configuration override"""
    return {
        # Override existing config for development
        'video_source': 'data/sample.mp4',
        'enhanced_visualization': True,
        'color_scheme': 'professional',
        'enable_controls': True,
        'audio_enabled': True,
        'show_performance': True,
    }

def create_production_config() -> Dict[str, Any]:
    """Create production configuration override"""
    return {
        'enhanced_visualization': True,
        'color_scheme': 'professional',
        'enable_controls': False,  # Disable for production
        'audio_enabled': True,
        'show_performance': False,
    }

def run_video_processing(environment: str = 'development', config_file: str = None):
    """Run video processing with specified environment configuration"""
    try:
        # Initialize configuration manager
        config_manager = get_config_manager(config_file, environment)
        config = config_manager.get_legacy_config()
        
        # Apply environment-specific overrides
        if environment == 'development':
            config.update(create_development_config())
        elif environment == 'production':
            config.update(create_production_config())
            
        # Initialize color manager with configured scheme
        color_manager = get_color_manager(config.get('color_scheme', 'professional'))
        
        logger.info(f"Starting application in {environment} mode")
        logger.info(f"Using color scheme: {color_manager.get_scheme().name}")
        
        # Create and run video processing service
        service = VideoProcessingService(config)
        success = service.run()
        
        if success:
            logger.info("Video processing completed successfully")
        else:
            logger.error("Video processing failed")
            return False
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        return False
        
    return True

def run_detector_tracker_test(environment: str = 'development'):
    """Run simple detector and tracker test"""
    try:
        from src.detectors.object_detector import ObjectDetector
        from src.detectors.tracker import Tracker
        import cv2
        
        # Get configuration
        config_manager = get_config_manager(environment=environment)
        config = config_manager.get_legacy_config()
        
        logger.info("Running detector+tracker test mode")
        
        # Initialize components
        detector = ObjectDetector(config)
        tracker = Tracker(config)
        cap = cv2.VideoCapture(config['video_source'])
        
        if not cap.isOpened():
            logger.error(f"Could not open video source: {config['video_source']}")
            return False
            
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detection and tracking
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            
            # Simple visualization
            for obj in tracks:
                x1, y1, x2, y2 = obj['bbox']
                track_id = obj['object_id']
                class_name = obj['class_name']
                conf = obj['confidence']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} {class_name}:{conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv2.imshow("Detection+Tracking Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
                
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Test completed. Processed {frame_count} frames total.")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Computer Vision Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                                   # Run in development mode
  python src/main.py --env production                  # Run in production mode
  python src/main.py --config config.json              # Use specific config file
  python src/main.py --test-detector-tracker           # Test mode
  python src/main.py --env production --config prod.json  # Production with config
        """
    )
    
    parser.add_argument(
        '--env', '--environment',
        choices=['development', 'production', 'testing'],
        default='development',
        help='Environment to run in (default: development)'
    )
    
    parser.add_argument(
        '--config', '--config-file',
        type=str,
        help='Path to configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--test-detector-tracker',
        action='store_true',
        help='Run simple detector+tracker test without full pipeline'
    )
    
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to specified file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 60)
    logger.info("Computer Vision Anomaly Detection System")
    logger.info("Architecture Demo")
    logger.info("=" * 60)
    
    # Save configuration if requested
    if args.save_config:
        try:
            config_manager = get_config_manager(args.config, args.env)
            config_manager.save_config(args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")
            return
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return
    
    # Run appropriate mode
    success = False
    
    if args.test_detector_tracker:
        success = run_detector_tracker_test(args.env)
    else:
        success = run_video_processing(args.env, args.config)
    
    # Exit
    exit_code = 0 if success else 1
    logger.info(f"Application exiting with code: {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 