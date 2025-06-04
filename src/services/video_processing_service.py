"""
Video Processing Service
Encapsulates the main video processing pipeline logic from main.py
Provides clean separation of concerns and better testability
"""

import cv2
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from src.core.config_manager import get_legacy_config
from src.detectors.object_detector import ObjectDetector
from src.detectors.tracker import Tracker
from src.actions.action_recognizer import ActionRecognizer
from src.anomaly.anomaly_scorer import AnomalyScorer
from src.utils.enhanced_action_history import ActionHistoryManager
from src.utils.enhanced_visualization import EnhancedVisualizer
from src.utils.visualization import draw_results
from src.utils.logger import setup_logger

try:
    from src.utils.controls import process_keyboard_input, draw_controls_overlay
    CONTROLS_AVAILABLE = True
except ImportError:
    CONTROLS_AVAILABLE = False

try:
    from src.utils.audio_alerts import test_audio_system
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

logger = setup_logger(__name__)

class VideoProcessingService:
    """
    Service for handling video processing pipeline
    Encapsulates detection, tracking, action recognition, and visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Use provided config or load default
        self.config = config or get_legacy_config()
        
        # Initialize components
        self.detector = ObjectDetector(self.config)
        self.tracker = Tracker(self.config)
        self.action_recognizer = ActionRecognizer(self.config)
        self.anomaly_scorer = AnomalyScorer(self.config)
        self.action_history_manager = ActionHistoryManager(self.config)
        
        # Initialize visualizer if enhanced visualization is enabled
        if self.config.get('enhanced_visualization', False):
            self.visualizer = EnhancedVisualizer(
                self.config.get('color_scheme', 'professional')
            )
        else:
            self.visualizer = None
            
        # Video capture and output
        self.cap: Optional[cv2.VideoCapture] = None
        self.out: Optional[cv2.VideoWriter] = None
        
        # Processing state
        self.frame_count = 0
        self.start_time = 0.0
        self.paused = False
        self.running = False
        
        logger.info("VideoProcessingService initialized")
        
    def initialize_video(self) -> bool:
        """Initialize video capture and output"""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.config['video_source'])
            if not self.cap.isOpened():
                logger.error(f"Could not open video source: {self.config['video_source']}")
                return False
                
            # Get video properties
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # Update config with actual FPS if not set
            if self.config['video_fps'] is None:
                self.config['video_fps'] = fps
            self.action_recognizer.fps = fps
            
            # Create output directory
            output_dir = os.path.dirname(self.config['output_video_path'])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Create screenshot directory
            screenshot_dir = self.config.get('screenshot_dir', 'src/screenshots')
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
                
            # Initialize video writer with fallback to image sequence
            output_path = self.config['output_video_path']
            
            # Try different approaches in order of preference
            approaches = [
                self._try_video_writer_with_codecs,
                self._try_image_sequence_fallback
            ]
            
            for approach in approaches:
                if approach(output_path, fps, frame_width, frame_height):
                    break
            else:
                logger.error("Failed to initialize any video output method")
                return False
            
            logger.info(f"Video initialized: {frame_width}x{frame_height} @ {fps}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize video: {e}")
            return False
            
    def _try_video_writer_with_codecs(self, output_path: str, fps: int, width: int, height: int) -> bool:
        """Try to initialize video writer with various codecs"""
        # Try different codecs in order of preference
        codecs_to_try = [
            ('mp4v', '.mp4'),  # MP4 container with MPEG-4 codec
            ('XVID', '.avi'),  # AVI container with XVID codec  
            ('MJPG', '.avi'),  # AVI container with Motion JPEG
            ('X264', '.mp4'),  # H.264 codec (if available)
        ]
        
        for codec_name, extension in codecs_to_try:
            try:
                # Adjust output path extension if needed
                if not output_path.endswith(extension):
                    base_name = os.path.splitext(output_path)[0]
                    test_output_path = base_name + extension
                else:
                    test_output_path = output_path
                    
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                test_writer = cv2.VideoWriter(test_output_path, fourcc, fps, (width, height))
                
                # Test if writer was successfully created by writing a test frame
                if test_writer.isOpened():
                    # Create a test frame
                    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    success = test_writer.write(test_frame)
                    
                    if success:
                        self.out = test_writer
                        self.output_mode = 'video'
                        self.config['output_video_path'] = test_output_path
                        logger.info(f"Successfully initialized video writer with codec: {codec_name}")
                        logger.info(f"Output will be saved to: {test_output_path}")
                        return True
                    else:
                        test_writer.release()
                        logger.warning(f"Codec {codec_name} opened but failed to write test frame")
                else:
                    test_writer.release()
                    
            except Exception as e:
                logger.warning(f"Failed to initialize video writer with codec {codec_name}: {e}")
                continue
        
        return False
        
    def _try_image_sequence_fallback(self, output_path: str, fps: int, width: int, height: int) -> bool:
        """Fallback to saving frames as image sequence"""
        try:
            # Create directory for image sequence
            base_name = os.path.splitext(output_path)[0]
            self.frames_dir = f"{base_name}_frames"
            os.makedirs(self.frames_dir, exist_ok=True)
            
            # Store metadata for later video creation
            self.frame_metadata = {
                'fps': fps,
                'width': width,
                'height': height,
                'frames_dir': self.frames_dir,
                'output_path': output_path
            }
            
            self.out = None  # No video writer
            self.output_mode = 'images'
            self.frame_counter = 0
            
            logger.info(f"Fallback to image sequence mode")
            logger.info(f"Frames will be saved to: {self.frames_dir}")
            logger.info(f"Final video creation will be attempted at cleanup")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize image sequence fallback: {e}")
            return False
            
    def process_frame(self, frame) -> Tuple[Any, bool]:
        """
        Process a single frame through the detection pipeline
        Returns: (processed_frame, should_continue)
        """
        try:
            # Object detection
            detections = self.detector.detect(frame)
            
            # Object tracking
            tracks = self.tracker.update(detections, frame)
            
            # Action recognition
            actions = self.action_recognizer.recognize(frame, tracks)
            
            # Add actions to history manager
            self.action_history_manager.add_actions(actions)
            
            # Get display actions using configured strategy
            display_actions = self.action_history_manager.get_display_actions(
                strategy=self.config.get('action_display_strategy', 'mixed')
            )
            
            # Anomaly scoring
            anomaly_scores = self.anomaly_scorer.score_actions(actions)
            display_anomaly_scores = self.anomaly_scorer.score_actions(display_actions)
            
            # Visualization
            if self.visualizer:
                vis_frame = self.visualizer.draw_enhanced_results(
                    frame, tracks, display_actions, display_anomaly_scores, 
                    self.action_history_manager
                )
            else:
                vis_frame = draw_results(frame, tracks, display_actions, display_anomaly_scores)
                
            # Add controls overlay if available
            if CONTROLS_AVAILABLE and self.config.get('enable_controls', True):
                vis_frame = draw_controls_overlay(vis_frame)
                
            return vis_frame, True
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, False
            
    def handle_keyboard_input(self, key: int) -> Dict[str, Any]:
        """Handle keyboard input and return control result"""
        if CONTROLS_AVAILABLE and self.config.get('enable_controls', True):
            return process_keyboard_input(key)
        else:
            # Basic controls
            if key == ord('q') or key == 27:  # 'q' or ESC
                return {'action': 'quit'}
            elif key == ord(' '):  # Space for pause
                self.paused = not self.paused
                return {'action': 'pause', 'paused': self.paused}
                
        return {'action': 'none'}
        
    def take_screenshot(self, frame, control_result: Dict[str, Any]):
        """Take a screenshot with current detections"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_dir = self.config.get('screenshot_dir', 'src/screenshots')
            screenshot_count = control_result.get('count', 1)
            
            screenshot_path = os.path.join(
                screenshot_dir, 
                f"screenshot_{timestamp}_{screenshot_count:03d}.jpg"
            )
            
            # Process frame for screenshot
            processed_frame, _ = self.process_frame(frame)
            cv2.imwrite(screenshot_path, processed_frame)
            
            logger.info(f"Screenshot saved: {screenshot_path}")
            
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            
    def log_statistics(self):
        """Log processing statistics"""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Get action history statistics
        action_stats = self.action_history_manager.get_statistics()
        
        logger.info(f"Processed {self.frame_count} frames, Avg FPS: {avg_fps:.2f}")
        logger.info(f"Action Statistics: {action_stats['total_actions_recorded']} total, "
                   f"{action_stats['persistent_actions_count']} persistent, "
                   f"{action_stats['actions_last_10_seconds']} in last 10s")
        if action_stats.get('most_frequent_action'):
            logger.info(f"Most frequent action: {action_stats['most_frequent_action']}")
            
    def run(self) -> bool:
        """Run the video processing pipeline"""
        if not self.initialize_video():
            return False
            
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        logger.info("Starting video processing pipeline...")
        
        if CONTROLS_AVAILABLE:
            logger.info("Press [H] for help with keyboard controls")
            
        try:
            while self.running:
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                control_result = self.handle_keyboard_input(key)
                
                if control_result['action'] == 'quit':
                    logger.info("Application quit requested")
                    break
                elif control_result['action'] == 'pause':
                    self.paused = control_result.get('paused', self.paused)
                    logger.info(f"{'Paused' if self.paused else 'Resumed'}")
                elif control_result['action'] == 'screenshot':
                    # Read frame for screenshot
                    ret, frame = self.cap.read()
                    if ret:
                        self.take_screenshot(frame, control_result)
                        # Reset frame position
                        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
                    continue
                elif control_result['action'] == 'reset_stats':
                    self.frame_count = 0
                    self.start_time = time.time()
                    logger.info("Statistics reset")
                    
                # Skip processing if paused
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video or error reading frame")
                    break
                    
                self.frame_count += 1
                
                # Process frame
                processed_frame, success = self.process_frame(frame)
                if not success:
                    logger.warning(f"Frame {self.frame_count} processing failed")
                    
                # Display and save
                cv2.imshow("Enhanced Anomaly Detection System", processed_frame)
                
                # Save frame based on output mode
                if hasattr(self, 'output_mode'):
                    if self.output_mode == 'video' and self.out and self.out.isOpened():
                        try:
                            success = self.out.write(processed_frame)
                            if not success:
                                logger.warning(f"Failed to write frame {self.frame_count} to output video")
                        except Exception as e:
                            logger.error(f"Error writing frame {self.frame_count}: {e}")
                    elif self.output_mode == 'images':
                        try:
                            # Save frame as image
                            frame_filename = os.path.join(self.frames_dir, f"frame_{self.frame_counter:06d}.jpg")
                            cv2.imwrite(frame_filename, processed_frame)
                            self.frame_counter += 1
                            
                            # Log progress occasionally
                            if self.frame_counter % 100 == 0:
                                logger.info(f"Saved {self.frame_counter} frames as images")
                        except Exception as e:
                            logger.error(f"Error saving frame {self.frame_count} as image: {e}")
                    else:
                        logger.warning("Unknown output mode or video writer not properly initialized")
                else:
                    logger.error("Output mode not set - video initialization may have failed")
                    
                # Log statistics periodically
                if self.frame_count % 100 == 0:
                    self.log_statistics()
                    
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            return False
        finally:
            self.cleanup()
            
        # Final statistics
        self.log_final_statistics()
        return True
        
    def log_final_statistics(self):
        """Log final processing statistics"""
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        logger.info("Video processing complete!")
        logger.info(f"Final Statistics:")
        logger.info(f"   Total frames processed: {self.frame_count}")
        logger.info(f"   Total time: {total_time:.2f} seconds")
        logger.info(f"   Average FPS: {avg_fps:.2f}")
        logger.info(f"   Output saved to: {self.config['output_video_path']}")
        
        if AUDIO_AVAILABLE:
            logger.info("Audio system was available")
        if CONTROLS_AVAILABLE:
            logger.info("Enhanced controls were available")
            
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
                
            if self.out:
                self.out.release()
                
            # Handle different output modes
            if hasattr(self, 'output_mode'):
                if self.output_mode == 'video':
                    self._verify_video_output()
                elif self.output_mode == 'images':
                    self._convert_images_to_video()
            else:
                logger.warning("No output mode set during cleanup")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            cv2.destroyAllWindows()
            logger.info("Resources cleaned up")
            
    def _verify_video_output(self):
        """Verify that video output was created successfully"""
        try:
            if os.path.exists(self.config['output_video_path']):
                file_size = os.path.getsize(self.config['output_video_path'])
                if file_size > 1000:  # More than 1KB indicates successful writing
                    logger.info(f"Output video saved successfully: {self.config['output_video_path']} ({file_size} bytes)")
                    
                    # Quick verification that the video can be opened
                    test_cap = cv2.VideoCapture(self.config['output_video_path'])
                    if test_cap.isOpened():
                        frame_count = test_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        logger.info(f"Output video verification: {frame_count} frames")
                        test_cap.release()
                    else:
                        logger.warning("Output video file cannot be opened for verification")
                else:
                    logger.error(f"Output video file is too small ({file_size} bytes) - likely corrupted")
            else:
                logger.error("Output video file was not created")
        except Exception as e:
            logger.error(f"Error verifying video output: {e}")
            
    def _convert_images_to_video(self):
        """Convert saved image sequence to video using external tools or simple concatenation"""
        try:
            if not hasattr(self, 'frame_metadata'):
                logger.error("No frame metadata available for video conversion")
                return
                
            logger.info(f"Converting {self.frame_counter} frames to video...")
            
            # First, try to create video using a simpler approach
            # Get list of frame files
            frame_files = []
            for i in range(self.frame_counter):
                frame_path = os.path.join(self.frames_dir, f"frame_{i:06d}.jpg")
                if os.path.exists(frame_path):
                    frame_files.append(frame_path)
                    
            if not frame_files:
                logger.error("No frame files found for video conversion")
                return
                
            logger.info(f"Found {len(frame_files)} frame files")
            
            # Try to create video with a very basic codec
            metadata = self.frame_metadata
            output_path = metadata['output_path']
            
            # Change extension to AVI for maximum compatibility
            base_name = os.path.splitext(output_path)[0]
            avi_output = base_name + ".avi"
            
            # Try creating video with uncompressed codec first
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG is widely supported
            writer = cv2.VideoWriter(avi_output, fourcc, metadata['fps'], 
                                   (metadata['width'], metadata['height']))
            
            if writer.isOpened():
                frames_written = 0
                for frame_path in frame_files:
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        # Resize frame if necessary
                        if frame.shape[:2] != (metadata['height'], metadata['width']):
                            frame = cv2.resize(frame, (metadata['width'], metadata['height']))
                        
                        success = writer.write(frame)
                        if success:
                            frames_written += 1
                        else:
                            logger.warning(f"Failed to write frame from {frame_path}")
                    
                    if frames_written % 50 == 0:
                        logger.info(f"Converted {frames_written}/{len(frame_files)} frames")
                
                writer.release()
                
                # Verify the created video
                if os.path.exists(avi_output):
                    file_size = os.path.getsize(avi_output)
                    if file_size > 1000:
                        self.config['output_video_path'] = avi_output  # Update config
                        logger.info(f"Successfully converted frames to video: {avi_output} ({file_size} bytes)")
                        
                        # Test the video
                        test_cap = cv2.VideoCapture(avi_output)
                        if test_cap.isOpened():
                            test_frames = test_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            test_cap.release()
                            logger.info(f"Conversion verification: {test_frames} frames in output video")
                        else:
                            logger.warning("Converted video cannot be opened")
                    else:
                        logger.error(f"Converted video file is too small ({file_size} bytes)")
                else:
                    logger.error("Video conversion failed - output file not created")
            else:
                logger.error("Failed to create video writer for frame conversion")
                writer.release()
                
                # Fallback: create info file about the frames
                info_file = base_name + "_frames_info.txt"
                with open(info_file, 'w') as f:
                    f.write(f"Frame sequence information:\n")
                    f.write(f"Frames directory: {self.frames_dir}\n")
                    f.write(f"Total frames: {self.frame_counter}\n")
                    f.write(f"FPS: {metadata['fps']}\n")
                    f.write(f"Resolution: {metadata['width']}x{metadata['height']}\n")
                    f.write(f"To convert to video manually, use ffmpeg:\n")
                    f.write(f"ffmpeg -r {metadata['fps']} -i {self.frames_dir}/frame_%06d.jpg -c:v libx264 {output_path}\n")
                
                logger.info(f"Created frame info file: {info_file}")
                logger.info(f"You can manually convert frames using ffmpeg or similar tools")
                
        except Exception as e:
            logger.error(f"Error converting images to video: {e}")
        
    def stop(self):
        """Stop the processing pipeline"""
        self.running = False 