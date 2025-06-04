import torch
from pytorchvideo.models.hub import slowfast_r50 # Or other models like x3d, mvit, etc.
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from collections import deque, defaultdict
import cv2 # For resizing if needed, though transforms should handle it
import time

from src.actions.transforms import ActionTransforms
from src.config import (
    NUM_FRAMES, 
    ACTION_MODEL_NAME, 
    ACTION_PREDICTION_THRESHOLD,
    KINETICS_CLASSES, # For mapping predictions to class names
    DEFAULT_VIDEO_FPS
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ActionStabilizer:
    """Handles temporal smoothing and stabilization of action predictions"""
    
    def __init__(self, history_length=5, min_confidence=0.0, stability_threshold=0.6):
        self.history_length = history_length
        self.min_confidence = min_confidence  # Set to 0.0 to allow all Global Actions
        self.stability_threshold = stability_threshold
        
        # Store action history: {action_class_id: deque of (confidence, timestamp)}
        self.action_history = defaultdict(lambda: deque(maxlen=history_length))
        self.last_stable_action = None
        self.last_stable_confidence = 0.0
        self.last_update_time = time.time()
        
    def add_prediction(self, action_class_id, confidence):
        """Add a new prediction to the history"""
        current_time = time.time()
        self.action_history[action_class_id].append((confidence, current_time))
        self.last_update_time = current_time
        
    def get_stable_action(self):
        """Get the most stable action based on recent history"""
        current_time = time.time()
        
        # Clean old entries (older than 2 seconds)
        for action_id in list(self.action_history.keys()):
            history = self.action_history[action_id]
            # Remove old entries
            while history and current_time - history[0][1] > 2.0:
                history.popleft()
            # Remove empty histories
            if not history:
                del self.action_history[action_id]
        
        if not self.action_history:
            return None, 0.0
            
        # Calculate stability scores for each action
        action_scores = {}
        for action_id, history in self.action_history.items():
            if len(history) < 2:  # Need at least 2 predictions
                continue
                
            # Calculate average confidence
            confidences = [conf for conf, _ in history]
            avg_confidence = np.mean(confidences)
            
            # Calculate consistency (lower std = more consistent)
            consistency = 1.0 / (1.0 + np.std(confidences))
            
            # Calculate recency (more recent = better)
            latest_time = history[-1][1]
            recency = max(0, 1.0 - (current_time - latest_time))
            
            # Calculate frequency (more frequent = more stable)
            frequency = len(history) / self.history_length
            
            # Combined stability score
            stability_score = avg_confidence * consistency * recency * frequency
            
            # For Global Actions, include all actions regardless of confidence
            action_scores[action_id] = (stability_score, avg_confidence)
        
        if not action_scores:
            return None, 0.0
            
        # Get the most stable action
        best_action = max(action_scores.items(), key=lambda x: x[1][0])
        action_id, (stability_score, avg_confidence) = best_action
        
        # Only return if stability is above threshold
        if stability_score >= self.stability_threshold:
            self.last_stable_action = action_id
            self.last_stable_confidence = avg_confidence
            return action_id, avg_confidence
        
        # If no stable action, return the last stable one if it's recent
        if (self.last_stable_action is not None and 
            current_time - self.last_update_time < 1.0):
            return self.last_stable_action, self.last_stable_confidence * 0.9  # Decay confidence
            
        return None, 0.0

class ActionRecognizer:
    def __init__(self, cfg: Dict[str, Any]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ActionRecognizer initializing on device: {self.device}")

        self.num_frames = cfg.get('num_frames_action', NUM_FRAMES)
        self.action_model_name = cfg.get('action_model_name', ACTION_MODEL_NAME)
        self.confidence_threshold = cfg.get('action_rec_conf_thresh', ACTION_PREDICTION_THRESHOLD)
        self.fps = cfg.get('video_fps', DEFAULT_VIDEO_FPS) # FPS for buffer management

        # Stabilization parameters
        self.stabilization_enabled = cfg.get('enable_action_stabilization', True)
        self.min_stable_confidence = cfg.get('min_stable_confidence', 0.0)  # No confidence constraint for Global Actions
        self.stability_threshold = cfg.get('stability_threshold', 0.7)
        self.history_length = cfg.get('action_history_length', 8)

        logger.info(f"Loading action recognition model: {self.action_model_name}")
        try:
            if self.action_model_name == "slowfast_r50":
                self.model = slowfast_r50(pretrained=True)
            # Add more models here if needed:
            # elif self.action_model_name == "x3d_m":
            #     self.model = torch.hub.load("facebookresearch/pytorchvideo", model="x3d_m", pretrained=True)
            else:
                raise ValueError(f"Unsupported action model: {self.action_model_name}")
            
            self.model.to(self.device).eval()
            logger.info(f"Action model {self.action_model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading action model {self.action_model_name}: {e}")
            raise

        self.transforms = ActionTransforms(num_frames=self.num_frames)
        
        # Frame buffer: stores tuples of (frame, list_of_tracked_objects_in_that_frame)
        # We need to decide how to associate actions with tracks.
        # For simplicity, we can run action recognition on the whole frame first.
        # Or, for a more advanced system, on crops around tracked persons.
        self.frame_buffer = deque(maxlen=self.num_frames)
        
        # Buffer for each tracked person, key: track_id
        # Value: deque of cropped frames around that person
        self.person_track_buffers: Dict[int, deque] = {}
        
        # Initialize action stabilizer
        if self.stabilization_enabled:
            self.stabilizer = ActionStabilizer(
                history_length=self.history_length,
                min_confidence=self.min_stable_confidence,
                stability_threshold=self.stability_threshold
            )
            logger.info(f"Action stabilization enabled with min_confidence={self.min_stable_confidence}, stability_threshold={self.stability_threshold}")
        else:
            self.stabilizer = None
            logger.info("Action stabilization disabled")

    def _pack_pathway_output(self, frames: torch.Tensor) -> List[torch.Tensor]:
        """
        Prepares the input for SlowFast by creating slow and fast pathways.
        Input 'frames' is (B, C, T, H, W), e.g., (1, 3, 32, 224, 224)
        """
        if self.action_model_name != "slowfast_r50":
            # For models not requiring pathway packing, just return the input in a list
            return [frames.to(self.device)]

        # For slowfast_r50, the slow pathway typically samples at 1/4th the frame rate (alpha=4)
        # The pretrained model from PyTorchVideo hub uses ALPHA = 4 in its configuration.
        alpha = 4 
        slow_temporal_stride = alpha

        fast_pathway = frames
        
        if frames.shape[2] < slow_temporal_stride:
            # If not enough frames for a slow pathway stride (e.g. at the beginning of video)
            # Duplicate the first frame of the fast pathway to make up the slow pathway
            # This is a simple strategy; more sophisticated padding or waiting might be needed
            num_slow_frames_needed = frames.shape[2] // slow_temporal_stride
            if num_slow_frames_needed == 0 : num_slow_frames_needed = 1 # Must have at least one
            
            slow_indices = torch.zeros(num_slow_frames_needed, device=frames.device).long()
        else:
            slow_indices = torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // slow_temporal_stride
            ).long()

        slow_pathway = frames[:, :, slow_indices, :, :]
        
        return [slow_pathway.to(self.device), fast_pathway.to(self.device)]

    def recognize(self, current_frame: np.ndarray, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Recognizes actions in the current_frame, potentially focusing on tracked persons.
        # Returns a list of action predictions, possibly associated with tracks.
        if current_frame is None or current_frame.size == 0:
            logger.warning("ActionRecognizer received an empty frame.")
            return []
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
        self.frame_buffer.append(frame_tensor)
        if len(self.frame_buffer) < self.num_frames:
            logger.debug(f"Buffering frames for action recognition, have {len(self.frame_buffer)}/{self.num_frames}")
            logger.info(f"ActionRecognizer: Buffering frames ({len(self.frame_buffer)}/{self.num_frames})")
            return []
        logger.info(f"ActionRecognizer: Processing {self.num_frames} frames for action recognition")
        clip_chw_list = list(self.frame_buffer)
        clip_tchw = torch.stack(clip_chw_list, dim=0)
        clip_thwc = clip_tchw.permute(0, 2, 3, 1)
        transformed_clip = self.transforms(clip_thwc)
        batch_clip = transformed_clip.unsqueeze(0).to(self.device)
        model_input = self._pack_pathway_output(batch_clip)
        
        action_predictions = []
        try:
            with torch.inference_mode():
                preds_raw = self.model(model_input)
            post_act = torch.nn.Softmax(dim=-1)
            preds_probs = post_act(preds_raw)
            
            # Get top predictions
            pred_scores, pred_classes_indices = torch.topk(preds_probs, k=3, dim=-1)
            
            # Get person tracks for individual action assignment
            person_tracks = [track for track in tracks if track.get('class_name', '').lower() == 'person']
            
            for i in range(pred_scores.shape[0]):
                # Process top prediction for global action
                score = pred_scores[i][0].item()
                class_idx = pred_classes_indices[i][0].item()
                class_name = KINETICS_CLASSES[class_idx] if 0 <= class_idx < len(KINETICS_CLASSES) else f"Unknown Class ({class_idx})"
                
                # Always add global action
                global_action = {
                    "action_class_id": class_idx,
                    "action_name": class_name,
                    "action_confidence": score,
                    "track_id": -1  # Global action indicator
                }
                action_predictions.append(global_action)
                
                # Add to stabilizer if enabled
                if self.stabilizer:
                    self.stabilizer.add_prediction(class_idx, score)
                
                # Associate actions with person tracks
                if person_tracks and score > 0.1:  # Only associate if there's reasonable confidence
                    # For simplicity, assign the top action to all detected persons
                    # In a more sophisticated system, you'd analyze crops around each person
                    for track in person_tracks:
                        track_id = track.get('object_id') or track.get('track_id')
                        if track_id is not None:
                            person_action = {
                                "action_class_id": class_idx,
                                "action_name": class_name,
                                "action_confidence": score * 0.9,  # Slightly reduce confidence for individual assignment
                                "track_id": track_id
                            }
                            action_predictions.append(person_action)
                
                logger.debug(f"Action Recognition: Class Idx: {class_idx}, Name: {class_name}, Score: {score:.4f}")
                    
        except Exception as e:
            logger.error(f"Error during action recognition inference: {e}")
            return []
        
        logger.info(f"ActionRecognizer: Detected {len(action_predictions)} Action(s) (global + person-specific)")
        return action_predictions

# Example Usage (for testing this module standalone)
if __name__ == '__main__':
    dummy_cfg = {
        'num_frames_action': 16, # Use fewer frames for faster test
        'action_model_name': 'slowfast_r50',
        'action_rec_conf_thresh': 0.01, # Low threshold for testing
        'video_fps': 30
    }
    
    recognizer = ActionRecognizer(dummy_cfg)
    logger.info("ActionRecognizer initialized for testing.")

    # Create dummy frames
    num_test_frames = dummy_cfg['num_frames_action'] + 5 # More than buffer
    dummy_frames = []
    for i in range(num_test_frames):
        frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8) # H, W, C (BGR)
        cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        dummy_frames.append(frame)

    logger.info(f"Processing {len(dummy_frames)} dummy frames...")
    for i, frame in enumerate(dummy_frames):
        # Dummy tracks (not used in current global recognition)
        dummy_tracks = [{'track_id': 1, 'bbox': [0,0,10,10], 'class_name': 'person'}] 
        
        predictions = recognizer.recognize(frame, dummy_tracks)
        
        if predictions:
            logger.info(f"Frame {i+1}: Detected actions: {predictions}")
        elif len(recognizer.frame_buffer) == recognizer.num_frames:
            logger.info(f"Frame {i+1}: No actions above threshold, but buffer was full.")
        
        # Simulate display
        # cv2.imshow("Action Rec Test", frame)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break
    
    # cv2.destroyAllWindows()
    logger.info("Action recognition test finished.") 