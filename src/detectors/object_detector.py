import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

from src.config import DETECTOR_MODEL_NAME, OBJECT_DETECTION_CONFIDENCE_THRESHOLD
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ObjectDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ObjectDetector initializing on device: {self.device}")

        model_name = cfg.get('detector_model_name', DETECTOR_MODEL_NAME)
        self.confidence_threshold = cfg.get(
            'detector_conf', 
            cfg.get('obj_detection_thresh', OBJECT_DETECTION_CONFIDENCE_THRESHOLD)
        )
        
        logger.info(f"Loading DETR model: {model_name}")
        try:
            self.processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
            self.model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")
            self.model.to(self.device).eval()
            logger.info(f"DETR model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading DETR model {model_name}: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            logger.warning("ObjectDetector received an empty frame.")
            return []

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=self.confidence_threshold
            )[0]
            
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                class_id = int(label.item())
                class_name = self.model.config.id2label[class_id]
                
                detections.append({
                    'bbox': [int(coord) for coord in box.tolist()],
                    'confidence': float(score.item()),
                    'class_id': class_id,
                    'class_name': class_name,
                    'object_id': None 
                })
                logger.debug(f"ObjectDetector:detect - Appended detection: {{'bbox': {[int(coord) for coord in box.tolist()]}, 'confidence': {float(score.item()):.2f}, 'class_id': {class_id}, 'class_name': '{class_name}'}}")
            
            logger.info(f"ObjectDetector detected {len(detections)} objects in frame")
            return detections
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []

if __name__ == '__main__':
    dummy_cfg = {
        'detector_model_name': 'facebook/detr-resnet-50',
        'detector_conf': 0.5 
    }
    
    detector = ObjectDetector(dummy_cfg)
    logger.info("ObjectDetector initialized for testing.")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    logger.info("Dummy frame created.")

    detections = detector.detect(dummy_frame)
    logger.info(f"Detected {len(detections)} objects.")

    if detections:
        for det in detections:
            logger.info(f"  - {det['class_name']} (Conf: {det['confidence']:.2f}) at {det['bbox']}")
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(dummy_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(dummy_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        logger.info("No objects detected in the dummy frame.")

    try:
        cv2.imshow("Object Detection Test", dummy_frame)
        logger.info("Displaying test frame. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        logger.info("Test finished.")
    except cv2.error as e:
        logger.warning(f"Could not display test frame (is a display environment available?): {e}")
        logger.info("Test finished (no display).") 