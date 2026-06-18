"""YOLO Detection Engine - YOLOv5/v8 Integration"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Detection:
    """Detection result container"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int


class YOLODetector:
    """YOLO v5/v8 Object Detection Engine"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cuda"
    ):
        """Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            input_size: Model input size
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Load model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                       path=model_path, force_reload=False)
            self.model.to(self.device)
            self.model.conf = confidence_threshold
            self.model.iou = nms_threshold
            logger.info(f"YOLO model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Class mapping
        self.class_names = {
            0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle',
            5: 'car', 7: 'truck', 9: 'bus'  # COCO classes
        }
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect vehicles in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of Detection objects
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []
        
        try:
            # Run inference
            results = self.model(frame, size=self.input_size, verbose=False)
            
            detections = []
            predictions = results.xyxy[0].cpu().numpy()
            
            for pred in predictions:
                x1, y1, x2, y2, conf, cls_id = pred
                x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
                
                # Filter vehicle classes only
                if cls_id not in [0, 1, 2, 3, 5, 7, 9]:  # Vehicle classes
                    continue
                
                # Calculate metrics
                w, h = x2 - x1, y2 - y1
                area = w * h
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                detection = Detection(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, 'unknown'),
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area=area
                )
                detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} vehicles")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        draw_labels: bool = True
    ) -> np.ndarray:
        """Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            draw_labels: Whether to draw labels
            
        Returns:
            Frame with drawn detections
        """
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)  # Green
            thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            cv2.circle(result, det.center, 3, color, -1)
            
            # Draw label
            if draw_labels:
                label = f"{det.class_name} {det.confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > 20 else y1 + 20
                
                cv2.rectangle(
                    result,
                    (text_x, text_y - text_size[1] - 4),
                    (text_x + text_size[0] + 4, text_y + 4),
                    color,
                    -1
                )
                cv2.putText(
                    result,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness
                )
        
        return result
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            "model_device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "input_size": self.input_size
        }
