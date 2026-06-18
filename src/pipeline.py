"""Main pipeline combining detection and tracking"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from loguru import logger
from datetime import datetime

from src.detection import YOLODetector, Detection
from src.tracking import DeepSORT


class DetectionTrackingPipeline:
    """Combined detection and tracking pipeline"""
    
    def __init__(
        self,
        detector: YOLODetector,
        tracker: DeepSORT
    ):
        """Initialize pipeline
        
        Args:
            detector: YOLO detector instance
            tracker: Deep SORT tracker instance
        """
        self.detector = detector
        self.tracker = tracker
        self.frame_count = 0
        self.fps = 0
        
        logger.info("Detection-Tracking pipeline initialized")
    
    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[List, np.ndarray]:
        """Process frame through detection and tracking
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            (tracked_objects, annotated_frame)
        """
        self.frame_count += 1
        
        # Detection
        detections = self.detector.detect(frame)
        logger.debug(f"Frame {self.frame_count}: {len(detections)} detections")
        
        # Tracking
        tracked_objects = self.tracker.update(detections, frame)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, detections, tracked_objects)
        
        return tracked_objects, annotated_frame
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracked_objects: List[Tuple]
    ) -> np.ndarray:
        """Annotate frame with detections and tracks"""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw tracked objects
        for track_id, bbox, class_name, hits in tracked_objects:
            x1, y1, x2, y2 = bbox
            
            # Confirmed tracks in green, provisional in yellow
            color = (0, 255, 0) if hits >= 3 else (0, 255, 255)
            thickness = 2
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw ID and class
            label = f"ID:{track_id} {class_name}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Draw center
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(result, center, 3, color, -1)
        
        # Add frame info
        info = [
            f"Frame: {self.frame_count}",
            f"Detections: {len(detections)}",
            f"Tracks: {len(tracked_objects)}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        for i, text in enumerate(info):
            cv2.putText(
                result,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
        
        return result
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "frame_count": self.frame_count,
            "detector_stats": self.detector.get_stats(),
            "tracker_stats": self.tracker.get_stats()
        }
