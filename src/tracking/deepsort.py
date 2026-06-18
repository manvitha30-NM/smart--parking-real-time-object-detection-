"""Deep SORT Tracking Implementation"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
from .kalman_filter import KalmanFilter


@dataclass
class TrackedObject:
    """Tracked object representation"""
    track_id: int
    detections: List[Tuple] = field(default_factory=list)
    kalman_filter: Optional[KalmanFilter] = None
    hits: int = 0
    age: int = 0
    class_name: str = ""
    time_since_update: int = 0
    
    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed"""
        return self.hits >= 3
    
    def update_state(self):
        """Update track state"""
        self.age += 1
        self.time_since_update += 1


class DeepSORT:
    """Deep SORT Multi-Object Tracker"""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.5
    ):
        """Initialize Deep SORT tracker
        
        Args:
            max_age: Maximum frames to track without detection
            min_hits: Minimum detections before confirming track
            max_iou_distance: Max IoU for matching
            max_cosine_distance: Max cosine distance for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        logger.info(
            f"Deep SORT initialized - max_age={max_age}, "
            f"min_hits={min_hits}"
        )
    
    def update(
        self,
        detections: List,
        frame: np.ndarray
    ) -> List[Tuple]:
        """Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            frame: Current frame
            
        Returns:
            List of (track_id, bbox, class_name) for confirmed tracks
        """
        self.frame_count += 1
        
        # Predict new locations
        for track_id, track in self.tracks.items():
            if track.kalman_filter:
                track.kalman_filter.predict()
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._match_detections(
            detections
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matched:
            detection = detections[det_idx]
            track = list(self.tracks.values())[trk_idx]
            
            track.detections.append(detection.bbox)
            track.hits += 1
            track.time_since_update = 0
            track.class_name = detection.class_name
            
            # Update Kalman filter
            if track.kalman_filter:
                x, y = detection.center
                track.kalman_filter.update([x, y])
            else:
                track.kalman_filter = KalmanFilter(detection.center)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            new_track = TrackedObject(
                track_id=self.next_track_id,
                class_name=detection.class_name,
                kalman_filter=KalmanFilter(detection.center)
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Update unmatched tracks
        for trk_idx in unmatched_trks:
            track = list(self.tracks.values())[trk_idx]
            track.time_since_update += 1
        
        # Remove old tracks
        self.tracks = {
            tid: track for tid, track in self.tracks.items()
            if track.time_since_update < self.max_age
        }
        
        # Return confirmed tracks
        results = []
        for track_id, track in self.tracks.items():
            if track.is_confirmed and track.detections:
                results.append((
                    track_id,
                    track.detections[-1],
                    track.class_name,
                    track.hits
                ))
        
        return results
    
    def _match_detections(self, detections: List) -> Tuple:
        """Match detections to tracks using IoU
        
        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), []
        
        # Calculate IoU matrix
        track_list = list(self.tracks.values())
        iou_matrix = np.zeros((len(detections), len(track_list)))
        
        for det_idx, detection in enumerate(detections):
            for trk_idx, track in enumerate(track_list):
                if track.detections:
                    iou = self._calculate_iou(
                        detection.bbox,
                        track.detections[-1]
                    )
                    iou_matrix[det_idx, trk_idx] = iou
        
        # Hungarian algorithm for matching
        matched_pairs = []
        matched_dets = set()
        matched_trks = set()
        
        for det_idx in range(len(detections)):
            best_iou = -1
            best_trk = -1
            
            for trk_idx in range(len(track_list)):
                if trk_idx not in matched_trks:
                    if iou_matrix[det_idx, trk_idx] > best_iou:
                        if iou_matrix[det_idx, trk_idx] > self.max_iou_distance:
                            best_iou = iou_matrix[det_idx, trk_idx]
                            best_trk = trk_idx
            
            if best_trk >= 0:
                matched_pairs.append((det_idx, best_trk))
                matched_dets.add(det_idx)
                matched_trks.add(best_trk)
        
        unmatched_dets = list(set(range(len(detections))) - matched_dets)
        unmatched_trks = list(set(range(len(track_list))) - matched_trks)
        
        return matched_pairs, unmatched_dets, unmatched_trks
    
    @staticmethod
    def _calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        confirmed_tracks = sum(1 for t in self.tracks.values() if t.is_confirmed)
        return {
            "frame_count": self.frame_count,
            "total_tracks": len(self.tracks),
            "confirmed_tracks": confirmed_tracks,
            "next_track_id": self.next_track_id
        }
