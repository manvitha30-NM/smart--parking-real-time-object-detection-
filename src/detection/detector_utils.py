"""Detection utilities and helper functions"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class BBox:
    """Bounding box representation"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate Intersection over Union (IoU)
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        
    Returns:
        IoU value (0-1)
    """
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
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_centroid_distance(center1: Tuple, center2: Tuple) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def is_point_in_polygon(point: Tuple[int, int], polygon: List[Tuple]) -> bool:
    """Check if point is inside polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def scale_bbox(bbox: Tuple, scale_x: float, scale_y: float) -> Tuple:
    """Scale bounding box coordinates"""
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y)
    )


def clip_bbox(bbox: Tuple, width: int, height: int) -> Tuple:
    """Clip bounding box to frame boundaries"""
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, width)),
        max(0, min(y1, height)),
        max(0, min(x2, width)),
        max(0, min(y2, height))
    )
