"""Kalman Filter for Motion Prediction"""

import numpy as np
from typing import Tuple


class KalmanFilter:
    """Kalman Filter for tracking object motion"""
    
    def __init__(
        self,
        initial_position: Tuple[int, int],
        dt: float = 1.0,
        process_variance: float = 0.01,
        measurement_variance: float = 1.0
    ):
        """Initialize Kalman Filter
        
        Args:
            initial_position: (x, y) initial position
            dt: Time step
            process_variance: Process noise covariance
            measurement_variance: Measurement noise covariance
        """
        self.dt = dt
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # State vector: [x, y, vx, vy]
        self.state = np.array([
            float(initial_position[0]),
            float(initial_position[1]),
            0.0,  # velocity x
            0.0   # velocity y
        ], dtype=np.float32)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.Q = np.eye(4, dtype=np.float32) * process_variance
        self.R = np.eye(2, dtype=np.float32) * measurement_variance
    
    def predict(self) -> Tuple[float, float]:
        """Predict next position
        
        Returns:
            (x, y) predicted position
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (int(self.state[0]), int(self.state[1]))
    
    def update(self, measurement: Tuple[float, float]):
        """Update state with measurement
        
        Args:
            measurement: (x, y) measured position
        """
        z = np.array([measurement[0], measurement[1]], dtype=np.float32)
        
        # Innovation
        y = z - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + (K @ y)
        
        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.state.copy()
    
    def get_position(self) -> Tuple[int, int]:
        """Get current position"""
        return (int(self.state[0]), int(self.state[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity"""
        return (float(self.state[2]), float(self.state[3]))
