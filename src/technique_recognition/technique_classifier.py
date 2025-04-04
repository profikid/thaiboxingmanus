"""
Technique Recognition and Classification Module for Thai Boxing Vision App

This module analyzes fighter pose data to recognize and classify Thai boxing techniques,
calculate strike speed, and assess impact/damage.
"""

import cv2
import numpy as np
import logging
import math
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('technique_recognition')

class TechniqueType(Enum):
    """Enum representing different Thai boxing techniques."""
    UNKNOWN = 0
    # Punches
    JAB = 1
    CROSS = 2
    HOOK = 3
    UPPERCUT = 4
    # Kicks
    FRONT_KICK = 5
    ROUNDHOUSE_KICK = 6
    SIDE_KICK = 7
    AXE_KICK = 8
    # Knees
    STRAIGHT_KNEE = 9
    DIAGONAL_KNEE = 10
    FLYING_KNEE = 11
    # Elbows
    HORIZONTAL_ELBOW = 12
    UPPERCUT_ELBOW = 13
    DIAGONAL_ELBOW = 14
    SPINNING_ELBOW = 15
    # Other
    CLINCH = 16
    SWEEP = 17
    BLOCK = 18
    DODGE = 19

class BodyTarget(Enum):
    """Enum representing different body targets for strikes."""
    UNKNOWN = 0
    HEAD = 1
    BODY = 2
    LEGS = 3
    ARMS = 4

@dataclass
class TechniqueEvent:
    """Data class to store information about a detected technique."""
    technique_type: TechniqueType
    fighter_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    speed: float  # in pixels per second
    confidence: float
    target: BodyTarget
    target_fighter_id: Optional[int]
    impact_score: float  # 0-1 scale representing estimated impact
    keypoints: List[np.ndarray]  # List of key pose landmarks during the technique

class PoseFeatureExtractor:
    """
    Extracts relevant features from pose landmarks for technique recognition.
    """
    
    # MediaPipe pose landmark indices
    # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    
    def __init__(self):
        """Initialize the feature extractor."""
        logger.info("PoseFeatureExtractor initialized")
    
    def extract_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract relevant features from pose landmarks.
        
        Args:
            landmarks: Pose landmarks as numpy array (shape: N x 4)
                       Each row contains [x, y, z, visibility]
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Skip if landmarks are not valid
        if landmarks is None or landmarks.shape[0] < 33:
            return features
        
        # Extract joint angles
        features.update(self._extract_joint_angles(landmarks))
        
        # Extract relative positions
        features.update(self._extract_relative_positions(landmarks))
        
        # Extract velocities (if temporal data is available)
        # This would require landmarks from previous frames
        
        return features
    
    def _extract_joint_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract joint angles from pose landmarks.
        
        Args:
            landmarks: Pose landmarks as numpy array
            
        Returns:
            Dictionary of joint angles in degrees
        """
        angles = {}
        
        # Calculate right arm angle (shoulder-elbow-wrist)
        angles['right_arm_angle'] = self._calculate_angle(
            landmarks[self.RIGHT_SHOULDER, :2],
            landmarks[self.RIGHT_ELBOW, :2],
            landmarks[self.RIGHT_WRIST, :2]
        )
        
        # Calculate left arm angle (shoulder-elbow-wrist)
        angles['left_arm_angle'] = self._calculate_angle(
            landmarks[self.LEFT_SHOULDER, :2],
            landmarks[self.LEFT_ELBOW, :2],
            landmarks[self.LEFT_WRIST, :2]
        )
        
        # Calculate right leg angle (hip-knee-ankle)
        angles['right_leg_angle'] = self._calculate_angle(
            landmarks[self.RIGHT_HIP, :2],
            landmarks[self.RIGHT_KNEE, :2],
            landmarks[self.RIGHT_ANKLE, :2]
        )
        
        # Calculate left leg angle (hip-knee-ankle)
        angles['left_leg_angle'] = self._calculate_angle(
            landmarks[self.LEFT_HIP, :2],
            landmarks[self.LEFT_KNEE, :2],
            landmarks[self.LEFT_ANKLE, :2]
        )
        
        # Calculate torso angle (shoulder-hip vertical)
        angles['torso_angle'] = self._calculate_vertical_angle(
            landmarks[self.RIGHT_SHOULDER, :2],
            landmarks[self.LEFT_SHOULDER, :2],
            landmarks[self.RIGHT_HIP, :2],
            landmarks[self.LEFT_HIP, :2]
        )
        
        # Calculate shoulder angle (left to right shoulder horizontal)
        angles['shoulder_angle'] = self._calculate_horizontal_angle(
            landmarks[self.LEFT_SHOULDER, :2],
            landmarks[self.RIGHT_SHOULDER, :2]
        )
        
        # Calculate hip angle (left to right hip horizontal)
        angles['hip_angle'] = self._calculate_horizontal_angle(
            landmarks[self.LEFT_HIP, :2],
            landmarks[self.RIGHT_HIP, :2]
        )
        
        return angles
    
    def _extract_relative_positions(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract relative positions between key body parts.
        
        Args:
            landmarks: Pose landmarks as numpy array
            
        Returns:
            Dictionary of relative positions
        """
        positions = {}
        
        # Calculate distances between key points
        
        # Wrist to face distance (for punch detection)
        positions['right_wrist_to_face'] = self._calculate_distance(
            landmarks[self.RIGHT_WRIST, :2],
            landmarks[self.NOSE, :2]
        )
        
        positions['left_wrist_to_face'] = self._calculate_distance(
            landmarks[self.LEFT_WRIST, :2],
            landmarks[self.NOSE, :2]
        )
        
        # Foot to hip distance (for kick detection)
        positions['right_foot_to_hip'] = self._calculate_distance(
            landmarks[self.RIGHT_FOOT_INDEX, :2],
            landmarks[self.LEFT_HIP, :2]
        )
        
        positions['left_foot_to_hip'] = self._calculate_distance(
            landmarks[self.LEFT_FOOT_INDEX, :2],
            landmarks[self.RIGHT_HIP, :2]
        )
        
        # Knee to torso distance (for knee strike detection)
        positions['right_knee_to_torso'] = self._calculate_distance(
            landmarks[self.RIGHT_KNEE, :2],
            self._midpoint(landmarks[self.LEFT_HIP, :2], landmarks[self.RIGHT_HIP, :2])
        )
        
        positions['left_knee_to_torso'] = self._calculate_distance(
            landmarks[self.LEFT_KNEE, :2],
            self._midpoint(landmarks[self.LEFT_HIP, :2], landmarks[self.RIGHT_HIP, :2])
        )
        
        # Elbow to face distance (for elbow strike detection)
        positions['right_elbow_to_face'] = self._calculate_distance(
            landmarks[self.RIGHT_ELBOW, :2],
            landmarks[self.NOSE, :2]
        )
        
        positions['left_elbow_to_face'] = self._calculate_distance(
            landmarks[self.LEFT_ELBOW, :2],
            landmarks[self.NOSE, :2]
        )
        
        # Calculate relative heights
        
        # Wrist height relative to shoulder (for punch type detection)
        positions['right_wrist_height'] = landmarks[self.RIGHT_WRIST, 1] - landmarks[self.RIGHT_SHOULDER, 1]
        positions['left_wrist_height'] = landmarks[self.LEFT_WRIST, 1] - landmarks[self.LEFT_SHOULDER, 1]
        
        # Foot height relative to knee (for kick type detection)
        positions['right_foot_height'] = landmarks[self.RIGHT_FOOT_INDEX, 1] - landmarks[self.RIGHT_KNEE, 1]
        positions['left_foot_height'] = landmarks[self.LEFT_FOOT_INDEX, 1] - landmarks[self.LEFT_KNEE, 1]
        
        return positions
    
    def calculate_limb_velocity(self, 
                               current_landmarks: np.ndarray, 
                               previous_landmarks: np.ndarray,
                               time_delta: float) -> Dict[str, float]:
        """
        Calculate velocity of key body parts between frames.
        
        Args:
            current_landmarks: Current frame pose landmarks
            previous_landmarks: Previous frame pose landmarks
            time_delta: Time between frames in seconds
            
        Returns:
            Dictionary of velocities in pixels per second
        """
        if current_landmarks is None or previous_landmarks is None or time_delta == 0:
            return {}
        
        velocities = {}
        
        # Calculate wrist velocities (for punch speed)
        velocities['right_wrist_velocity'] = self._calculate_velocity(
            current_landmarks[self.RIGHT_WRIST, :2],
            previous_landmarks[self.RIGHT_WRIST, :2],
            time_delta
        )
        
        velocities['left_wrist_velocity'] = self._calculate_velocity(
            current_landmarks[self.LEFT_WRIST, :2],
            previous_landmarks[self.LEFT_WRIST, :2],
            time_delta
        )
        
        # Calculate foot velocities (for kick speed)
        velocities['right_foot_velocity'] = self._calculate_velocity(
            current_landmarks[self.RIGHT_FOOT_INDEX, :2],
            previous_landmarks[self.RIGHT_FOOT_INDEX, :2],
            time_delta
        )
        
        velocities['left_foot_velocity'] = self._calculate_velocity(
            current_landmarks[self.LEFT_FOOT_INDEX, :2],
            previous_landmarks[self.LEFT_FOOT_INDEX, :2],
            time_delta
        )
        
        # Calculate knee velocities (for knee strike speed)
        velocities['right_knee_velocity'] = self._calculate_velocity(
            current_landmarks[self.RIGHT_KNEE, :2],
            previous_landmarks[self.RIGHT_KNEE, :2],
            time_delta
        )
        
        velocities['left_knee_velocity'] = self._calculate_velocity(
            current_landmarks[self.LEFT_KNEE, :2],
            previous_landmarks[self.LEFT_KNEE, :2],
            time_delta
        )
        
        # Calculate elbow velocities (for elbow strike speed)
        velocities['right_elbow_velocity'] = self._calculate_velocity(
            current_landmarks[self.RIGHT_ELBOW, :2],
            previous_landmarks[self.RIGHT_ELBOW, :2],
            time_delta
        )
        
        velocities['left_elbow_velocity'] = self._calculate_velocity(
            current_landmarks[self.LEFT_ELBOW, :2],
            previous_landmarks[self.LEFT_ELBOW, :2],
            time_delta
        )
        
        return velocities
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            p1, p2, p3: Points in [x, y] format where p2 is the vertex
            
        Returns:
            Angle in degrees
        """
        # Convert to vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate dot product
        dot_product = np.dot(v1, v2)
        
        # Calculate magnitudes
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        # Calculate angle
        cos_angle = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0
        cos_angle = min(1.0, max(-1.0, cos_angle))  # Ensure within valid range
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _calculate_vertical_angle(self, 
                                 right_shoulder: np.ndarray, 
                                 left_shoulder: np.ndarray,
                                 right_hip: np.ndarray, 
                                 left_hip: np.ndarray) -> float:
        """
        Calculate the angle of the torso relative to vertical.
        
        Args:
            right_shoulder, left_shoulder, right_hip, left_hip: Points in [x, y] format
            
        Returns:
            Angle in degrees
        """
        # Calculate midpoints
        shoulder_mid = self._midpoint(right_shoulder, left_shoulder)
        hip_mid = self._midpoint(right_hip, left_hip)
        
        # Calculate torso vector
        torso = shoulder_mid - hip_mid
        
        # Calculate angle with vertical (0, -1)
        vertical = np.array([0, -1])
        
        # Calculate dot product
        dot_product = np.dot(torso, vertical)
        
        # Calculate magnitudes
        mag_torso = np.linalg.norm(torso)
        mag_vertical = np.linalg.norm(vertical)
        
        # Calculate angle
        cos_angle = dot_product / (mag_torso * mag_vertical) if mag_torso * mag_vertical > 0 else 0
        cos_angle = min(1.0, max(-1.0, cos_angle))  # Ensure within valid range
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _calculate_horizontal_angle(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate the angle of a line relative to horizontal.
        
        Args:
            p1, p2: Points in [x, y] format
            
        Returns:
            Angle in degrees
        """
        # Calculate vector
        vector = p2 - p1
        
        # Calculate angle with horizontal (1, 0)
        horizontal = np.array([1, 0])
        
        # Calculate dot product
        dot_product = np.dot(vector, horizontal)
        
        # Calculate magnitudes
        mag_vector = np.linalg.norm(vector)
        mag_horizontal = np.linalg.norm(horizontal)
        
        # Calculate angle
        cos_angle = dot_product / (mag_vector * mag_horizontal) if mag_vector * mag_horizontal > 0 else 0
        cos_angle = min(1.0, max(-1.0, cos_angle))  # Ensure within valid range
        angle_rad = np.arccos(cos_angle)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1, p2: Points in [x, y] format
            
        Returns:
            Distance in pixels
        """
        return np.linalg.norm(p1 - p2)
    
    def _calculate_velocity(self, 
                           current_pos: np.ndarray, 
                           previous_pos: np.ndarray,
                           time_delta: float) -> float:
        """
        Calculate velocity between two positions.
        
        Args:
            current_pos: Current position in [x, y] format
            previous_pos: Previous position in [x, y] format
            time_delta: Time between positions in seconds
            
        Returns:
            Velocity in pixels per second
        """
        if time_delta == 0:
            return 0
        
        distance = self._calculate_distance(current_pos, previous_pos)
        return distance / time_delta
    
    def _midpoint(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Calculate the midpoint between two points.
        
        Args:
            p1, p2: Points in [x, y] format
            
        Returns:
            Midpoint in [x, y] format
        """
        return (p1 + p2) / 2


class TechniqueRecognizer:
    """
    Recognizes Thai boxing techniques based on pose features.
    """
    
    def __init__(self, 
                 feature_extractor: PoseFeatureExtractor,
                 buffer_size: int = 30,
                 velocity_threshold: float = 200,
                 confidence_threshold: float = 0.6):
        """
        Initialize the technique recognizer.
        
        Args:
            feature_extractor: PoseFeatureExtractor instance
            buffer_size: Number of frames to keep in history buffer
            velocity_threshold: Minimum velocity to consider for technique detection
            confidence_threshold: Minimum confidence for technique detection
        """
        self.feature_extractor = feature_extractor
        self.buffer_size = buffer_size
        self.velocity_threshold = velocity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Buffer to store historical pose data for each fighter
        self.pose_history = {}  # fighter_id -> deque of (frame_id, timestamp, landmarks)
        
        # Buffer to store detected techniques
        self.technique_buffer = {}  # fighter_id -> list of in-progress techniques
        
        # Completed techniques
        self.detected_techniques = []  # List of TechniqueEvent objects
        
        logger.info(f"TechniqueRecognizer initialized with velocity_threshold={velocity_threshold}")
    
    def update(self, 
              fighter_id: int, 
              frame_id: int, 
              timestamp: float, 
              landmarks: np.ndarray) -> List[TechniqueEvent]:
        """
        Update the recognizer with new pose data and detect techniques.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            landmarks: Pose landmarks as numpy array
            
        Returns:
            List of newly completed technique events
        """
        # Skip if landmarks are not valid
        if landmarks is None or landmarks.shape[0] < 33:
            return []
        
        # Initialize history buffer for this fighter if not exists
        if fighter_id not in self.pose_history:
            self.pose_history[fighter_id] = deque(maxlen=self.buffer_size)
            self.technique_buffer[fighter_id] = []
        
        # Add current pose to history
        self.pose_history[fighter_id].append((frame_id, timestamp, landmarks))
        
        # Need at least 2 frames to calculate velocity
        if len(self.pose_history[fighter_id]) < 2:
            return []
        
        # Extract features from current pose
        features = self.feature_extractor.extract_features(landmarks)
        
        # Calculate velocities
        current_frame_data = self.pose_history[fighter_id][-1]
        previous_frame_data = self.pose_history[fighter_id][-2]
        
        time_delta = current_frame_data[1] - previous_frame_data[1]
        velocities = self.feature_extractor.calculate_limb_velocity(
            current_frame_data[2], previous_frame_data[2], time_delta
        )
        
        # Combine features and velocities
        all_features = {**features, **velocities}
        
        # Detect techniques
        self._detect_techniques(fighter_id, frame_id, timestamp, landmarks, all_features)
        
        # Update in-progress techniques
        self._update_techniques(fighter_id, frame_id, timestamp)
        
        # Return newly completed techniques
        completed_techniques = []
        
        return completed_techniques
    
    def _detect_techniques(self, 
                          fighter_id: int, 
                          frame_id: int, 
                          timestamp: float,
                          landmarks: np.ndarray, 
                          features: Dict[str, float]) -> None:
        """
        Detect techniques based on pose features.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            landmarks: Pose landmarks as numpy array
            features: Extracted pose features
        """
        # Check for punches
        self._detect_punches(fighter_id, frame_id, timestamp, landmarks, features)
        
        # Check for kicks
        self._detect_kicks(fighter_id, frame_id, timestamp, landmarks, features)
        
        # Check for knees
        self._detect_knees(fighter_id, frame_id, timestamp, landmarks, features)
        
        # Check for elbows
        self._detect_elbows(fighter_id, frame_id, timestamp, landmarks, features)
        
        # Check for clinch
        self._detect_clinch(fighter_id, frame_id, timestamp, landmarks, features)
    
    def _detect_punches(self, 
                       fighter_id: int, 
                       frame_id: int, 
                       timestamp: float,
                       landmarks: np.ndarray, 
                       features: Dict[str, float]) -> None:
        """
        Detect punch techniques.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            landmarks: Pose landmarks as numpy array
            features: Extracted pose features
        """
        # Check right hand punches
        if 'right_wrist_velocity' in features and features['right_wrist_velocity'] > self.velocity_threshold:
            # Determine punch type based on arm angle and wrist position
            punch_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # JAB or CROSS (based on arm extension)
            if features.get('right_arm_angle', 0) > 150:  # Nearly straight arm
                # Determine if it's a jab or cross based on shoulder rotation
                if features.get('shoulder_angle', 0) < 10:  # Minimal shoulder rotation
                    punch_type = TechniqueType.JAB
                    confidence = 0.7
                else:
                    punch_type = TechniqueType.CROSS
                    confidence = 0.8
            
            # HOOK (based on arm angle)
            elif 70 < features.get('right_arm_angle', 0) < 120:  # Bent arm
                punch_type = TechniqueType.HOOK
                confidence = 0.75
            
            # UPPERCUT (based on vertical trajectory)
            elif features.get('right_wrist_height', 0) < -0.1:  # Wrist moving upward relative to shoulder
                punch_type = TechniqueType.UPPERCUT
                confidence = 0.7
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=punch_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['right_wrist_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
        
        # Check left hand punches (similar logic)
        if 'left_wrist_velocity' in features and features['left_wrist_velocity'] > self.velocity_threshold:
            # Determine punch type based on arm angle and wrist position
            punch_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # JAB or CROSS (based on arm extension)
            if features.get('left_arm_angle', 0) > 150:  # Nearly straight arm
                # Determine if it's a jab or cross based on shoulder rotation
                if features.get('shoulder_angle', 0) < 10:  # Minimal shoulder rotation
                    punch_type = TechniqueType.JAB
                    confidence = 0.7
                else:
                    punch_type = TechniqueType.CROSS
                    confidence = 0.8
            
            # HOOK (based on arm angle)
            elif 70 < features.get('left_arm_angle', 0) < 120:  # Bent arm
                punch_type = TechniqueType.HOOK
                confidence = 0.75
            
            # UPPERCUT (based on vertical trajectory)
            elif features.get('left_wrist_height', 0) < -0.1:  # Wrist moving upward relative to shoulder
                punch_type = TechniqueType.UPPERCUT
                confidence = 0.7
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=punch_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['left_wrist_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
    
    def _detect_kicks(self, 
                     fighter_id: int, 
                     frame_id: int, 
                     timestamp: float,
                     landmarks: np.ndarray, 
                     features: Dict[str, float]) -> None:
        """
        Detect kick techniques.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            landmarks: Pose landmarks as numpy array
            features: Extracted pose features
        """
        # Check right leg kicks
        if 'right_foot_velocity' in features and features['right_foot_velocity'] > self.velocity_threshold:
            # Determine kick type based on leg angle and foot position
            kick_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # FRONT_KICK (based on leg extension forward)
            if features.get('right_leg_angle', 0) > 150:  # Nearly straight leg
                kick_type = TechniqueType.FRONT_KICK
                confidence = 0.75
            
            # ROUNDHOUSE_KICK (based on hip rotation and leg angle)
            elif 100 < features.get('right_leg_angle', 0) < 170 and features.get('hip_angle', 0) > 20:
                kick_type = TechniqueType.ROUNDHOUSE_KICK
                confidence = 0.8
            
            # SIDE_KICK (based on hip position and leg angle)
            elif 140 < features.get('right_leg_angle', 0) < 180 and features.get('hip_angle', 0) > 30:
                kick_type = TechniqueType.SIDE_KICK
                confidence = 0.7
            
            # AXE_KICK (based on foot height and downward trajectory)
            elif features.get('right_foot_height', 0) < -0.3:  # Foot high above knee
                kick_type = TechniqueType.AXE_KICK
                confidence = 0.65
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=kick_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['right_foot_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
        
        # Check left leg kicks (similar logic)
        if 'left_foot_velocity' in features and features['left_foot_velocity'] > self.velocity_threshold:
            # Determine kick type based on leg angle and foot position
            kick_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # FRONT_KICK (based on leg extension forward)
            if features.get('left_leg_angle', 0) > 150:  # Nearly straight leg
                kick_type = TechniqueType.FRONT_KICK
                confidence = 0.75
            
            # ROUNDHOUSE_KICK (based on hip rotation and leg angle)
            elif 100 < features.get('left_leg_angle', 0) < 170 and features.get('hip_angle', 0) > 20:
                kick_type = TechniqueType.ROUNDHOUSE_KICK
                confidence = 0.8
            
            # SIDE_KICK (based on hip position and leg angle)
            elif 140 < features.get('left_leg_angle', 0) < 180 and features.get('hip_angle', 0) > 30:
                kick_type = TechniqueType.SIDE_KICK
                confidence = 0.7
            
            # AXE_KICK (based on foot height and downward trajectory)
            elif features.get('left_foot_height', 0) < -0.3:  # Foot high above knee
                kick_type = TechniqueType.AXE_KICK
                confidence = 0.65
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=kick_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['left_foot_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
    
    def _detect_knees(self, 
                     fighter_id: int, 
                     frame_id: int, 
                     timestamp: float,
                     landmarks: np.ndarray, 
                     features: Dict[str, float]) -> None:
        """
        Detect knee techniques.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            landmarks: Pose landmarks as numpy array
            features: Extracted pose features
        """
        # Check right knee strikes
        if 'right_knee_velocity' in features and features['right_knee_velocity'] > self.velocity_threshold:
            # Determine knee strike type
            knee_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # STRAIGHT_KNEE (based on knee position relative to hip)
            if features.get('right_knee_to_torso', 0) < 0.3:  # Knee close to torso
                knee_type = TechniqueType.STRAIGHT_KNEE
                confidence = 0.75
            
            # DIAGONAL_KNEE (based on knee trajectory)
            elif features.get('right_leg_angle', 0) < 120:  # Bent leg
                knee_type = TechniqueType.DIAGONAL_KNEE
                confidence = 0.7
            
            # FLYING_KNEE (based on both feet off ground)
            # This would require checking if both feet are elevated
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=knee_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['right_knee_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
        
        # Check left knee strikes (similar logic)
        if 'left_knee_velocity' in features and features['left_knee_velocity'] > self.velocity_threshold:
            # Determine knee strike type
            knee_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # STRAIGHT_KNEE (based on knee position relative to hip)
            if features.get('left_knee_to_torso', 0) < 0.3:  # Knee close to torso
                knee_type = TechniqueType.STRAIGHT_KNEE
                confidence = 0.75
            
            # DIAGONAL_KNEE (based on knee trajectory)
            elif features.get('left_leg_angle', 0) < 120:  # Bent leg
                knee_type = TechniqueType.DIAGONAL_KNEE
                confidence = 0.7
            
            # FLYING_KNEE (based on both feet off ground)
            # This would require checking if both feet are elevated
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=knee_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['left_knee_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
    
    def _detect_elbows(self, 
                      fighter_id: int, 
                      frame_id: int, 
                      timestamp: float,
                      landmarks: np.ndarray, 
                      features: Dict[str, float]) -> None:
        """
        Detect elbow techniques.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            landmarks: Pose landmarks as numpy array
            features: Extracted pose features
        """
        # Check right elbow strikes
        if 'right_elbow_velocity' in features and features['right_elbow_velocity'] > self.velocity_threshold:
            # Determine elbow strike type
            elbow_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # HORIZONTAL_ELBOW (based on elbow trajectory)
            if 70 < features.get('right_arm_angle', 0) < 120:  # Bent arm
                elbow_type = TechniqueType.HORIZONTAL_ELBOW
                confidence = 0.75
            
            # UPPERCUT_ELBOW (based on upward trajectory)
            elif features.get('right_elbow_to_face', 0) < 0.3:  # Elbow close to face
                elbow_type = TechniqueType.UPPERCUT_ELBOW
                confidence = 0.7
            
            # DIAGONAL_ELBOW (based on arm and torso angles)
            elif 50 < features.get('right_arm_angle', 0) < 100 and features.get('torso_angle', 0) > 20:
                elbow_type = TechniqueType.DIAGONAL_ELBOW
                confidence = 0.7
            
            # SPINNING_ELBOW (based on torso rotation)
            # This would require tracking torso orientation over multiple frames
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=elbow_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['right_elbow_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
        
        # Check left elbow strikes (similar logic)
        if 'left_elbow_velocity' in features and features['left_elbow_velocity'] > self.velocity_threshold:
            # Determine elbow strike type
            elbow_type = TechniqueType.UNKNOWN
            confidence = 0.0
            
            # HORIZONTAL_ELBOW (based on elbow trajectory)
            if 70 < features.get('left_arm_angle', 0) < 120:  # Bent arm
                elbow_type = TechniqueType.HORIZONTAL_ELBOW
                confidence = 0.75
            
            # UPPERCUT_ELBOW (based on upward trajectory)
            elif features.get('left_elbow_to_face', 0) < 0.3:  # Elbow close to face
                elbow_type = TechniqueType.UPPERCUT_ELBOW
                confidence = 0.7
            
            # DIAGONAL_ELBOW (based on arm and torso angles)
            elif 50 < features.get('left_arm_angle', 0) < 100 and features.get('torso_angle', 0) > 20:
                elbow_type = TechniqueType.DIAGONAL_ELBOW
                confidence = 0.7
            
            # SPINNING_ELBOW (based on torso rotation)
            # This would require tracking torso orientation over multiple frames
            
            # If confidence is high enough, start tracking this technique
            if confidence > self.confidence_threshold:
                self._start_technique(
                    fighter_id=fighter_id,
                    technique_type=elbow_type,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    speed=features['left_elbow_velocity'],
                    confidence=confidence,
                    landmarks=landmarks
                )
    
    def _detect_clinch(self, 
                      fighter_id: int, 
                      frame_id: int, 
                      timestamp: float,
                      landmarks: np.ndarray, 
                      features: Dict[str, float]) -> None:
        """
        Detect clinch techniques.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            landmarks: Pose landmarks as numpy array
            features: Extracted pose features
        """
        # Clinch detection would require comparing poses between two fighters
        # This is a placeholder for future implementation
        pass
    
    def _start_technique(self, 
                        fighter_id: int, 
                        technique_type: TechniqueType,
                        frame_id: int, 
                        timestamp: float,
                        speed: float,
                        confidence: float,
                        landmarks: np.ndarray) -> None:
        """
        Start tracking a new technique.
        
        Args:
            fighter_id: ID of the fighter
            technique_type: Type of technique
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            speed: Initial speed of the technique
            confidence: Confidence score for the technique
            landmarks: Pose landmarks as numpy array
        """
        # Check if we're already tracking this technique type
        for technique in self.technique_buffer[fighter_id]:
            if technique['type'] == technique_type and frame_id - technique['last_frame'] < 15:
                # Update existing technique
                technique['last_frame'] = frame_id
                technique['last_time'] = timestamp
                technique['speed'] = max(technique['speed'], speed)
                technique['confidence'] = max(technique['confidence'], confidence)
                technique['keypoints'].append(landmarks)
                return
        
        # Start tracking a new technique
        self.technique_buffer[fighter_id].append({
            'type': technique_type,
            'start_frame': frame_id,
            'last_frame': frame_id,
            'start_time': timestamp,
            'last_time': timestamp,
            'speed': speed,
            'confidence': confidence,
            'keypoints': [landmarks],
            'target': BodyTarget.UNKNOWN,
            'target_fighter_id': None,
            'impact_score': 0.0
        })
        
        logger.info(f"Started tracking {technique_type.name} for fighter {fighter_id}")
    
    def _update_techniques(self, fighter_id: int, frame_id: int, timestamp: float) -> None:
        """
        Update in-progress techniques and finalize completed ones.
        
        Args:
            fighter_id: ID of the fighter
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
        """
        if fighter_id not in self.technique_buffer:
            return
        
        completed_indices = []
        
        # Check each in-progress technique
        for i, technique in enumerate(self.technique_buffer[fighter_id]):
            # If technique hasn't been updated for a while, consider it complete
            if frame_id - technique['last_frame'] > 10:  # Adjust threshold as needed
                # Create a TechniqueEvent
                event = TechniqueEvent(
                    technique_type=technique['type'],
                    fighter_id=fighter_id,
                    start_frame=technique['start_frame'],
                    end_frame=technique['last_frame'],
                    start_time=technique['start_time'],
                    end_time=technique['last_time'],
                    speed=technique['speed'],
                    confidence=technique['confidence'],
                    target=technique['target'],
                    target_fighter_id=technique['target_fighter_id'],
                    impact_score=technique['impact_score'],
                    keypoints=technique['keypoints']
                )
                
                # Add to completed techniques
                self.detected_techniques.append(event)
                
                # Mark for removal
                completed_indices.append(i)
                
                logger.info(f"Completed {technique['type'].name} for fighter {fighter_id}")
        
        # Remove completed techniques (in reverse order to avoid index issues)
        for i in sorted(completed_indices, reverse=True):
            del self.technique_buffer[fighter_id][i]
    
    def get_detected_techniques(self) -> List[TechniqueEvent]:
        """
        Get all detected techniques.
        
        Returns:
            List of detected technique events
        """
        return self.detected_techniques
    
    def clear_detected_techniques(self) -> None:
        """Clear the list of detected techniques."""
        self.detected_techniques = []


class ImpactAssessor:
    """
    Assesses the impact and damage of detected techniques.
    """
    
    def __init__(self):
        """Initialize the impact assessor."""
        logger.info("ImpactAssessor initialized")
    
    def assess_impact(self, 
                     technique: TechniqueEvent, 
                     target_landmarks: Optional[np.ndarray] = None) -> float:
        """
        Assess the impact of a technique.
        
        Args:
            technique: Technique event to assess
            target_landmarks: Pose landmarks of the target fighter (if available)
            
        Returns:
            Impact score between 0 and 1
        """
        # Base impact on technique speed
        impact = min(1.0, technique.speed / 1000)
        
        # Adjust based on technique type
        if technique.technique_type in [TechniqueType.ROUNDHOUSE_KICK, TechniqueType.SIDE_KICK]:
            impact *= 1.5  # Higher impact for powerful kicks
        elif technique.technique_type in [TechniqueType.HOOK, TechniqueType.UPPERCUT]:
            impact *= 1.3  # Higher impact for power punches
        elif technique.technique_type in [TechniqueType.STRAIGHT_KNEE, TechniqueType.FLYING_KNEE]:
            impact *= 1.4  # Higher impact for knees
        elif technique.technique_type in [TechniqueType.HORIZONTAL_ELBOW, TechniqueType.SPINNING_ELBOW]:
            impact *= 1.3  # Higher impact for elbows
        
        # Adjust based on target area
        if technique.target == BodyTarget.HEAD:
            impact *= 1.5  # Higher impact for head strikes
        elif technique.target == BodyTarget.BODY:
            impact *= 1.2  # Medium impact for body strikes
        
        # Cap at 1.0
        impact = min(1.0, impact)
        
        return impact
    
    def determine_target(self, 
                        technique: TechniqueEvent, 
                        target_landmarks: Optional[np.ndarray] = None) -> BodyTarget:
        """
        Determine the target of a technique.
        
        Args:
            technique: Technique event to assess
            target_landmarks: Pose landmarks of the target fighter (if available)
            
        Returns:
            Body target enum
        """
        # If no target landmarks, use heuristics based on technique type
        if target_landmarks is None:
            # Use the last keypoint as the end of the technique
            if not technique.keypoints:
                return BodyTarget.UNKNOWN
            
            landmarks = technique.keypoints[-1]
            
            # For punches, check wrist height
            if technique.technique_type in [TechniqueType.JAB, TechniqueType.CROSS, 
                                          TechniqueType.HOOK, TechniqueType.UPPERCUT]:
                # Extract relevant keypoints
                nose_y = landmarks[PoseFeatureExtractor.NOSE, 1]
                left_hip_y = landmarks[PoseFeatureExtractor.LEFT_HIP, 1]
                right_hip_y = landmarks[PoseFeatureExtractor.RIGHT_HIP, 1]
                left_wrist_y = landmarks[PoseFeatureExtractor.LEFT_WRIST, 1]
                right_wrist_y = landmarks[PoseFeatureExtractor.RIGHT_WRIST, 1]
                
                # Determine which wrist to check based on technique
                wrist_y = right_wrist_y  # Default to right
                if technique.technique_type == TechniqueType.JAB:
                    # Jab is typically with the lead hand (assume left for now)
                    wrist_y = left_wrist_y
                
                # Check wrist position relative to body landmarks
                hip_y = (left_hip_y + right_hip_y) / 2
                
                if wrist_y < nose_y:
                    return BodyTarget.HEAD
                elif wrist_y < hip_y:
                    return BodyTarget.BODY
                else:
                    return BodyTarget.LEGS
            
            # For kicks, check foot height
            elif technique.technique_type in [TechniqueType.FRONT_KICK, TechniqueType.ROUNDHOUSE_KICK,
                                            TechniqueType.SIDE_KICK, TechniqueType.AXE_KICK]:
                # Extract relevant keypoints
                nose_y = landmarks[PoseFeatureExtractor.NOSE, 1]
                left_hip_y = landmarks[PoseFeatureExtractor.LEFT_HIP, 1]
                right_hip_y = landmarks[PoseFeatureExtractor.RIGHT_HIP, 1]
                left_foot_y = landmarks[PoseFeatureExtractor.LEFT_FOOT_INDEX, 1]
                right_foot_y = landmarks[PoseFeatureExtractor.RIGHT_FOOT_INDEX, 1]
                
                # Determine which foot to check
                foot_y = right_foot_y  # Default to right
                
                # Check foot position relative to body landmarks
                hip_y = (left_hip_y + right_hip_y) / 2
                
                if foot_y < nose_y:
                    return BodyTarget.HEAD
                elif foot_y < hip_y:
                    return BodyTarget.BODY
                else:
                    return BodyTarget.LEGS
            
            # For knees, typically body or head
            elif technique.technique_type in [TechniqueType.STRAIGHT_KNEE, 
                                            TechniqueType.DIAGONAL_KNEE, 
                                            TechniqueType.FLYING_KNEE]:
                return BodyTarget.BODY  # Default to body for knees
            
            # For elbows, typically head
            elif technique.technique_type in [TechniqueType.HORIZONTAL_ELBOW, 
                                            TechniqueType.UPPERCUT_ELBOW,
                                            TechniqueType.DIAGONAL_ELBOW, 
                                            TechniqueType.SPINNING_ELBOW]:
                return BodyTarget.HEAD  # Default to head for elbows
        
        # If we have target landmarks, we could do more precise targeting
        # by checking the proximity of the striking limb to different body parts
        
        return BodyTarget.UNKNOWN


class TechniqueRecognitionSystem:
    """
    Complete system for recognizing and classifying Thai boxing techniques.
    """
    
    def __init__(self, 
                 velocity_threshold: float = 200,
                 confidence_threshold: float = 0.6):
        """
        Initialize the technique recognition system.
        
        Args:
            velocity_threshold: Minimum velocity to consider for technique detection
            confidence_threshold: Minimum confidence for technique detection
        """
        self.feature_extractor = PoseFeatureExtractor()
        
        self.technique_recognizer = TechniqueRecognizer(
            feature_extractor=self.feature_extractor,
            velocity_threshold=velocity_threshold,
            confidence_threshold=confidence_threshold
        )
        
        self.impact_assessor = ImpactAssessor()
        
        # Store fighter data for the current frame
        self.current_fighters = {}  # fighter_id -> landmarks
        
        # Store detected techniques
        self.techniques = []
        
        logger.info("TechniqueRecognitionSystem initialized")
    
    def process_fighters(self, 
                        fighter_poses: List[Any],  # List of FighterPose objects
                        frame_id: int, 
                        timestamp: float) -> List[TechniqueEvent]:
        """
        Process fighter poses to detect and classify techniques.
        
        Args:
            fighter_poses: List of fighter poses from tracking system
            frame_id: Current frame ID
            timestamp: Current timestamp in seconds
            
        Returns:
            List of detected technique events
        """
        # Update current fighters
        self.current_fighters = {}
        for pose in fighter_poses:
            self.current_fighters[pose.fighter_id] = pose.landmarks
        
        # Process each fighter
        new_techniques = []
        for pose in fighter_poses:
            # Update technique recognizer
            techniques = self.technique_recognizer.update(
                fighter_id=pose.fighter_id,
                frame_id=frame_id,
                timestamp=timestamp,
                landmarks=pose.landmarks
            )
            
            # Assess impact and determine targets for new techniques
            for technique in techniques:
                # Find potential target fighter
                target_fighter_id = self._find_target_fighter(pose.fighter_id, technique)
                
                # Get target landmarks if available
                target_landmarks = None
                if target_fighter_id in self.current_fighters:
                    target_landmarks = self.current_fighters[target_fighter_id]
                
                # Determine target area
                target = self.impact_assessor.determine_target(technique, target_landmarks)
                technique.target = target
                technique.target_fighter_id = target_fighter_id
                
                # Assess impact
                impact = self.impact_assessor.assess_impact(technique, target_landmarks)
                technique.impact_score = impact
                
                # Add to list of new techniques
                new_techniques.append(technique)
        
        # Add new techniques to the list
        self.techniques.extend(new_techniques)
        
        return new_techniques
    
    def _find_target_fighter(self, fighter_id: int, technique: TechniqueEvent) -> Optional[int]:
        """
        Find the potential target fighter for a technique.
        
        Args:
            fighter_id: ID of the fighter performing the technique
            technique: Technique event
            
        Returns:
            ID of the potential target fighter, or None if not found
        """
        # Simple heuristic: the target is the closest other fighter
        min_distance = float('inf')
        target_id = None
        
        # Get the last keypoint of the technique
        if not technique.keypoints:
            return None
        
        attacker_landmarks = technique.keypoints[-1]
        
        # Calculate attacker's center position
        attacker_center = self._calculate_center(attacker_landmarks)
        
        # Find the closest fighter
        for other_id, other_landmarks in self.current_fighters.items():
            if other_id == fighter_id:
                continue
            
            other_center = self._calculate_center(other_landmarks)
            distance = np.linalg.norm(attacker_center - other_center)
            
            if distance < min_distance:
                min_distance = distance
                target_id = other_id
        
        return target_id
    
    def _calculate_center(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate the center position of a fighter.
        
        Args:
            landmarks: Pose landmarks as numpy array
            
        Returns:
            Center position as [x, y]
        """
        # Use the midpoint between shoulders and hips
        shoulder_mid = (landmarks[PoseFeatureExtractor.LEFT_SHOULDER, :2] + 
                       landmarks[PoseFeatureExtractor.RIGHT_SHOULDER, :2]) / 2
        
        hip_mid = (landmarks[PoseFeatureExtractor.LEFT_HIP, :2] + 
                  landmarks[PoseFeatureExtractor.RIGHT_HIP, :2]) / 2
        
        return (shoulder_mid + hip_mid) / 2
    
    def get_techniques(self) -> List[TechniqueEvent]:
        """
        Get all detected techniques.
        
        Returns:
            List of detected technique events
        """
        return self.techniques
    
    def clear_techniques(self) -> None:
        """Clear the list of detected techniques."""
        self.techniques = []
        self.technique_recognizer.clear_detected_techniques()
    
    def draw_techniques(self, frame: np.ndarray, techniques: List[TechniqueEvent]) -> np.ndarray:
        """
        Draw technique information on the frame.
        
        Args:
            frame: Input frame
            techniques: List of technique events to draw
            
        Returns:
            Annotated frame
        """
        result_frame = frame.copy()
        
        for technique in techniques:
            # Get technique color based on type
            if technique.technique_type in [TechniqueType.JAB, TechniqueType.CROSS, 
                                          TechniqueType.HOOK, TechniqueType.UPPERCUT]:
                color = (0, 0, 255)  # Red for punches
            elif technique.technique_type in [TechniqueType.FRONT_KICK, TechniqueType.ROUNDHOUSE_KICK,
                                            TechniqueType.SIDE_KICK, TechniqueType.AXE_KICK]:
                color = (0, 255, 0)  # Green for kicks
            elif technique.technique_type in [TechniqueType.STRAIGHT_KNEE, 
                                            TechniqueType.DIAGONAL_KNEE, 
                                            TechniqueType.FLYING_KNEE]:
                color = (255, 0, 0)  # Blue for knees
            elif technique.technique_type in [TechniqueType.HORIZONTAL_ELBOW, 
                                            TechniqueType.UPPERCUT_ELBOW,
                                            TechniqueType.DIAGONAL_ELBOW, 
                                            TechniqueType.SPINNING_ELBOW]:
                color = (255, 0, 255)  # Purple for elbows
            else:
                color = (255, 255, 0)  # Yellow for other techniques
            
            # Draw technique name and info
            text = f"{technique.technique_type.name} ({technique.confidence:.2f})"
            cv2.putText(result_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw speed and impact
            speed_text = f"Speed: {technique.speed:.1f} px/s"
            cv2.putText(result_frame, speed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            impact_text = f"Impact: {technique.impact_score:.2f}"
            cv2.putText(result_frame, impact_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw target
            target_text = f"Target: {technique.target.name}"
            cv2.putText(result_frame, target_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result_frame


if __name__ == "__main__":
    # Example usage
    import argparse
    import time
    from pathlib import Path
    
    # Add parent directory to path to import fighter_tracking
    sys.path.append(str(Path(__file__).parent.parent))
    from fighter_tracking.fighter_tracker import FighterTrackingSystem
    
    parser = argparse.ArgumentParser(description="Technique recognition for Thai boxing analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--display", action="store_true", help="Display video while processing")
    
    args = parser.parse_args()
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        exit(1)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize output video if specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize tracking system
    tracking_system = FighterTrackingSystem()
    
    # Initialize technique recognition system
    technique_system = TechniqueRecognitionSystem()
    
    # Process video
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = frame_count / fps
        
        # Process frame for fighter tracking
        result_frame, tracked_poses = tracking_system.process_frame(frame, timestamp)
        
        # Process fighters for technique recognition
        techniques = technique_system.process_fighters(tracked_poses, frame_count, timestamp)
        
        # Draw technique information
        if techniques:
            result_frame = technique_system.draw_techniques(result_frame, techniques)
        
        # Write to output video
        if out:
            out.write(result_frame)
        
        # Display if requested
        if args.display:
            cv2.imshow('Technique Recognition', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"Processed {frame_count} frames ({fps_processing:.2f} FPS)")
            
            # Print detected techniques
            all_techniques = technique_system.get_techniques()
            print(f"Detected {len(all_techniques)} techniques so far")
    
    # Release resources
    cap.release()
    if out:
        out.release()
    tracking_system.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
    
    # Print technique statistics
    all_techniques = technique_system.get_techniques()
    print(f"Detected {len(all_techniques)} techniques total")
    
    # Count by type
    technique_counts = {}
    for technique in all_techniques:
        technique_type = technique.technique_type.name
        if technique_type not in technique_counts:
            technique_counts[technique_type] = 0
        technique_counts[technique_type] += 1
    
    print("Technique breakdown:")
    for technique_type, count in technique_counts.items():
        print(f"  {technique_type}: {count}")
