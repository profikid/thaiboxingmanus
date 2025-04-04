"""
Fighter Detection and Tracking Module for Thai Boxing Vision App

This module handles the detection and tracking of fighters in Thai boxing matches
using MediaPipe's pose estimation capabilities. It identifies fighters, tracks their
movements, and extracts pose data for technique recognition.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fighter_tracking')

class FighterRole(Enum):
    """Enum representing the role of a fighter in a match."""
    UNKNOWN = 0
    RED_CORNER = 1
    BLUE_CORNER = 2

@dataclass
class FighterPose:
    """Data class to store pose information for a fighter."""
    fighter_id: int
    role: FighterRole
    landmarks: np.ndarray  # Shape: (33, 3) for MediaPipe full body pose
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    timestamp: float
    frame_id: int

class PoseEstimator:
    """
    Handles pose estimation using MediaPipe.
    """
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_segmentation: bool = False):
        """
        Initialize the pose estimator.
        
        Args:
            model_complexity: Model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_segmentation: Whether to enable segmentation
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=enable_segmentation
        )
        
        logger.info(f"PoseEstimator initialized with model_complexity={model_complexity}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Process a frame to detect poses.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple containing the annotated frame and pose results
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        # Draw pose landmarks on the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return annotated_frame, results
    
    def get_pose_landmarks(self, results: Any) -> Optional[np.ndarray]:
        """
        Extract pose landmarks from results.
        
        Args:
            results: MediaPipe pose results
            
        Returns:
            Numpy array of landmarks or None if no pose detected
        """
        if not results.pose_landmarks:
            return None
        
        # Convert landmarks to numpy array
        landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                             for lm in results.pose_landmarks.landmark])
        
        return landmarks
    
    def get_bounding_box(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from landmarks.
        
        Args:
            landmarks: Pose landmarks as numpy array
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            Tuple of (x, y, width, height)
        """
        height, width = frame_shape[:2]
        
        # Extract x, y coordinates and scale to pixel values
        x_coords = landmarks[:, 0] * width
        y_coords = landmarks[:, 1] * height
        
        # Calculate bounding box
        x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
        x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def calculate_pose_confidence(self, landmarks: np.ndarray) -> float:
        """
        Calculate overall confidence score for the pose.
        
        Args:
            landmarks: Pose landmarks as numpy array
            
        Returns:
            Confidence score between 0 and 1
        """
        # Use visibility scores as confidence
        visibility_scores = landmarks[:, 3]
        return float(np.mean(visibility_scores))
    
    def release(self):
        """Release resources."""
        self.pose.close()


class FighterDetector:
    """
    Detects and identifies fighters in Thai boxing matches.
    """
    
    def __init__(self, pose_estimator: PoseEstimator):
        """
        Initialize the fighter detector.
        
        Args:
            pose_estimator: PoseEstimator instance
        """
        self.pose_estimator = pose_estimator
        self.frame_count = 0
        logger.info("FighterDetector initialized")
    
    def detect_fighters(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, List[FighterPose]]:
        """
        Detect fighters in a frame.
        
        Args:
            frame: Input frame as numpy array
            timestamp: Frame timestamp in seconds
            
        Returns:
            Tuple containing annotated frame and list of detected fighter poses
        """
        self.frame_count += 1
        
        # Process the frame with pose estimator
        annotated_frame, results = self.pose_estimator.process_frame(frame)
        
        # If no pose detected, return empty list
        if not results.pose_landmarks:
            return annotated_frame, []
        
        # Extract landmarks
        landmarks = self.pose_estimator.get_pose_landmarks(results)
        
        # Calculate bounding box
        bbox = self.pose_estimator.get_bounding_box(landmarks, frame.shape)
        
        # Calculate confidence
        confidence = self.pose_estimator.calculate_pose_confidence(landmarks)
        
        # Create fighter pose object
        fighter_pose = FighterPose(
            fighter_id=1,  # For now, assume single fighter
            role=FighterRole.UNKNOWN,  # Role assignment will be done by tracker
            landmarks=landmarks,
            bounding_box=bbox,
            confidence=confidence,
            timestamp=timestamp,
            frame_id=self.frame_count
        )
        
        return annotated_frame, [fighter_pose]
    
    def draw_fighter_info(self, frame: np.ndarray, fighter_poses: List[FighterPose]) -> np.ndarray:
        """
        Draw fighter information on the frame.
        
        Args:
            frame: Input frame
            fighter_poses: List of detected fighter poses
            
        Returns:
            Annotated frame
        """
        result_frame = frame.copy()
        
        for pose in fighter_poses:
            # Draw bounding box
            x, y, w, h = pose.bounding_box
            
            # Color based on fighter role
            if pose.role == FighterRole.RED_CORNER:
                color = (0, 0, 255)  # Red
            elif pose.role == FighterRole.BLUE_CORNER:
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 0)  # Green
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw fighter ID and confidence
            label = f"Fighter {pose.fighter_id} ({pose.confidence:.2f})"
            cv2.putText(result_frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_frame


class FighterTracker:
    """
    Tracks fighters across frames and assigns consistent IDs and roles.
    """
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_frames_to_skip: int = 10,
                 max_trace_length: int = 30):
        """
        Initialize the fighter tracker.
        
        Args:
            iou_threshold: Intersection over Union threshold for matching
            max_frames_to_skip: Maximum number of frames a fighter can be missing
            max_trace_length: Maximum length of fighter's position history
        """
        self.iou_threshold = iou_threshold
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        
        self.fighters = {}  # Dictionary of tracked fighters
        self.next_fighter_id = 1
        
        # For role assignment
        self.ring_center = None
        self.roles_assigned = False
        
        logger.info(f"FighterTracker initialized with iou_threshold={iou_threshold}")
    
    def update(self, fighter_poses: List[FighterPose]) -> List[FighterPose]:
        """
        Update tracker with new detections.
        
        Args:
            fighter_poses: List of detected fighter poses
            
        Returns:
            Updated list of fighter poses with consistent IDs and roles
        """
        # If no fighters are being tracked yet, initialize with detections
        if not self.fighters:
            for pose in fighter_poses:
                self._add_new_fighter(pose)
            return fighter_poses
        
        # Match detections with existing tracks
        matched_poses = self._match_fighters(fighter_poses)
        
        # Assign roles if not already done
        if not self.roles_assigned and len(self.fighters) >= 2:
            self._assign_fighter_roles()
        
        return matched_poses
    
    def _add_new_fighter(self, pose: FighterPose) -> None:
        """
        Add a new fighter to tracking.
        
        Args:
            pose: Fighter pose to add
        """
        fighter_id = self.next_fighter_id
        self.next_fighter_id += 1
        
        self.fighters[fighter_id] = {
            'pose': pose,
            'frames_since_seen': 0,
            'trace': [pose.bounding_box],
            'first_seen': pose.frame_id
        }
        
        # Update the pose with the new fighter ID
        pose.fighter_id = fighter_id
        
        logger.info(f"Added new fighter with ID {fighter_id}")
    
    def _match_fighters(self, poses: List[FighterPose]) -> List[FighterPose]:
        """
        Match detected poses with existing tracked fighters.
        
        Args:
            poses: List of detected fighter poses
            
        Returns:
            Updated list of fighter poses with consistent IDs
        """
        # Increment frames_since_seen for all fighters
        for fighter_id in self.fighters:
            self.fighters[fighter_id]['frames_since_seen'] += 1
        
        # If no detections, update tracking and return empty list
        if not poses:
            self._remove_lost_fighters()
            return []
        
        matched_poses = []
        unmatched_detections = poses.copy()
        
        # For each tracked fighter, find best matching detection
        for fighter_id, fighter_data in list(self.fighters.items()):
            if not unmatched_detections:
                break
                
            last_bbox = fighter_data['trace'][-1]
            best_match_idx = -1
            best_iou = self.iou_threshold
            
            # Find detection with highest IoU
            for i, pose in enumerate(unmatched_detections):
                iou = self._calculate_iou(last_bbox, pose.bounding_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
            
            # If a match is found, update the fighter
            if best_match_idx >= 0:
                matched_pose = unmatched_detections.pop(best_match_idx)
                matched_pose.fighter_id = fighter_id
                matched_pose.role = fighter_data['pose'].role
                
                # Update fighter data
                self.fighters[fighter_id]['pose'] = matched_pose
                self.fighters[fighter_id]['frames_since_seen'] = 0
                self.fighters[fighter_id]['trace'].append(matched_pose.bounding_box)
                
                # Limit trace length
                if len(self.fighters[fighter_id]['trace']) > self.max_trace_length:
                    self.fighters[fighter_id]['trace'] = self.fighters[fighter_id]['trace'][-self.max_trace_length:]
                
                matched_poses.append(matched_pose)
        
        # Add new fighters for unmatched detections
        for pose in unmatched_detections:
            self._add_new_fighter(pose)
            matched_poses.append(pose)
        
        # Remove fighters that haven't been seen for too long
        self._remove_lost_fighters()
        
        return matched_poses
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, width, height)
            bbox2: Second bounding box (x, y, width, height)
            
        Returns:
            IoU score between 0 and 1
        """
        # Convert to (x1, y1, x2, y2) format
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _remove_lost_fighters(self) -> None:
        """Remove fighters that haven't been seen for too long."""
        for fighter_id in list(self.fighters.keys()):
            if self.fighters[fighter_id]['frames_since_seen'] > self.max_frames_to_skip:
                logger.info(f"Removing fighter {fighter_id} - not seen for {self.fighters[fighter_id]['frames_since_seen']} frames")
                del self.fighters[fighter_id]
    
    def _assign_fighter_roles(self) -> None:
        """
        Assign red and blue corner roles to fighters based on their positions.
        
        In Thai boxing, fighters typically start in their respective corners.
        This method uses the first few frames to determine which fighter is in which corner.
        """
        if len(self.fighters) < 2:
            return
        
        # Get the two fighters with the longest tracking history
        sorted_fighters = sorted(self.fighters.items(), 
                                key=lambda x: x[1]['first_seen'])
        
        if len(sorted_fighters) >= 2:
            fighter1_id, fighter1_data = sorted_fighters[0]
            fighter2_id, fighter2_data = sorted_fighters[1]
            
            # Get average x position from trace
            fighter1_x = np.mean([bbox[0] + bbox[2]/2 for bbox in fighter1_data['trace']])
            fighter2_x = np.mean([bbox[0] + bbox[2]/2 for bbox in fighter2_data['trace']])
            
            # Assign roles based on x position (left is typically red corner)
            if fighter1_x < fighter2_x:
                self.fighters[fighter1_id]['pose'].role = FighterRole.RED_CORNER
                self.fighters[fighter2_id]['pose'].role = FighterRole.BLUE_CORNER
            else:
                self.fighters[fighter1_id]['pose'].role = FighterRole.BLUE_CORNER
                self.fighters[fighter2_id]['pose'].role = FighterRole.RED_CORNER
            
            logger.info(f"Assigned fighter {fighter1_id} to {self.fighters[fighter1_id]['pose'].role}")
            logger.info(f"Assigned fighter {fighter2_id} to {self.fighters[fighter2_id]['pose'].role}")
            
            self.roles_assigned = True
    
    def get_fighter_traces(self) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Get the position history for all tracked fighters.
        
        Returns:
            Dictionary mapping fighter IDs to their position history
        """
        return {fighter_id: data['trace'] for fighter_id, data in self.fighters.items()}
    
    def get_fighter_by_role(self, role: FighterRole) -> Optional[int]:
        """
        Get fighter ID by role.
        
        Args:
            role: Fighter role to look for
            
        Returns:
            Fighter ID or None if not found
        """
        for fighter_id, data in self.fighters.items():
            if data['pose'].role == role:
                return fighter_id
        return None


class FighterTrackingSystem:
    """
    Complete system for detecting and tracking fighters in Thai boxing matches.
    """
    
    def __init__(self, 
                 model_complexity: int = 1,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 iou_threshold: float = 0.3):
        """
        Initialize the fighter tracking system.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            detection_confidence: Minimum confidence for pose detection
            tracking_confidence: Minimum confidence for pose tracking
            iou_threshold: IoU threshold for fighter matching
        """
        self.pose_estimator = PoseEstimator(
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        self.fighter_detector = FighterDetector(self.pose_estimator)
        
        self.fighter_tracker = FighterTracker(
            iou_threshold=iou_threshold
        )
        
        logger.info("FighterTrackingSystem initialized")
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, List[FighterPose]]:
        """
        Process a video frame to detect and track fighters.
        
        Args:
            frame: Input video frame
            timestamp: Frame timestamp in seconds
            
        Returns:
            Tuple containing annotated frame and list of tracked fighter poses
        """
        # Detect fighters in the frame
        annotated_frame, detected_poses = self.fighter_detector.detect_fighters(frame, timestamp)
        
        # Update tracker with new detections
        tracked_poses = self.fighter_tracker.update(detected_poses)
        
        # Draw fighter information on the frame
        result_frame = self.fighter_detector.draw_fighter_info(annotated_frame, tracked_poses)
        
        return result_frame, tracked_poses
    
    def get_fighter_by_role(self, role: FighterRole) -> Optional[int]:
        """
        Get fighter ID by role.
        
        Args:
            role: Fighter role to look for
            
        Returns:
            Fighter ID or None if not found
        """
        return self.fighter_tracker.get_fighter_by_role(role)
    
    def get_fighter_traces(self) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Get the position history for all tracked fighters.
        
        Returns:
            Dictionary mapping fighter IDs to their position history
        """
        return self.fighter_tracker.get_fighter_traces()
    
    def release(self):
        """Release resources."""
        self.pose_estimator.release()
        logger.info("FighterTrackingSystem resources released")


if __name__ == "__main__":
    # Example usage
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Fighter tracking for Thai boxing analysis")
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
    
    # Process video
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = frame_count / fps
        
        # Process frame
        result_frame, tracked_poses = tracking_system.process_frame(frame, timestamp)
        
        # Write to output video
        if out:
            out.write(result_frame)
        
        # Display if requested
        if args.display:
            cv2.imshow('Fighter Tracking', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"Processed {frame_count} frames ({fps_processing:.2f} FPS)")
    
    # Release resources
    cap.release()
    if out:
        out.write(result_frame)
    tracking_system.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
