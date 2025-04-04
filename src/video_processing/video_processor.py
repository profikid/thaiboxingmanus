"""
Video Processor Module for Thai Boxing Vision App

This module handles video input, frame extraction, and preprocessing for the
Thai Boxing Vision App. It supports multiple input sources including file uploads,
camera feeds, and streaming sources.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Generator, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_processor')

class VideoProcessor:
    """
    Handles video input, frame extraction, and preprocessing for Thai boxing analysis.
    """
    
    def __init__(self, 
                 resize_dims: Tuple[int, int] = (640, 480),
                 fps_target: Optional[int] = None,
                 buffer_size: int = 30):
        """
        Initialize the video processor.
        
        Args:
            resize_dims: Target dimensions for resizing frames (width, height)
            fps_target: Target frames per second (None for original video FPS)
            buffer_size: Number of frames to keep in memory buffer
        """
        self.resize_dims = resize_dims
        self.fps_target = fps_target
        self.buffer_size = buffer_size
        self.cap = None
        self.frame_buffer = []
        self.source_info = {}
        logger.info(f"VideoProcessor initialized with resize_dims={resize_dims}, fps_target={fps_target}")
    
    def open_video(self, source: Union[str, int]) -> bool:
        """
        Open a video source.
        
        Args:
            source: Path to video file, URL, or camera index
            
        Returns:
            bool: True if video was opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                return False
            
            # Get video properties
            self.source_info = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'source': source
            }
            
            # Use original FPS if target not specified
            if self.fps_target is None:
                self.fps_target = self.source_info['fps']
                
            logger.info(f"Video opened successfully: {self.source_info}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video source {source}: {str(e)}")
            return False
    
    def get_source_info(self) -> Dict:
        """
        Get information about the current video source.
        
        Returns:
            Dict: Video properties including dimensions, FPS, and frame count
        """
        return self.source_info
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a video frame for analysis.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        if frame is None:
            return None
        
        # Resize frame
        if self.resize_dims:
            frame = cv2.resize(frame, self.resize_dims)
        
        # Additional preprocessing can be added here:
        # - Color normalization
        # - Noise reduction
        # - Contrast enhancement
        
        return frame
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame (if successful)
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Attempted to read frame but no video is open")
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to buffer
        self._update_buffer(processed_frame)
        
        return ret, processed_frame
    
    def _update_buffer(self, frame: np.ndarray) -> None:
        """
        Update the frame buffer with a new frame.
        
        Args:
            frame: New frame to add to buffer
        """
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
    
    def get_buffer(self) -> List[np.ndarray]:
        """
        Get the current frame buffer.
        
        Returns:
            List[np.ndarray]: List of frames in the buffer
        """
        return self.frame_buffer
    
    def extract_frames(self, 
                       start_time: Optional[float] = None, 
                       end_time: Optional[float] = None,
                       step: int = 1) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from the video within a time range.
        
        Args:
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end of video)
            step: Frame step (1 = every frame, 2 = every other frame, etc.)
            
        Yields:
            np.ndarray: Extracted frames
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Attempted to extract frames but no video is open")
            return
        
        # Convert times to frame numbers
        fps = self.source_info['fps']
        start_frame = 0 if start_time is None else int(start_time * fps)
        end_frame = self.source_info['frame_count'] if end_time is None else int(end_time * fps)
        
        # Seek to start position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if (current_frame - start_frame) % step == 0:
                yield self.preprocess_frame(frame)
                
            current_frame += 1
    
    def release(self) -> None:
        """
        Release the video capture resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.frame_buffer = []
            logger.info("Video resource released")
    
    def __del__(self):
        """
        Destructor to ensure resources are released.
        """
        self.release()


class VideoFrameExtractor:
    """
    Specialized class for extracting and managing frames from videos
    with focus on Thai boxing analysis requirements.
    """
    
    def __init__(self, 
                 processor: VideoProcessor,
                 sample_rate: float = 1.0,
                 round_detection: bool = True):
        """
        Initialize the frame extractor.
        
        Args:
            processor: VideoProcessor instance
            sample_rate: Fraction of frames to process (1.0 = all frames)
            round_detection: Whether to attempt to detect rounds automatically
        """
        self.processor = processor
        self.sample_rate = sample_rate
        self.round_detection = round_detection
        self.rounds = []  # List of (start_frame, end_frame) tuples for detected rounds
        logger.info(f"VideoFrameExtractor initialized with sample_rate={sample_rate}")
    
    def detect_rounds(self) -> List[Tuple[int, int]]:
        """
        Attempt to detect rounds in the video based on visual cues.
        
        Returns:
            List[Tuple[int, int]]: List of (start_frame, end_frame) for each detected round
        """
        # This is a placeholder for round detection logic
        # In a real implementation, this would analyze the video for round cards,
        # bell sounds, or significant changes in activity
        
        if not self.round_detection:
            return []
            
        # For now, we'll just divide the video into standard 3-minute rounds
        # with 1-minute breaks, starting after a 10-second pre-fight buffer
        
        source_info = self.processor.get_source_info()
        fps = source_info['fps']
        total_frames = source_info['frame_count']
        
        # Standard Muay Thai timing: 3-minute rounds, 1-minute breaks
        round_frames = int(3 * 60 * fps)  # 3 minutes in frames
        break_frames = int(1 * 60 * fps)  # 1 minute in frames
        pre_fight_frames = int(10 * fps)  # 10 seconds in frames
        
        rounds = []
        start_frame = pre_fight_frames
        
        while start_frame < total_frames:
            end_frame = min(start_frame + round_frames, total_frames)
            rounds.append((start_frame, end_frame))
            
            # Skip the break
            start_frame = end_frame + break_frames
        
        self.rounds = rounds
        logger.info(f"Detected {len(rounds)} rounds in the video")
        return rounds
    
    def extract_round_frames(self, round_idx: int, step: int = 1) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from a specific round.
        
        Args:
            round_idx: Index of the round to extract
            step: Frame step (1 = every frame, 2 = every other frame, etc.)
            
        Yields:
            np.ndarray: Frames from the specified round
        """
        if not self.rounds:
            self.detect_rounds()
            
        if round_idx < 0 or round_idx >= len(self.rounds):
            logger.error(f"Invalid round index: {round_idx}")
            return
            
        start_frame, end_frame = self.rounds[round_idx]
        fps = self.processor.get_source_info()['fps']
        
        # Convert frame numbers to times
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Use the processor to extract frames
        yield from self.processor.extract_frames(start_time, end_time, step)
    
    def extract_all_rounds(self, step: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from all rounds, yielding round index with each frame.
        
        Args:
            step: Frame step (1 = every frame, 2 = every other frame, etc.)
            
        Yields:
            Tuple[int, np.ndarray]: Round index and frame
        """
        if not self.rounds:
            self.detect_rounds()
            
        for round_idx in range(len(self.rounds)):
            for frame in self.extract_round_frames(round_idx, step):
                yield round_idx, frame
    
    def get_round_info(self) -> List[Dict]:
        """
        Get information about detected rounds.
        
        Returns:
            List[Dict]: List of dictionaries with round information
        """
        if not self.rounds:
            self.detect_rounds()
            
        fps = self.processor.get_source_info()['fps']
        
        round_info = []
        for i, (start_frame, end_frame) in enumerate(self.rounds):
            round_info.append({
                'round_number': i + 1,
                'start_time': start_frame / fps,
                'end_time': end_frame / fps,
                'duration': (end_frame - start_frame) / fps
            })
            
        return round_info


def create_sample_frames(output_dir: str, video_path: str, interval: int = 30) -> List[str]:
    """
    Create sample frames from a video file at regular intervals.
    
    Args:
        output_dir: Directory to save sample frames
        video_path: Path to the video file
        interval: Interval in seconds between sample frames
        
    Returns:
        List[str]: Paths to the saved sample frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processor = VideoProcessor()
    if not processor.open_video(video_path):
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    source_info = processor.get_source_info()
    fps = source_info['fps']
    frame_interval = int(fps * interval)
    
    saved_frames = []
    frame_count = 0
    
    while True:
        ret, frame = processor.read_frame()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = str(output_path / f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
            logger.info(f"Saved sample frame: {frame_path}")
            
        frame_count += 1
    
    processor.release()
    return saved_frames


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Video processing for Thai boxing analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Interval in seconds between sample frames")
    
    args = parser.parse_args()
    
    print(f"Processing video: {args.video}")
    print(f"Saving sample frames to: {args.output}")
    
    sample_frames = create_sample_frames(args.output, args.video, args.interval)
    print(f"Created {len(sample_frames)} sample frames")
