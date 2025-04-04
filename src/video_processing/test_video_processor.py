"""
Test script for the video processor module.

This script demonstrates the functionality of the VideoProcessor and VideoFrameExtractor
classes by processing a sample video and extracting frames.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to the path to import the video_processor module
sys.path.append(str(Path(__file__).parent.parent))
from video_processing.video_processor import VideoProcessor, VideoFrameExtractor

def display_frames(frames, title="Extracted Frames", cols=3, figsize=(15, 10)):
    """
    Display a grid of frames using matplotlib.
    
    Args:
        frames: List of frames to display
        title: Title for the figure
        cols: Number of columns in the grid
        figsize: Figure size (width, height)
    """
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Convert to 1D array for easier indexing
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, frame in enumerate(frames):
        row, col = i // cols, i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_frame)
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(frames), rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "sample_frames.png")
    
    plt.show()

def test_video_processor(video_path):
    """
    Test the VideoProcessor class by opening a video and extracting frames.
    
    Args:
        video_path: Path to the test video
    """
    print(f"Testing VideoProcessor with video: {video_path}")
    
    # Initialize the processor
    processor = VideoProcessor(resize_dims=(640, 480))
    
    # Open the video
    if not processor.open_video(video_path):
        print("Failed to open video")
        return
    
    # Print video information
    info = processor.get_source_info()
    print(f"Video info: {info}")
    
    # Extract some frames
    frames = []
    for i in range(10):  # Get 10 frames
        ret, frame = processor.read_frame()
        if not ret:
            break
        frames.append(frame)
    
    # Display the frames
    if frames:
        display_frames(frames, "Sample Frames from VideoProcessor")
    else:
        print("No frames extracted")
    
    # Release resources
    processor.release()

def test_frame_extractor(video_path):
    """
    Test the VideoFrameExtractor class by detecting rounds and extracting frames.
    
    Args:
        video_path: Path to the test video
    """
    print(f"Testing VideoFrameExtractor with video: {video_path}")
    
    # Initialize the processor and extractor
    processor = VideoProcessor(resize_dims=(640, 480))
    if not processor.open_video(video_path):
        print("Failed to open video")
        return
    
    extractor = VideoFrameExtractor(processor, sample_rate=0.5)
    
    # Detect rounds
    rounds = extractor.detect_rounds()
    print(f"Detected {len(rounds)} rounds")
    
    # Get round information
    round_info = extractor.get_round_info()
    for i, info in enumerate(round_info):
        print(f"Round {i+1}: {info}")
    
    # Extract frames from the first round
    if rounds:
        frames = list(extractor.extract_round_frames(0, step=30))[:9]  # Get up to 9 frames
        
        if frames:
            display_frames(frames, f"Sample Frames from Round 1")
        else:
            print("No frames extracted from round")
    
    # Release resources
    processor.release()

def main():
    """
    Main function to run the tests.
    """
    # Check if a video path was provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Use a default video if available
        default_videos = [
            "/home/ubuntu/thai_boxing_vision_app/test_data/sample_match.mp4",
            "/home/ubuntu/thai_boxing_vision_app/test_data/muay_thai_sample.mp4"
        ]
        
        video_path = None
        for path in default_videos:
            if os.path.exists(path):
                video_path = path
                break
        
        if not video_path:
            print("No video file provided and no default video found.")
            print("Usage: python test_video_processor.py [video_path]")
            return
    
    # Run the tests
    test_video_processor(video_path)
    test_frame_extractor(video_path)

if __name__ == "__main__":
    main()
