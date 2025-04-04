"""
Test script for the fighter tracking module.

This script demonstrates the functionality of the fighter tracking system
by processing a sample video and visualizing the results.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Add the parent directory to the path to import the fighter_tracking module
sys.path.append(str(Path(__file__).parent.parent))
from fighter_tracking.fighter_tracker import FighterTrackingSystem, FighterRole

def test_fighter_tracking(video_path, output_dir=None, display=False, save_frames=False):
    """
    Test the fighter tracking system on a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output frames (optional)
        display: Whether to display the video while processing
        save_frames: Whether to save sample frames
    """
    print(f"Testing fighter tracking with video: {video_path}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize tracking system
    tracking_system = FighterTrackingSystem(
        model_complexity=1,
        detection_confidence=0.5,
        tracking_confidence=0.5
    )
    
    # Process video
    frame_count = 0
    saved_frames = []
    processing_times = []
    
    # Sample frames to save (at 2-second intervals)
    sample_interval = int(fps * 2)
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = frame_count / fps
        
        # Process frame and measure time
        frame_start = time.time()
        result_frame, tracked_poses = tracking_system.process_frame(frame, timestamp)
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)
        
        # Save sample frames
        if save_frames and frame_count % sample_interval == 0:
            if output_dir:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, result_frame)
                saved_frames.append(result_frame)
                print(f"Saved frame {frame_count} to {frame_path}")
        
        # Display if requested
        if display:
            cv2.imshow('Fighter Tracking', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} FPS)")
            
            # Get fighter traces
            traces = tracking_system.get_fighter_traces()
            print(f"Tracking {len(traces)} fighters")
            
            # Get fighters by role
            red_fighter = tracking_system.get_fighter_by_role(FighterRole.RED_CORNER)
            blue_fighter = tracking_system.get_fighter_by_role(FighterRole.BLUE_CORNER)
            
            if red_fighter:
                print(f"Red corner fighter: {red_fighter}")
            if blue_fighter:
                print(f"Blue corner fighter: {blue_fighter}")
    
    # Release resources
    cap.release()
    tracking_system.release()
    cv2.destroyAllWindows()
    
    # Calculate processing statistics
    total_time = time.time() - start_time
    avg_time_per_frame = np.mean(processing_times)
    avg_fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0
    
    print(f"Processing complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per frame: {avg_time_per_frame*1000:.2f} ms")
    print(f"  Average FPS: {avg_fps:.2f}")
    
    # Display saved frames
    if saved_frames and len(saved_frames) > 0:
        display_frames(saved_frames, "Fighter Tracking Results")

def display_frames(frames, title="Fighter Tracking Results", cols=3, figsize=(15, 10)):
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
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    
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
    plt.savefig(output_dir / "fighter_tracking_results.png")
    
    plt.show()

def main():
    """
    Main function to run the test.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fighter tracking system")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--display", action="store_true", help="Display video while processing")
    parser.add_argument("--save-frames", action="store_true", help="Save sample frames")
    
    args = parser.parse_args()
    
    # Check if a video path was provided
    if args.video:
        video_path = args.video
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
            print("Usage: python test_fighter_tracking.py --video [video_path]")
            return
    
    # Run the test
    test_fighter_tracking(
        video_path=video_path,
        output_dir=args.output,
        display=args.display,
        save_frames=args.save_frames
    )

if __name__ == "__main__":
    main()
