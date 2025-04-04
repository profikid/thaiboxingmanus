"""
Test script for the technique recognition module.

This script demonstrates the functionality of the technique recognition system
by processing a sample video and visualizing the results.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Add the parent directory to the path to import the modules
sys.path.append(str(Path(__file__).parent.parent))
from fighter_tracking.fighter_tracker import FighterTrackingSystem
from technique_recognition.technique_classifier import (
    TechniqueRecognitionSystem, 
    TechniqueType, 
    BodyTarget
)

def test_technique_recognition(video_path, output_dir=None, display=False, save_frames=False):
    """
    Test the technique recognition system on a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output frames (optional)
        display: Whether to display the video while processing
        save_frames: Whether to save sample frames
    """
    print(f"Testing technique recognition with video: {video_path}")
    
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
    
    # Initialize technique recognition system
    technique_system = TechniqueRecognitionSystem(
        velocity_threshold=150,  # Lower threshold for testing
        confidence_threshold=0.5  # Lower threshold for testing
    )
    
    # Process video
    frame_count = 0
    saved_frames = []
    processing_times = []
    
    # Sample frames to save (at 2-second intervals)
    sample_interval = int(fps * 2)
    
    # Technique statistics
    technique_counts = {}
    technique_by_target = {}
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = frame_count / fps
        
        # Process frame for fighter tracking
        frame_start = time.time()
        result_frame, tracked_poses = tracking_system.process_frame(frame, timestamp)
        
        # Process fighters for technique recognition
        techniques = technique_system.process_fighters(tracked_poses, frame_count, timestamp)
        
        # Draw technique information
        if techniques:
            result_frame = technique_system.draw_techniques(result_frame, techniques)
            
            # Update statistics
            for technique in techniques:
                technique_type = technique.technique_type.name
                if technique_type not in technique_counts:
                    technique_counts[technique_type] = 0
                technique_counts[technique_type] += 1
                
                target = technique.target.name
                if target not in technique_by_target:
                    technique_by_target[target] = 0
                technique_by_target[target] += 1
        
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
            cv2.imshow('Technique Recognition', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} FPS)")
            
            # Print detected techniques
            all_techniques = technique_system.get_techniques()
            print(f"Detected {len(all_techniques)} techniques so far")
    
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
    
    # Print technique statistics
    all_techniques = technique_system.get_techniques()
    print(f"Detected {len(all_techniques)} techniques total")
    
    print("Technique breakdown:")
    for technique_type, count in technique_counts.items():
        print(f"  {technique_type}: {count}")
    
    print("Target breakdown:")
    for target, count in technique_by_target.items():
        print(f"  {target}: {count}")
    
    # Display saved frames
    if saved_frames and len(saved_frames) > 0:
        display_frames(saved_frames, "Technique Recognition Results")
    
    # Create and save technique distribution chart
    if technique_counts:
        create_technique_chart(technique_counts, output_dir)
    
    # Create and save target distribution chart
    if technique_by_target:
        create_target_chart(technique_by_target, output_dir)

def display_frames(frames, title="Technique Recognition Results", cols=3, figsize=(15, 10)):
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
    plt.savefig(output_dir / "technique_recognition_results.png")
    
    plt.show()

def create_technique_chart(technique_counts, output_dir=None):
    """
    Create a bar chart of technique distribution.
    
    Args:
        technique_counts: Dictionary mapping technique types to counts
        output_dir: Directory to save the chart
    """
    plt.figure(figsize=(12, 6))
    
    # Sort techniques by count
    sorted_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
    techniques = [t[0] for t in sorted_techniques]
    counts = [t[1] for t in sorted_techniques]
    
    # Create bar chart
    bars = plt.bar(techniques, counts)
    
    # Color bars by technique category
    colors = {
        'JAB': 'red', 'CROSS': 'red', 'HOOK': 'red', 'UPPERCUT': 'red',  # Punches
        'FRONT_KICK': 'green', 'ROUNDHOUSE_KICK': 'green', 'SIDE_KICK': 'green', 'AXE_KICK': 'green',  # Kicks
        'STRAIGHT_KNEE': 'blue', 'DIAGONAL_KNEE': 'blue', 'FLYING_KNEE': 'blue',  # Knees
        'HORIZONTAL_ELBOW': 'purple', 'UPPERCUT_ELBOW': 'purple', 'DIAGONAL_ELBOW': 'purple', 'SPINNING_ELBOW': 'purple',  # Elbows
    }
    
    for i, bar in enumerate(bars):
        technique = techniques[i]
        bar.set_color(colors.get(technique, 'gray'))
    
    plt.title('Technique Distribution')
    plt.xlabel('Technique Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        output_path = Path(output_dir) / "technique_distribution.png"
    else:
        output_path = Path("./output") / "technique_distribution.png"
    
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    
    plt.show()

def create_target_chart(target_counts, output_dir=None):
    """
    Create a pie chart of target distribution.
    
    Args:
        target_counts: Dictionary mapping target areas to counts
        output_dir: Directory to save the chart
    """
    plt.figure(figsize=(8, 8))
    
    # Sort targets by count
    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
    targets = [t[0] for t in sorted_targets]
    counts = [t[1] for t in sorted_targets]
    
    # Create pie chart
    colors = {
        'HEAD': 'red',
        'BODY': 'blue',
        'LEGS': 'green',
        'ARMS': 'purple',
        'UNKNOWN': 'gray'
    }
    
    plt.pie(counts, labels=targets, autopct='%1.1f%%', 
            colors=[colors.get(target, 'gray') for target in targets])
    
    plt.title('Target Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the figure
    if output_dir:
        output_path = Path(output_dir) / "target_distribution.png"
    else:
        output_path = Path("./output") / "target_distribution.png"
    
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    
    plt.show()

def main():
    """
    Main function to run the test.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test technique recognition system")
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
            print("Usage: python test_technique_recognition.py --video [video_path]")
            return
    
    # Run the test
    test_technique_recognition(
        video_path=video_path,
        output_dir=args.output,
        display=args.display,
        save_frames=args.save_frames
    )

if __name__ == "__main__":
    main()
