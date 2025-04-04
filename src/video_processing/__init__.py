"""
Video Processing Module for Thai Boxing Vision App

This module provides components for video input, frame extraction, and preprocessing
for Thai boxing match analysis.
"""

from .video_processor import VideoProcessor, VideoFrameExtractor, create_sample_frames

__all__ = ['VideoProcessor', 'VideoFrameExtractor', 'create_sample_frames']
