# Computer Vision Frameworks for Thai Boxing Analysis

This document provides a comprehensive overview of computer vision frameworks suitable for Thai boxing analysis, focusing on human pose estimation technologies that can track fighters, recognize techniques, measure speed, and analyze matches.

## Key Requirements for Thai Boxing Vision App

1. **Track fighters** and their movements in the ring
2. **Recognize techniques** (punches, kicks, knees, elbows)
3. **Measure speed and location** of strikes on the body
4. **Detect clinches and knockdowns**
5. **Track rounds and time** within rounds
6. **Assess damage** via a hit meter
7. **Create timeline of events**
8. **Compare analytics** across multiple matches

## Human Pose Estimation Frameworks

### OpenPose

**Overview:**
- Developed by Carnegie Mellon University
- First real-time multi-person human pose identification library
- Can detect body, foot, hand, and facial key points on single images

**Features:**
- 3D single-person keypoint detection in real-time
- 2D multi-person keypoint detections in real-time
- Single-person tracking to speed up recognition
- Calibration toolkit for camera parameters

**Technical Details:**
- Uses Part Affinity Fields (PAFs) to show relationships between body parts
- First few layers extract features from an image
- Predicts 18 confidence maps for human stance skeleton
- Predicts 38 PAFs showing relationships between parts

**Limitations:**
- Low-resolution outputs, limiting detail in keypoint estimates
- Less suitable for applications requiring high precision (elite sports)
- Extremely inefficient (160 billion floating-point operations per inference)
- Still popular for single-person HPE and markerless motion capture

**Licensing:**
- Available for academic and non-profit organizations and noncommercial research use only
- Non-exclusive commercial license requires $25,000 annual royalty
- Commercial license cannot be used in the field of Sports (defined as "any athletic competition between individuals, groups of individuals or teams")

**Platform Support:**
- GPUs, CPUs, Cross-Platform Support, OpenCL (AMD GPU)
- Caffe, TensorFlow and PyTorch implementations

### MediaPipe

**Overview:**
- Cross-platform pipeline framework developed by Google
- Initially created to analyze YouTube videos and audio in real-time
- Currently in alpha stage, covers Android, iOS, and embedded devices

**Features:**
- Framework for inference from sensory input
- Set of tools for performance evaluation
- Library of reusable inference and processing components
- Tracer module for performance monitoring

**Technical Details:**
- Vision pipeline described as a directed graph of components ("Calculators")
- Data "Streams" connect calculators in the graph
- Each stream represents a time series of data "Packets"
- Supports more complex and dynamic behavior than neural networks
- Operates at a higher semantic level

**Advantages:**
- Remarkably accurate and efficient real-time human pose estimation
- Uses CNNs trained on large datasets of annotated human position photos
- Can handle occlusion, variations in body proportions and forms
- Learns directly from data rather than manually constructed features
- Offers unparalleled speed and efficiency for real-time applications
- Hardware acceleration techniques for interactive experiences on various devices

**Licensing:**
- Open-source framework with fewer licensing restrictions

**Platform Support:**
- Mac OS X, Debian Linux, iOS, and Android
- Built on C++ library (C++ 11)
- Supports Python and JavaScript implementations

### TensorFlow Pose-detection

**Overview:**
- Repository for building Human pose estimation models for commercial purposes
- Built on TensorFlow.js, can also run in Python

**Models:**
1. **MoveNet (2021)** - Can detect 17 key points of a body
   - Variants: MoveNet.Lightning and MoveNet.Thunder
   - Outperforms PoseNet on various datasets, especially fitness action images
   
2. **BlazePose (MediaPipe BlazePose)** - Can detect 33 key points (beyond the 17 COCO key points)

3. **PoseNet (2018)** - Can detect multiple poses, each with 17 key points

**Licensing:**
- Allows commercial use

### AlphaPose

**Overview:**
- Developed by Shanghai Jiao Tong University
- Models include AccuratePose, SinglePose, and MultiPose

**Features:**
- Freely available for non-commercial use
- Commercial license requires agreement with Shanghai Jiao Tong University

**Technical Details:**
- Different license types: Copyright by University of Michigan, Apache License, MIT License

**Platform Support:**
- Edge devices like NVIDIA Jetson series, Google Coral, or Raspberry Pi

### YOLO-NAS

**Overview:**
- Developed by Deci's Neural Architecture Search Technology
- Demonstrates excellent balance between latency and accuracy

**Licensing:**
- Commercial use, especially in production environments, is restricted unless agreed upon with Deci
- Deci.AI holds the license

### DensePose (by Meta)

**Overview:**
- Maps all human pixels of 2D RGB images to a 3D surface-based model of the body in real-time
- Aims at establishing dense correspondences between image pixels and 3D object geometry

**Features:**
- Chart-based Dense Pose Estimation for Humans and Animals
- Continuous Surface Embeddings for Dense Pose Estimation

**Licensing:**
- Allows commercial use

### HRNet (High-Resolution Network)

**Overview:**
- Designed to extract features at multiple scales simultaneously
- Learns high-level semantic information and fine-grained details

## Comparison of Frameworks

| Framework | Licensing for Sports | Real-time Performance | Multi-person Tracking | Precision | Platform Support |
|-----------|----------------------|----------------------|----------------------|-----------|-----------------|
| OpenPose | Restricted ($25,000/year) | Good | Yes | Medium | GPUs, CPUs, Cross-Platform |
| MediaPipe | Allowed | Excellent | Yes | High | Android, iOS, Web, Embedded |
| TensorFlow Pose | Allowed | Good | Yes | Good | Web, Mobile, Python |
| AlphaPose | Commercial license needed | Good | Yes | High | Edge devices, GPUs |
| YOLO-NAS | Restricted | Excellent | Yes | High | Edge devices |
| DensePose | Allowed | Medium | Yes | Very High | GPUs |
| HRNet | Allowed | Good | Yes | High | GPUs |

## Recommendations for Thai Boxing Analysis

Based on the requirements for the Thai boxing vision app and the comparison of frameworks:

1. **Primary Recommendation: MediaPipe**
   - Excellent real-time performance
   - Good multi-person tracking
   - No licensing restrictions for sports applications
   - Cross-platform support
   - Efficient on various hardware
   - Handles occlusion well (important for clinch detection)

2. **Alternative: TensorFlow Pose-detection (MoveNet)**
   - Good performance
   - No licensing restrictions
   - Well-documented
   - Good community support

3. **For Higher Precision (if real-time is not critical): DensePose**
   - Provides detailed 3D mapping
   - Better for damage assessment visualization
   - May require more computational resources

## Implementation Considerations

1. **Hardware Requirements:**
   - GPU acceleration recommended for real-time processing
   - Edge devices possible with MediaPipe or optimized TensorFlow models

2. **Integration Strategy:**
   - Use MediaPipe for real-time tracking and basic pose estimation
   - Implement custom classifiers on top of pose data for technique recognition
   - Consider ensemble approach combining multiple models for different aspects (e.g., MediaPipe for tracking, custom classifier for technique identification)

3. **Performance Optimization:**
   - Consider frame rate requirements (30fps minimum for smooth analysis)
   - Optimize for target hardware
   - Consider batch processing for non-real-time analysis features

4. **Technique Recognition:**
   - Will require custom training data of Thai boxing techniques
   - Consider temporal analysis for movement patterns
   - May need sequence modeling (LSTM/GRU) for technique classification
