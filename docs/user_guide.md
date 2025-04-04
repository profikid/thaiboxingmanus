# Thai Boxing Vision App - User Guide

## Overview

Thai Boxing Vision is an AI-powered application designed to analyze Muay Thai boxing matches using computer vision technology. The application can track fighters, recognize techniques, measure strike speed and impact, and provide comprehensive match statistics and visualizations.

## Features

- **Fighter Detection and Tracking**: Automatically identifies and tracks fighters throughout the match
- **Technique Recognition**: Identifies various strikes including punches, kicks, knees, and elbows
- **Speed and Impact Measurement**: Calculates the speed of strikes and estimates impact force
- **Round and Time Tracking**: Automatically detects rounds and tracks time within rounds
- **Damage Assessment**: Evaluates the potential damage of strikes based on type, speed, and target area
- **Match Statistics**: Generates comprehensive statistics on technique distribution, target areas, and fighter performance
- **Match Comparison**: Allows comparison of statistics across multiple matches
- **Timeline Visualization**: Creates a chronological timeline of all techniques used in a match

## Getting Started

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/thai-boxing-vision-app.git
   cd thai-boxing-vision-app
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   cd ui
   npm install
   ```

3. Start the application:
   ```
   # Start the backend services
   python src/main.py
   
   # In a separate terminal, start the frontend
   cd ui
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:3000`

### Analyzing a Match

1. From the dashboard, click on "Upload Match Video" or navigate to the Analysis page
2. Select a video file from your computer
3. Configure analysis settings if needed (detection confidence, tracking mode, etc.)
4. Click "Start Analysis" to begin processing
5. Wait for the analysis to complete (processing time depends on video length)
6. Review the results in the Results tab

### Viewing Match Statistics

After analysis is complete, you can view various statistics and visualizations:

- **Technique Timeline**: Chronological sequence of all detected techniques
- **Technique Distribution**: Breakdown of different technique types used in the match
- **Fighter Activity**: Comparison of technique usage between fighters
- **Round Analysis**: Technique distribution across different rounds
- **Target Areas**: Visualization of which body parts were targeted
- **Damage Assessment**: Evaluation of potential damage from different strikes

### Comparing Matches

1. Navigate to the Compare page
2. Select two previously analyzed matches or upload new videos
3. Configure comparison settings
4. Generate the comparison
5. View side-by-side statistics and visualizations

## Technical Details

### System Architecture

The Thai Boxing Vision app consists of several key components:

1. **Video Processing Module**: Handles video input, frame extraction, and preprocessing
2. **Fighter Tracking Module**: Detects and tracks fighters using MediaPipe pose estimation
3. **Technique Recognition Module**: Identifies techniques based on pose data and motion analysis
4. **Match Statistics Module**: Processes raw data to generate insights and visualizations
5. **User Interface**: Next.js frontend for interaction and visualization

### Technologies Used

- **Computer Vision**: MediaPipe for pose estimation
- **Backend**: Python with OpenCV, NumPy, and custom analysis algorithms
- **Frontend**: Next.js with React, Tailwind CSS
- **Deployment**: Cloudflare Workers for serverless deployment

## Troubleshooting

### Common Issues

- **Video Processing Fails**: Ensure the video is in a supported format (MP4, AVI, MOV)
- **Poor Detection Accuracy**: Try adjusting the detection confidence and tracking mode settings
- **Performance Issues**: For long videos, consider reducing the resolution or using the "Fast" tracking mode

### Getting Help

If you encounter any issues or have questions, please:
- Check the [GitHub Issues](https://github.com/your-username/thai-boxing-vision-app/issues) page
- Submit a new issue with detailed information about your problem
- Contact the development team at support@thaiboxingvision.com

## Future Enhancements

We're continuously working to improve Thai Boxing Vision. Planned enhancements include:

- Real-time analysis capabilities
- Mobile application support
- Integration with training management systems
- Enhanced AI models for even more accurate technique recognition
- Expanded fighter database with style analysis

Thank you for using Thai Boxing Vision!
