"""
Test script for the match statistics and visualization module.

This script demonstrates the functionality of the match statistics system
by processing sample data and generating visualizations.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import json
import argparse

# Add the parent directory to the path to import the modules
sys.path.append(str(Path(__file__).parent.parent))
from fighter_tracking.fighter_tracker import FighterTrackingSystem, FighterRole
from technique_recognition.technique_classifier import (
    TechniqueRecognitionSystem, 
    TechniqueType, 
    BodyTarget,
    TechniqueEvent
)
from match_statistics.match_analyzer import (
    MatchStatisticsSystem,
    MatchAnalyzer,
    MatchVisualizer
)

def generate_sample_data(output_dir):
    """
    Generate sample match data for testing.
    
    Args:
        output_dir: Directory to save sample data
    
    Returns:
        Path to the generated sample data file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a match statistics system
    match_stats_system = MatchStatisticsSystem(match_id="sample_match_001")
    
    # Set match info
    match_stats_system.set_match_info(date="2025-04-04", location="Bangkok, Thailand")
    
    # Register fighters
    match_stats_system.register_fighter(fighter_id=1, name="Fighter A", role="Red Corner")
    match_stats_system.register_fighter(fighter_id=2, name="Fighter B", role="Blue Corner")
    
    # Simulate 5 rounds
    total_rounds = 5
    round_duration = 180  # 3 minutes in seconds
    
    for round_num in range(1, total_rounds + 1):
        # Start round
        round_start_time = (round_num - 1) * (round_duration + 60)  # 60s break between rounds
        match_stats_system.start_round(round_num, round_start_time)
        
        # Generate random techniques for this round
        num_techniques = np.random.randint(30, 50)  # Random number of techniques per round
        
        for i in range(num_techniques):
            # Random timestamp within the round
            timestamp = round_start_time + np.random.uniform(0, round_duration)
            
            # Random fighter (1 or 2)
            fighter_id = np.random.choice([1, 2])
            
            # Random technique type
            technique_types = [
                TechniqueType.JAB, TechniqueType.CROSS, TechniqueType.HOOK, TechniqueType.UPPERCUT,
                TechniqueType.FRONT_KICK, TechniqueType.ROUNDHOUSE_KICK, TechniqueType.SIDE_KICK,
                TechniqueType.STRAIGHT_KNEE, TechniqueType.DIAGONAL_KNEE,
                TechniqueType.HORIZONTAL_ELBOW, TechniqueType.DIAGONAL_ELBOW
            ]
            technique_type = np.random.choice(technique_types)
            
            # Random target
            targets = [BodyTarget.HEAD, BodyTarget.BODY, BodyTarget.LEGS, BodyTarget.ARMS]
            target = np.random.choice(targets)
            
            # Random speed and impact
            speed = np.random.uniform(50, 200)
            impact = np.random.uniform(0.1, 1.0)
            
            # Create technique event
            technique = TechniqueEvent(
                fighter_id=fighter_id,
                technique_type=technique_type,
                start_time=timestamp,
                end_time=timestamp + np.random.uniform(0.1, 0.5),
                speed=speed,
                impact_score=impact,
                target=target,
                target_fighter_id=3 - fighter_id  # Other fighter
            )
            
            # Process technique
            match_stats_system.process_technique(technique)
        
        # End round
        round_end_time = round_start_time + round_duration
        match_stats_system.end_round(round_num, round_end_time)
    
    # Set match outcome (random winner)
    winner_id = np.random.choice([1, 2])
    outcome_types = ["Decision", "KO", "TKO", "Split Decision"]
    outcome_type = np.random.choice(outcome_types)
    match_stats_system.set_match_outcome(winner_id, outcome_type)
    
    # Finalize analysis
    match_stats_system.finalize_analysis()
    
    # Save match statistics
    output_file = os.path.join(output_dir, "sample_match_data.json")
    match_stats_system.save_match_stats(output_file)
    
    print(f"Generated sample match data at {output_file}")
    
    return output_file

def test_match_statistics(data_file, output_dir):
    """
    Test the match statistics system with the provided data file.
    
    Args:
        data_file: Path to the match data JSON file
        output_dir: Directory to save output files
    """
    print(f"Testing match statistics with data file: {data_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a match statistics system
    match_stats_system = MatchStatisticsSystem()
    
    # Load match statistics
    match_stats_system.load_match_stats(data_file)
    
    # Create visualizations
    charts = match_stats_system.create_match_summary_dashboard(
        output_dir=output_dir,
        base_filename="match_summary"
    )
    
    print(f"Created {len(charts)} visualization charts")
    
    # Create HTML report
    report_file = os.path.join(output_dir, "match_report.html")
    match_stats_system.create_html_report(
        output_path=report_file,
        chart_files=charts
    )
    
    print(f"Created HTML report at {report_file}")
    
    # Generate a second sample match for comparison
    print("Generating second sample match for comparison...")
    match_stats_system2 = MatchStatisticsSystem(match_id="sample_match_002")
    
    # Set match info
    match_stats_system2.set_match_info(date="2025-04-05", location="Phuket, Thailand")
    
    # Register fighters
    match_stats_system2.register_fighter(fighter_id=1, name="Fighter C", role="Red Corner")
    match_stats_system2.register_fighter(fighter_id=2, name="Fighter D", role="Blue Corner")
    
    # Simulate 5 rounds
    total_rounds = 5
    round_duration = 180  # 3 minutes in seconds
    
    for round_num in range(1, total_rounds + 1):
        # Start round
        round_start_time = (round_num - 1) * (round_duration + 60)  # 60s break between rounds
        match_stats_system2.start_round(round_num, round_start_time)
        
        # Generate random techniques for this round
        num_techniques = np.random.randint(30, 50)  # Random number of techniques per round
        
        for i in range(num_techniques):
            # Random timestamp within the round
            timestamp = round_start_time + np.random.uniform(0, round_duration)
            
            # Random fighter (1 or 2)
            fighter_id = np.random.choice([1, 2])
            
            # Random technique type
            technique_types = [
                TechniqueType.JAB, TechniqueType.CROSS, TechniqueType.HOOK, TechniqueType.UPPERCUT,
                TechniqueType.FRONT_KICK, TechniqueType.ROUNDHOUSE_KICK, TechniqueType.SIDE_KICK,
                TechniqueType.STRAIGHT_KNEE, TechniqueType.DIAGONAL_KNEE,
                TechniqueType.HORIZONTAL_ELBOW, TechniqueType.DIAGONAL_ELBOW
            ]
            technique_type = np.random.choice(technique_types)
            
            # Random target
            targets = [BodyTarget.HEAD, BodyTarget.BODY, BodyTarget.LEGS, BodyTarget.ARMS]
            target = np.random.choice(targets)
            
            # Random speed and impact
            speed = np.random.uniform(50, 200)
            impact = np.random.uniform(0.1, 1.0)
            
            # Create technique event
            technique = TechniqueEvent(
                fighter_id=fighter_id,
                technique_type=technique_type,
                start_time=timestamp,
                end_time=timestamp + np.random.uniform(0.1, 0.5),
                speed=speed,
                impact_score=impact,
                target=target,
                target_fighter_id=3 - fighter_id  # Other fighter
            )
            
            # Process technique
            match_stats_system2.process_technique(technique)
        
        # End round
        round_end_time = round_start_time + round_duration
        match_stats_system2.end_round(round_num, round_end_time)
    
    # Set match outcome (random winner)
    winner_id = np.random.choice([1, 2])
    outcome_types = ["Decision", "KO", "TKO", "Split Decision"]
    outcome_type = np.random.choice(outcome_types)
    match_stats_system2.set_match_outcome(winner_id, outcome_type)
    
    # Finalize analysis
    match_stats_system2.finalize_analysis()
    
    # Save second match statistics
    output_file2 = os.path.join(output_dir, "sample_match_data2.json")
    match_stats_system2.save_match_stats(output_file2)
    
    # Compare the two matches
    print("Comparing the two matches...")
    comparison_charts = match_stats_system.compare_matches(
        other_match_stats=match_stats_system2.match_analyzer.get_match_stats(),
        output_dir=output_dir,
        base_filename="match_comparison"
    )
    
    print(f"Created {len(comparison_charts)} comparison charts")
    
    # Create HTML report for the comparison
    comparison_report_file = os.path.join(output_dir, "match_comparison_report.html")
    with open(comparison_report_file, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Thai Boxing Match Comparison</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1, h2 {
                    color: #444;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    background-color: #f8f8f8;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 5px solid #3498db;
                }
                .chart-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .chart {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .footer {
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    color: #777;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Thai Boxing Match Comparison</h1>
                    <p>Comparing match sample_match_001 vs sample_match_002</p>
                </div>
                
                <div class="chart-container">
                    <h2>Technique Distribution Comparison</h2>
                    <img class="chart" src="match_comparison_techniques.png" alt="Technique Distribution Comparison">
                </div>
                
                <div class="chart-container">
                    <h2>Target Distribution Comparison</h2>
                    <img class="chart" src="match_comparison_targets.png" alt="Target Distribution Comparison">
                </div>
                
                <div class="chart-container">
                    <h2>Round Intensity Comparison</h2>
                    <img class="chart" src="match_comparison_intensity.png" alt="Round Intensity Comparison">
                </div>
                
                <div class="footer">
                    <p>Generated by Thai Boxing Vision App</p>
                    <p>© 2025 All Rights Reserved</p>
                </div>
            </div>
        </body>
        </html>
        """)
    
    print(f"Created comparison HTML report at {comparison_report_file}")
    
    print("Match statistics testing complete")

def main():
    """
    Main function to run the test.
    """
    parser = argparse.ArgumentParser(description="Test match statistics and visualization module")
    parser.add_argument("--data", type=str, help="Path to match data JSON file")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--generate", action="store_true", help="Generate sample data")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Generate sample data if requested or if no data file provided
    if args.generate or not args.data:
        data_file = generate_sample_data(args.output)
    else:
        data_file = args.data
    
    # Test match statistics
    test_match_statistics(data_file, args.output)

if __name__ == "__main__":
    main()
