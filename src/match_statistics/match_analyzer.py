"""
Match Statistics and Visualization Module for Thai Boxing Vision App

This module provides components for analyzing match data, generating statistics,
and creating visualizations for Thai boxing matches.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import timedelta
import time

# Import from other modules
from ..technique_recognition.technique_classifier import TechniqueEvent, TechniqueType, BodyTarget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('match_statistics')

@dataclass
class FighterStats:
    """Data class to store statistics for a fighter."""
    fighter_id: int
    name: str = "Fighter"
    role: str = "Unknown"
    
    # Technique counts
    total_techniques: int = 0
    punches: int = 0
    kicks: int = 0
    knees: int = 0
    elbows: int = 0
    clinches: int = 0
    
    # Technique breakdown
    technique_counts: Dict[str, int] = field(default_factory=dict)
    
    # Target breakdown
    target_counts: Dict[str, int] = field(default_factory=dict)
    
    # Round statistics
    round_activity: Dict[int, int] = field(default_factory=dict)
    
    # Speed and impact metrics
    avg_speed: float = 0.0
    max_speed: float = 0.0
    avg_impact: float = 0.0
    max_impact: float = 0.0
    
    # Damage received
    damage_received: float = 0.0
    damage_by_region: Dict[str, float] = field(default_factory=dict)
    
    # Timeline of techniques
    technique_timeline: List[Dict] = field(default_factory=list)

@dataclass
class MatchStats:
    """Data class to store statistics for a match."""
    match_id: str
    date: str = ""
    location: str = ""
    
    # Fighters
    fighter_stats: Dict[int, FighterStats] = field(default_factory=dict)
    
    # Round information
    total_rounds: int = 0
    round_durations: List[float] = field(default_factory=list)
    
    # Match timeline
    timeline: List[Dict] = field(default_factory=list)
    
    # Overall statistics
    total_techniques: int = 0
    technique_distribution: Dict[str, int] = field(default_factory=dict)
    target_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Match intensity metrics
    intensity_by_round: Dict[int, float] = field(default_factory=dict)
    
    # Match outcome
    winner_id: Optional[int] = None
    outcome_type: str = "Decision"  # Decision, KO, TKO, etc.


class MatchAnalyzer:
    """
    Analyzes match data and generates statistics.
    """
    
    def __init__(self, match_id: str = "match_001"):
        """
        Initialize the match analyzer.
        
        Args:
            match_id: Unique identifier for the match
        """
        self.match_id = match_id
        self.match_stats = MatchStats(match_id=match_id)
        self.current_round = 0
        self.round_start_time = 0.0
        self.round_end_time = 0.0
        
        logger.info(f"MatchAnalyzer initialized for match {match_id}")
    
    def set_match_info(self, date: str, location: str) -> None:
        """
        Set basic match information.
        
        Args:
            date: Match date
            location: Match location
        """
        self.match_stats.date = date
        self.match_stats.location = location
    
    def register_fighter(self, fighter_id: int, name: str = None, role: str = None) -> None:
        """
        Register a fighter for the match.
        
        Args:
            fighter_id: Unique identifier for the fighter
            name: Fighter name (optional)
            role: Fighter role (e.g., "Red Corner", "Blue Corner")
        """
        if fighter_id not in self.match_stats.fighter_stats:
            fighter_name = name if name else f"Fighter {fighter_id}"
            fighter_role = role if role else "Unknown"
            
            self.match_stats.fighter_stats[fighter_id] = FighterStats(
                fighter_id=fighter_id,
                name=fighter_name,
                role=fighter_role
            )
            
            logger.info(f"Registered fighter {fighter_id} ({fighter_name}, {fighter_role})")
    
    def start_round(self, round_num: int, timestamp: float) -> None:
        """
        Start a new round.
        
        Args:
            round_num: Round number
            timestamp: Start time of the round in seconds
        """
        self.current_round = round_num
        self.round_start_time = timestamp
        
        # Ensure we have enough rounds in our data structure
        if len(self.match_stats.round_durations) < round_num:
            # Add empty rounds up to the current one
            for _ in range(len(self.match_stats.round_durations), round_num):
                self.match_stats.round_durations.append(0.0)
        
        # Update total rounds if needed
        self.match_stats.total_rounds = max(self.match_stats.total_rounds, round_num)
        
        logger.info(f"Started round {round_num} at timestamp {timestamp:.2f}s")
        
        # Add round start event to timeline
        self._add_to_timeline({
            'event_type': 'round_start',
            'round': round_num,
            'timestamp': timestamp
        })
    
    def end_round(self, round_num: int, timestamp: float) -> None:
        """
        End the current round.
        
        Args:
            round_num: Round number
            timestamp: End time of the round in seconds
        """
        if round_num != self.current_round:
            logger.warning(f"Ending round {round_num} but current round is {self.current_round}")
        
        self.round_end_time = timestamp
        
        # Calculate round duration
        duration = self.round_end_time - self.round_start_time
        
        # Update round duration
        if 0 <= round_num - 1 < len(self.match_stats.round_durations):
            self.match_stats.round_durations[round_num - 1] = duration
        
        logger.info(f"Ended round {round_num} at timestamp {timestamp:.2f}s (duration: {duration:.2f}s)")
        
        # Add round end event to timeline
        self._add_to_timeline({
            'event_type': 'round_end',
            'round': round_num,
            'timestamp': timestamp,
            'duration': duration
        })
    
    def process_technique(self, technique: TechniqueEvent) -> None:
        """
        Process a detected technique and update statistics.
        
        Args:
            technique: Detected technique event
        """
        fighter_id = technique.fighter_id
        
        # Register fighter if not already registered
        if fighter_id not in self.match_stats.fighter_stats:
            self.register_fighter(fighter_id)
        
        # Get fighter stats
        fighter_stats = self.match_stats.fighter_stats[fighter_id]
        
        # Update technique counts
        fighter_stats.total_techniques += 1
        self.match_stats.total_techniques += 1
        
        # Update technique type counts
        technique_type = technique.technique_type.name
        
        # Update fighter's technique counts
        if technique_type not in fighter_stats.technique_counts:
            fighter_stats.technique_counts[technique_type] = 0
        fighter_stats.technique_counts[technique_type] += 1
        
        # Update match technique distribution
        if technique_type not in self.match_stats.technique_distribution:
            self.match_stats.technique_distribution[technique_type] = 0
        self.match_stats.technique_distribution[technique_type] += 1
        
        # Update technique category counts
        if technique.technique_type in [TechniqueType.JAB, TechniqueType.CROSS, 
                                      TechniqueType.HOOK, TechniqueType.UPPERCUT]:
            fighter_stats.punches += 1
        elif technique.technique_type in [TechniqueType.FRONT_KICK, TechniqueType.ROUNDHOUSE_KICK,
                                        TechniqueType.SIDE_KICK, TechniqueType.AXE_KICK]:
            fighter_stats.kicks += 1
        elif technique.technique_type in [TechniqueType.STRAIGHT_KNEE, 
                                        TechniqueType.DIAGONAL_KNEE, 
                                        TechniqueType.FLYING_KNEE]:
            fighter_stats.knees += 1
        elif technique.technique_type in [TechniqueType.HORIZONTAL_ELBOW, 
                                        TechniqueType.UPPERCUT_ELBOW,
                                        TechniqueType.DIAGONAL_ELBOW, 
                                        TechniqueType.SPINNING_ELBOW]:
            fighter_stats.elbows += 1
        elif technique.technique_type == TechniqueType.CLINCH:
            fighter_stats.clinches += 1
        
        # Update target counts
        target = technique.target.name
        
        # Update fighter's target counts
        if target not in fighter_stats.target_counts:
            fighter_stats.target_counts[target] = 0
        fighter_stats.target_counts[target] += 1
        
        # Update match target distribution
        if target not in self.match_stats.target_distribution:
            self.match_stats.target_distribution[target] = 0
        self.match_stats.target_distribution[target] += 1
        
        # Update round activity
        round_num = self._get_round_for_timestamp(technique.start_time)
        if round_num not in fighter_stats.round_activity:
            fighter_stats.round_activity[round_num] = 0
        fighter_stats.round_activity[round_num] += 1
        
        # Update speed metrics
        fighter_stats.avg_speed = ((fighter_stats.avg_speed * (fighter_stats.total_techniques - 1)) + 
                                 technique.speed) / fighter_stats.total_techniques
        fighter_stats.max_speed = max(fighter_stats.max_speed, technique.speed)
        
        # Update impact metrics
        fighter_stats.avg_impact = ((fighter_stats.avg_impact * (fighter_stats.total_techniques - 1)) + 
                                  technique.impact_score) / fighter_stats.total_techniques
        fighter_stats.max_impact = max(fighter_stats.max_impact, technique.impact_score)
        
        # Update damage received for target fighter
        if technique.target_fighter_id is not None:
            target_fighter_id = technique.target_fighter_id
            
            # Register target fighter if not already registered
            if target_fighter_id not in self.match_stats.fighter_stats:
                self.register_fighter(target_fighter_id)
            
            # Get target fighter stats
            target_fighter_stats = self.match_stats.fighter_stats[target_fighter_id]
            
            # Update damage received
            damage = technique.impact_score
            target_fighter_stats.damage_received += damage
            
            # Update damage by region
            if target not in target_fighter_stats.damage_by_region:
                target_fighter_stats.damage_by_region[target] = 0.0
            target_fighter_stats.damage_by_region[target] += damage
        
        # Add technique to fighter's timeline
        fighter_stats.technique_timeline.append({
            'technique_type': technique_type,
            'start_time': technique.start_time,
            'end_time': technique.end_time,
            'speed': technique.speed,
            'impact': technique.impact_score,
            'target': target,
            'round': round_num
        })
        
        # Add technique to match timeline
        self._add_to_timeline({
            'event_type': 'technique',
            'technique_type': technique_type,
            'fighter_id': fighter_id,
            'target_fighter_id': technique.target_fighter_id,
            'timestamp': technique.start_time,
            'duration': technique.end_time - technique.start_time,
            'speed': technique.speed,
            'impact': technique.impact_score,
            'target': target,
            'round': round_num
        })
        
        # Update match intensity metrics
        self._update_intensity_metrics(round_num, technique.impact_score)
        
        logger.debug(f"Processed {technique_type} by fighter {fighter_id} at {technique.start_time:.2f}s")
    
    def set_match_outcome(self, winner_id: int, outcome_type: str) -> None:
        """
        Set the match outcome.
        
        Args:
            winner_id: ID of the winning fighter
            outcome_type: Type of outcome (e.g., "Decision", "KO", "TKO")
        """
        self.match_stats.winner_id = winner_id
        self.match_stats.outcome_type = outcome_type
        
        logger.info(f"Set match outcome: Fighter {winner_id} wins by {outcome_type}")
        
        # Add outcome to timeline
        self._add_to_timeline({
            'event_type': 'match_outcome',
            'winner_id': winner_id,
            'outcome_type': outcome_type,
            'timestamp': self.round_end_time  # Use the end time of the last round
        })
    
    def get_match_stats(self) -> MatchStats:
        """
        Get the complete match statistics.
        
        Returns:
            MatchStats object with all match statistics
        """
        return self.match_stats
    
    def get_fighter_stats(self, fighter_id: int) -> Optional[FighterStats]:
        """
        Get statistics for a specific fighter.
        
        Args:
            fighter_id: ID of the fighter
            
        Returns:
            FighterStats object for the specified fighter, or None if not found
        """
        return self.match_stats.fighter_stats.get(fighter_id)
    
    def get_technique_timeline(self) -> List[Dict]:
        """
        Get the complete timeline of techniques for the match.
        
        Returns:
            List of technique events in chronological order
        """
        # Filter timeline to only include technique events
        technique_events = [event for event in self.match_stats.timeline 
                          if event['event_type'] == 'technique']
        
        # Sort by timestamp
        return sorted(technique_events, key=lambda x: x['timestamp'])
    
    def get_round_summary(self, round_num: int) -> Dict:
        """
        Get a summary of statistics for a specific round.
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary with round statistics
        """
        if round_num < 1 or round_num > self.match_stats.total_rounds:
            logger.warning(f"Invalid round number: {round_num}")
            return {}
        
        # Get round duration
        duration = self.match_stats.round_durations[round_num - 1] if round_num <= len(self.match_stats.round_durations) else 0.0
        
        # Get techniques in this round
        round_techniques = [event for event in self.match_stats.timeline 
                          if event['event_type'] == 'technique' and event.get('round') == round_num]
        
        # Count techniques by fighter
        techniques_by_fighter = {}
        for technique in round_techniques:
            fighter_id = technique['fighter_id']
            if fighter_id not in techniques_by_fighter:
                techniques_by_fighter[fighter_id] = 0
            techniques_by_fighter[fighter_id] += 1
        
        # Get round intensity
        intensity = self.match_stats.intensity_by_round.get(round_num, 0.0)
        
        return {
            'round_number': round_num,
            'duration': duration,
            'total_techniques': len(round_techniques),
            'techniques_by_fighter': techniques_by_fighter,
            'intensity': intensity
        }
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save match statistics to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        # Convert dataclasses to dictionaries
        match_dict = self._match_stats_to_dict()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(match_dict, f, indent=2)
        
        logger.info(f"Saved match statistics to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load match statistics from a JSON file.
        
        Args:
            filepath: Path to the JSON file
        """
        # Load from file
        with open(filepath, 'r') as f:
            match_dict = json.load(f)
        
        # Convert dictionary to dataclasses
        self._dict_to_match_stats(match_dict)
        
        logger.info(f"Loaded match statistics from {filepath}")
    
    def _match_stats_to_dict(self) -> Dict:
        """
        Convert MatchStats dataclass to a dictionary for serialization.
        
        Returns:
            Dictionary representation of match statistics
        """
        # Convert fighter stats to dictionaries
        fighter_stats_dict = {}
        for fighter_id, stats in self.match_stats.fighter_stats.items():
            fighter_stats_dict[str(fighter_id)] = {
                'fighter_id': stats.fighter_id,
                'name': stats.name,
                'role': stats.role,
                'total_techniques': stats.total_techniques,
                'punches': stats.punches,
                'kicks': stats.kicks,
                'knees': stats.knees,
                'elbows': stats.elbows,
                'clinches': stats.clinches,
                'technique_counts': stats.technique_counts,
                'target_counts': stats.target_counts,
                'round_activity': {str(k): v for k, v in stats.round_activity.items()},
                'avg_speed': stats.avg_speed,
                'max_speed': stats.max_speed,
                'avg_impact': stats.avg_impact,
                'max_impact': stats.max_impact,
                'damage_received': stats.damage_received,
                'damage_by_region': stats.damage_by_region,
                'technique_timeline': stats.technique_timeline
            }
        
        # Convert match stats to dictionary
        match_dict = {
            'match_id': self.match_stats.match_id,
            'date': self.match_stats.date,
            'location': self.match_stats.location,
            'fighter_stats': fighter_stats_dict,
            'total_rounds': self.match_stats.total_rounds,
            'round_durations': self.match_stats.round_durations,
            'timeline': self.match_stats.timeline,
            'total_techniques': self.match_stats.total_techniques,
            'technique_distribution': self.match_stats.technique_distribution,
            'target_distribution': self.match_stats.target_distribution,
            'intensity_by_round': {str(k): v for k, v in self.match_stats.intensity_by_round.items()},
            'winner_id': self.match_stats.winner_id,
            'outcome_type': self.match_stats.outcome_type
        }
        
        return match_dict
    
    def _dict_to_match_stats(self, match_dict: Dict) -> None:
        """
        Convert dictionary to MatchStats dataclass.
        
        Args:
            match_dict: Dictionary representation of match statistics
        """
        # Create new MatchStats object
        self.match_stats = MatchStats(match_id=match_dict['match_id'])
        
        # Set basic match info
        self.match_stats.date = match_dict.get('date', '')
        self.match_stats.location = match_dict.get('location', '')
        self.match_stats.total_rounds = match_dict.get('total_rounds', 0)
        self.match_stats.round_durations = match_dict.get('round_durations', [])
        self.match_stats.timeline = match_dict.get('timeline', [])
        self.match_stats.total_techniques = match_dict.get('total_techniques', 0)
        self.match_stats.technique_distribution = match_dict.get('technique_distribution', {})
        self.match_stats.target_distribution = match_dict.get('target_distribution', {})
        self.match_stats.winner_id = match_dict.get('winner_id')
        self.match_stats.outcome_type = match_dict.get('outcome_type', 'Decision')
        
        # Convert intensity_by_round keys back to integers
        intensity_by_round = match_dict.get('intensity_by_round', {})
        self.match_stats.intensity_by_round = {int(k): v for k, v in intensity_by_round.items()}
        
        # Convert fighter stats
        fighter_stats_dict = match_dict.get('fighter_stats', {})
        for fighter_id_str, stats_dict in fighter_stats_dict.items():
            fighter_id = int(fighter_id_str)
            
            # Create FighterStats object
            fighter_stats = FighterStats(fighter_id=fighter_id)
            
            # Set fighter info
            fighter_stats.name = stats_dict.get('name', f"Fighter {fighter_id}")
            fighter_stats.role = stats_dict.get('role', 'Unknown')
            fighter_stats.total_techniques = stats_dict.get('total_techniques', 0)
            fighter_stats.punches = stats_dict.get('punches', 0)
            fighter_stats.kicks = stats_dict.get('kicks', 0)
            fighter_stats.knees = stats_dict.get('knees', 0)
            fighter_stats.elbows = stats_dict.get('elbows', 0)
            fighter_stats.clinches = stats_dict.get('clinches', 0)
            fighter_stats.technique_counts = stats_dict.get('technique_counts', {})
            fighter_stats.target_counts = stats_dict.get('target_counts', {})
            fighter_stats.avg_speed = stats_dict.get('avg_speed', 0.0)
            fighter_stats.max_speed = stats_dict.get('max_speed', 0.0)
            fighter_stats.avg_impact = stats_dict.get('avg_impact', 0.0)
            fighter_stats.max_impact = stats_dict.get('max_impact', 0.0)
            fighter_stats.damage_received = stats_dict.get('damage_received', 0.0)
            fighter_stats.damage_by_region = stats_dict.get('damage_by_region', {})
            fighter_stats.technique_timeline = stats_dict.get('technique_timeline', [])
            
            # Convert round_activity keys back to integers
            round_activity = stats_dict.get('round_activity', {})
            fighter_stats.round_activity = {int(k): v for k, v in round_activity.items()}
            
            # Add to match stats
            self.match_stats.fighter_stats[fighter_id] = fighter_stats
    
    def _add_to_timeline(self, event: Dict) -> None:
        """
        Add an event to the match timeline.
        
        Args:
            event: Event dictionary to add
        """
        self.match_stats.timeline.append(event)
    
    def _get_round_for_timestamp(self, timestamp: float) -> int:
        """
        Determine which round a timestamp belongs to.
        
        Args:
            timestamp: Timestamp in seconds
            
        Returns:
            Round number (1-based)
        """
        # If we're currently in a round and the timestamp is within it
        if self.current_round > 0 and self.round_start_time <= timestamp <= self.round_end_time:
            return self.current_round
        
        # Otherwise, try to find the round based on the timeline
        round_events = [event for event in self.match_stats.timeline 
                      if event['event_type'] in ['round_start', 'round_end']]
        
        # Sort by timestamp
        round_events.sort(key=lambda x: x['timestamp'])
        
        # Find the round that contains this timestamp
        current_round = 0
        round_start = 0.0
        
        for event in round_events:
            if event['event_type'] == 'round_start':
                current_round = event['round']
                round_start = event['timestamp']
            elif event['event_type'] == 'round_end' and current_round == event['round']:
                round_end = event['timestamp']
                if round_start <= timestamp <= round_end:
                    return current_round
        
        # If we couldn't determine the round, use the current round
        return self.current_round if self.current_round > 0 else 1
    
    def _update_intensity_metrics(self, round_num: int, impact_score: float) -> None:
        """
        Update match intensity metrics.
        
        Args:
            round_num: Round number
            impact_score: Impact score of the technique
        """
        # Initialize round intensity if not exists
        if round_num not in self.match_stats.intensity_by_round:
            self.match_stats.intensity_by_round[round_num] = 0.0
        
        # Update intensity (simple accumulation for now)
        self.match_stats.intensity_by_round[round_num] += impact_score


class MatchVisualizer:
    """
    Creates visualizations for match statistics.
    """
    
    def __init__(self, match_stats: MatchStats):
        """
        Initialize the match visualizer.
        
        Args:
            match_stats: Match statistics to visualize
        """
        self.match_stats = match_stats
        
        # Set up color schemes
        self.technique_colors = {
            'JAB': '#FF9999',  # Light red
            'CROSS': '#FF3333',  # Medium red
            'HOOK': '#CC0000',  # Dark red
            'UPPERCUT': '#990000',  # Very dark red
            
            'FRONT_KICK': '#99FF99',  # Light green
            'ROUNDHOUSE_KICK': '#33FF33',  # Medium green
            'SIDE_KICK': '#00CC00',  # Dark green
            'AXE_KICK': '#009900',  # Very dark green
            
            'STRAIGHT_KNEE': '#9999FF',  # Light blue
            'DIAGONAL_KNEE': '#3333FF',  # Medium blue
            'FLYING_KNEE': '#0000CC',  # Dark blue
            
            'HORIZONTAL_ELBOW': '#FF99FF',  # Light purple
            'UPPERCUT_ELBOW': '#FF33FF',  # Medium purple
            'DIAGONAL_ELBOW': '#CC00CC',  # Dark purple
            'SPINNING_ELBOW': '#990099',  # Very dark purple
            
            'CLINCH': '#FFFF99',  # Light yellow
            'SWEEP': '#FFCC00',  # Orange
            'BLOCK': '#CCCCCC',  # Gray
            'DODGE': '#999999',  # Dark gray
            
            'UNKNOWN': '#666666'  # Very dark gray
        }
        
        self.target_colors = {
            'HEAD': '#FF0000',  # Red
            'BODY': '#0000FF',  # Blue
            'LEGS': '#00FF00',  # Green
            'ARMS': '#FF00FF',  # Purple
            'UNKNOWN': '#999999'  # Gray
        }
        
        self.fighter_colors = {
            'Red Corner': '#FF0000',  # Red
            'Blue Corner': '#0000FF',  # Blue
            'Unknown': '#999999'  # Gray
        }
        
        logger.info("MatchVisualizer initialized")
    
    def create_technique_distribution_chart(self, 
                                          fighter_id: Optional[int] = None, 
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a bar chart showing the distribution of techniques.
        
        Args:
            fighter_id: ID of the fighter to visualize (None for all fighters)
            save_path: Path to save the chart image (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Get technique distribution
        if fighter_id is not None:
            if fighter_id not in self.match_stats.fighter_stats:
                logger.warning(f"Fighter {fighter_id} not found in match statistics")
                return plt.figure()
            
            fighter_stats = self.match_stats.fighter_stats[fighter_id]
            technique_counts = fighter_stats.technique_counts
            title = f"Technique Distribution - {fighter_stats.name}"
        else:
            technique_counts = self.match_stats.technique_distribution
            title = "Technique Distribution - All Fighters"
        
        # Sort techniques by count
        sorted_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
        techniques = [t[0] for t in sorted_techniques]
        counts = [t[1] for t in sorted_techniques]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar chart
        bars = ax.bar(techniques, counts)
        
        # Color bars by technique category
        for i, bar in enumerate(bars):
            technique = techniques[i]
            bar.set_color(self.technique_colors.get(technique, '#666666'))
        
        # Add labels and title
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Technique Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved technique distribution chart to {save_path}")
        
        return fig
    
    def create_target_distribution_chart(self, 
                                       fighter_id: Optional[int] = None, 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a pie chart showing the distribution of target areas.
        
        Args:
            fighter_id: ID of the fighter to visualize (None for all fighters)
            save_path: Path to save the chart image (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Get target distribution
        if fighter_id is not None:
            if fighter_id not in self.match_stats.fighter_stats:
                logger.warning(f"Fighter {fighter_id} not found in match statistics")
                return plt.figure()
            
            fighter_stats = self.match_stats.fighter_stats[fighter_id]
            target_counts = fighter_stats.target_counts
            title = f"Target Distribution - {fighter_stats.name}"
        else:
            target_counts = self.match_stats.target_distribution
            title = "Target Distribution - All Fighters"
        
        # Sort targets by count
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        targets = [t[0] for t in sorted_targets]
        counts = [t[1] for t in sorted_targets]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=targets, 
            autopct='%1.1f%%',
            colors=[self.target_colors.get(target, '#999999') for target in targets],
            startangle=90
        )
        
        # Customize text
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_color('white')
        
        # Add title
        ax.set_title(title, fontsize=16)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved target distribution chart to {save_path}")
        
        return fig
    
    def create_round_activity_chart(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a bar chart showing activity by round for each fighter.
        
        Args:
            save_path: Path to save the chart image (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Get fighter IDs
        fighter_ids = list(self.match_stats.fighter_stats.keys())
        
        # Get round numbers
        round_numbers = list(range(1, self.match_stats.total_rounds + 1))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set up bar width and positions
        bar_width = 0.8 / len(fighter_ids)
        
        # Create grouped bar chart
        for i, fighter_id in enumerate(fighter_ids):
            fighter_stats = self.match_stats.fighter_stats[fighter_id]
            
            # Get activity counts for each round
            activity = [fighter_stats.round_activity.get(round_num, 0) for round_num in round_numbers]
            
            # Calculate bar positions
            positions = [r + (i - len(fighter_ids)/2 + 0.5) * bar_width for r in round_numbers]
            
            # Create bars
            bars = ax.bar(
                positions, 
                activity, 
                width=bar_width, 
                label=f"{fighter_stats.name} ({fighter_stats.role})",
                color=self.fighter_colors.get(fighter_stats.role, '#999999')
            )
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Add labels and title
        ax.set_title('Activity by Round', fontsize=16)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Number of Techniques', fontsize=12)
        
        # Set x-axis ticks at round numbers
        ax.set_xticks(round_numbers)
        ax.set_xticklabels([f'Round {r}' for r in round_numbers])
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved round activity chart to {save_path}")
        
        return fig
    
    def create_technique_timeline(self, 
                                save_path: Optional[str] = None,
                                highlight_fighter: Optional[int] = None) -> plt.Figure:
        """
        Create a timeline visualization of techniques throughout the match.
        
        Args:
            save_path: Path to save the chart image (optional)
            highlight_fighter: ID of a fighter to highlight (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Get technique timeline
        timeline = self.match_stats.timeline
        
        # Filter to only include technique events
        technique_events = [event for event in timeline if event['event_type'] == 'technique']
        
        # Sort by timestamp
        technique_events.sort(key=lambda x: x['timestamp'])
        
        # Get round start/end times
        round_events = [event for event in timeline 
                      if event['event_type'] in ['round_start', 'round_end']]
        round_events.sort(key=lambda x: x['timestamp'])
        
        round_times = {}
        for event in round_events:
            round_num = event['round']
            if round_num not in round_times:
                round_times[round_num] = {'start': None, 'end': None}
            
            if event['event_type'] == 'round_start':
                round_times[round_num]['start'] = event['timestamp']
            elif event['event_type'] == 'round_end':
                round_times[round_num]['end'] = event['timestamp']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Set up y-axis positions for each fighter
        fighter_ids = list(self.match_stats.fighter_stats.keys())
        fighter_positions = {fighter_id: i for i, fighter_id in enumerate(fighter_ids)}
        
        # Draw round boundaries
        for round_num, times in round_times.items():
            if times['start'] is not None and times['end'] is not None:
                # Draw light gray background for the round
                ax.axvspan(times['start'], times['end'], alpha=0.2, color='gray')
                
                # Add round label
                ax.text(
                    (times['start'] + times['end']) / 2, 
                    -0.5, 
                    f'Round {round_num}',
                    ha='center', 
                    va='center',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
                )
        
        # Draw techniques
        for event in technique_events:
            fighter_id = event['fighter_id']
            technique_type = event['technique_type']
            timestamp = event['timestamp']
            impact = event.get('impact', 0.5)
            target = event.get('target', 'UNKNOWN')
            
            # Skip if highlighting a specific fighter and this isn't them
            if highlight_fighter is not None and fighter_id != highlight_fighter:
                continue
            
            # Get y-position for this fighter
            y_pos = fighter_positions.get(fighter_id, 0)
            
            # Get color for this technique
            color = self.technique_colors.get(technique_type, '#666666')
            
            # Draw marker
            marker_size = 100 * (0.5 + impact)  # Scale marker size by impact
            ax.scatter(
                timestamp, 
                y_pos, 
                s=marker_size, 
                color=color, 
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                zorder=3
            )
            
            # Add target indicator (small marker with target color)
            target_color = self.target_colors.get(target, '#999999')
            ax.scatter(
                timestamp, 
                y_pos, 
                s=marker_size/4, 
                color=target_color, 
                alpha=1.0,
                zorder=4
            )
        
        # Set y-axis ticks and labels
        ax.set_yticks(list(range(len(fighter_ids))))
        ax.set_yticklabels([f"{self.match_stats.fighter_stats[fid].name} ({self.match_stats.fighter_stats[fid].role})" 
                          for fid in fighter_ids])
        
        # Format x-axis as time
        ax.set_xlabel('Time (mm:ss)', fontsize=12)
        
        # Convert seconds to mm:ss format for x-axis
        def format_time(x, pos):
            minutes = int(x // 60)
            seconds = int(x % 60)
            return f'{minutes:02d}:{seconds:02d}'
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
        
        # Add title
        if highlight_fighter is not None:
            fighter_name = self.match_stats.fighter_stats[highlight_fighter].name
            ax.set_title(f'Technique Timeline - {fighter_name}', fontsize=16)
        else:
            ax.set_title('Technique Timeline', fontsize=16)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend for technique types
        technique_patches = [patches.Patch(color=color, label=technique) 
                           for technique, color in self.technique_colors.items()
                           if technique in self.match_stats.technique_distribution]
        
        # Add legend for targets
        target_patches = [patches.Patch(color=color, label=f"Target: {target}") 
                        for target, color in self.target_colors.items()
                        if target in self.match_stats.target_distribution]
        
        # Combine legends
        all_patches = technique_patches + target_patches
        
        # Add legend outside the plot
        ax.legend(
            handles=all_patches, 
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8
        )
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved technique timeline to {save_path}")
        
        return fig
    
    def create_damage_heatmap(self, 
                            fighter_id: int, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap visualization of damage received by a fighter.
        
        Args:
            fighter_id: ID of the fighter to visualize
            save_path: Path to save the chart image (optional)
            
        Returns:
            Matplotlib figure object
        """
        if fighter_id not in self.match_stats.fighter_stats:
            logger.warning(f"Fighter {fighter_id} not found in match statistics")
            return plt.figure()
        
        fighter_stats = self.match_stats.fighter_stats[fighter_id]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 10))
        
        # Load fighter silhouette image (or create a simple one)
        # For simplicity, we'll create a basic silhouette here
        silhouette = np.zeros((100, 50))
        
        # Head (rows 0-20)
        silhouette[0:20, 15:35] = 0.3
        
        # Body (rows 20-60)
        silhouette[20:60, 10:40] = 0.3
        
        # Arms
        silhouette[20:50, 0:10] = 0.3  # Left arm
        silhouette[20:50, 40:50] = 0.3  # Right arm
        
        # Legs
        silhouette[60:100, 10:25] = 0.3  # Left leg
        silhouette[60:100, 25:40] = 0.3  # Right leg
        
        # Create a copy for the heatmap
        heatmap = silhouette.copy()
        
        # Apply damage values to different regions
        # Map damage by region to the silhouette
        damage_by_region = fighter_stats.damage_by_region
        
        # Normalize damage values
        max_damage = max(damage_by_region.values()) if damage_by_region else 1.0
        
        # Apply damage to head
        if 'HEAD' in damage_by_region:
            normalized_damage = damage_by_region['HEAD'] / max_damage
            heatmap[0:20, 15:35] = normalized_damage
        
        # Apply damage to body
        if 'BODY' in damage_by_region:
            normalized_damage = damage_by_region['BODY'] / max_damage
            heatmap[20:60, 10:40] = normalized_damage
        
        # Apply damage to arms
        if 'ARMS' in damage_by_region:
            normalized_damage = damage_by_region['ARMS'] / max_damage
            heatmap[20:50, 0:10] = normalized_damage  # Left arm
            heatmap[20:50, 40:50] = normalized_damage  # Right arm
        
        # Apply damage to legs
        if 'LEGS' in damage_by_region:
            normalized_damage = damage_by_region['LEGS'] / max_damage
            heatmap[60:100, 10:25] = normalized_damage  # Left leg
            heatmap[60:100, 25:40] = normalized_damage  # Right leg
        
        # Create a custom colormap (white to red)
        cmap = LinearSegmentedColormap.from_list('damage_cmap', ['white', 'yellow', 'orange', 'red'])
        
        # Display the heatmap
        im = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Damage Intensity', fontsize=12)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title(f'Damage Heatmap - {fighter_stats.name}', fontsize=16)
        
        # Add region labels
        ax.text(25, 10, 'HEAD', ha='center', va='center', color='black', fontsize=8)
        ax.text(25, 40, 'BODY', ha='center', va='center', color='black', fontsize=8)
        ax.text(5, 35, 'ARMS', ha='center', va='center', color='black', fontsize=8)
        ax.text(45, 35, 'ARMS', ha='center', va='center', color='black', fontsize=8)
        ax.text(25, 80, 'LEGS', ha='center', va='center', color='black', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved damage heatmap to {save_path}")
        
        return fig
    
    def create_fighter_comparison_chart(self, 
                                      fighter_id1: int, 
                                      fighter_id2: int,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a radar chart comparing two fighters.
        
        Args:
            fighter_id1: ID of the first fighter
            fighter_id2: ID of the second fighter
            save_path: Path to save the chart image (optional)
            
        Returns:
            Matplotlib figure object
        """
        if fighter_id1 not in self.match_stats.fighter_stats or fighter_id2 not in self.match_stats.fighter_stats:
            logger.warning("One or both fighters not found in match statistics")
            return plt.figure()
        
        fighter1 = self.match_stats.fighter_stats[fighter_id1]
        fighter2 = self.match_stats.fighter_stats[fighter_id2]
        
        # Define metrics to compare
        metrics = [
            'Punches',
            'Kicks',
            'Knees',
            'Elbows',
            'Avg Speed',
            'Max Speed',
            'Avg Impact',
            'Max Impact'
        ]
        
        # Get values for each fighter
        fighter1_values = [
            fighter1.punches,
            fighter1.kicks,
            fighter1.knees,
            fighter1.elbows,
            fighter1.avg_speed,
            fighter1.max_speed,
            fighter1.avg_impact,
            fighter1.max_impact
        ]
        
        fighter2_values = [
            fighter2.punches,
            fighter2.kicks,
            fighter2.knees,
            fighter2.elbows,
            fighter2.avg_speed,
            fighter2.max_speed,
            fighter2.avg_impact,
            fighter2.max_impact
        ]
        
        # Normalize values
        max_values = [max(f1, f2) for f1, f2 in zip(fighter1_values, fighter2_values)]
        max_values = [max(1.0, val) for val in max_values]  # Avoid division by zero
        
        fighter1_normalized = [val / max_val for val, max_val in zip(fighter1_values, max_values)]
        fighter2_normalized = [val / max_val for val, max_val in zip(fighter2_values, max_values)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Number of metrics
        N = len(metrics)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first metric at the end to close the loop
        fighter1_normalized += fighter1_normalized[:1]
        fighter2_normalized += fighter2_normalized[:1]
        
        # Draw the radar chart
        ax.plot(angles, fighter1_normalized, 'o-', linewidth=2, 
               label=f"{fighter1.name} ({fighter1.role})",
               color=self.fighter_colors.get(fighter1.role, '#999999'))
        ax.fill(angles, fighter1_normalized, alpha=0.1, 
               color=self.fighter_colors.get(fighter1.role, '#999999'))
        
        ax.plot(angles, fighter2_normalized, 'o-', linewidth=2, 
               label=f"{fighter2.name} ({fighter2.role})",
               color=self.fighter_colors.get(fighter2.role, '#999999'))
        ax.fill(angles, fighter2_normalized, alpha=0.1, 
               color=self.fighter_colors.get(fighter2.role, '#999999'))
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add title
        ax.set_title('Fighter Comparison', fontsize=16)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved fighter comparison chart to {save_path}")
        
        return fig
    
    def create_match_summary_dashboard(self, 
                                     output_dir: str,
                                     base_filename: str = "match_summary") -> List[str]:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            output_dir: Directory to save the charts
            base_filename: Base name for the output files
            
        Returns:
            List of paths to the generated chart files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all charts
        chart_files = []
        
        # 1. Technique distribution (all fighters)
        technique_chart_path = os.path.join(output_dir, f"{base_filename}_techniques.png")
        self.create_technique_distribution_chart(save_path=technique_chart_path)
        chart_files.append(technique_chart_path)
        
        # 2. Target distribution (all fighters)
        target_chart_path = os.path.join(output_dir, f"{base_filename}_targets.png")
        self.create_target_distribution_chart(save_path=target_chart_path)
        chart_files.append(target_chart_path)
        
        # 3. Round activity
        activity_chart_path = os.path.join(output_dir, f"{base_filename}_round_activity.png")
        self.create_round_activity_chart(save_path=activity_chart_path)
        chart_files.append(activity_chart_path)
        
        # 4. Technique timeline
        timeline_chart_path = os.path.join(output_dir, f"{base_filename}_timeline.png")
        self.create_technique_timeline(save_path=timeline_chart_path)
        chart_files.append(timeline_chart_path)
        
        # 5. Individual fighter charts
        for fighter_id, fighter_stats in self.match_stats.fighter_stats.items():
            # Technique distribution
            fighter_technique_path = os.path.join(
                output_dir, 
                f"{base_filename}_fighter{fighter_id}_techniques.png"
            )
            self.create_technique_distribution_chart(
                fighter_id=fighter_id, 
                save_path=fighter_technique_path
            )
            chart_files.append(fighter_technique_path)
            
            # Target distribution
            fighter_target_path = os.path.join(
                output_dir, 
                f"{base_filename}_fighter{fighter_id}_targets.png"
            )
            self.create_target_distribution_chart(
                fighter_id=fighter_id, 
                save_path=fighter_target_path
            )
            chart_files.append(fighter_target_path)
            
            # Damage heatmap
            fighter_damage_path = os.path.join(
                output_dir, 
                f"{base_filename}_fighter{fighter_id}_damage.png"
            )
            self.create_damage_heatmap(
                fighter_id=fighter_id, 
                save_path=fighter_damage_path
            )
            chart_files.append(fighter_damage_path)
        
        # 6. Fighter comparison (if there are at least 2 fighters)
        fighter_ids = list(self.match_stats.fighter_stats.keys())
        if len(fighter_ids) >= 2:
            comparison_path = os.path.join(
                output_dir, 
                f"{base_filename}_comparison.png"
            )
            self.create_fighter_comparison_chart(
                fighter_id1=fighter_ids[0],
                fighter_id2=fighter_ids[1],
                save_path=comparison_path
            )
            chart_files.append(comparison_path)
        
        logger.info(f"Created match summary dashboard with {len(chart_files)} charts")
        
        return chart_files
    
    def create_html_report(self, 
                         output_path: str,
                         chart_files: List[str] = None) -> str:
        """
        Create an HTML report with match statistics and visualizations.
        
        Args:
            output_path: Path to save the HTML file
            chart_files: List of paths to chart images (optional)
            
        Returns:
            Path to the generated HTML file
        """
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Thai Boxing Match Analysis</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #444;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background-color: #f8f8f8;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 5px solid #3498db;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f8f8;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .fighter-red {{
                    color: #e74c3c;
                }}
                .fighter-blue {{
                    color: #3498db;
                }}
                .stat-highlight {{
                    font-weight: bold;
                    color: #2ecc71;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    color: #777;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Thai Boxing Match Analysis</h1>
                    <p>Match ID: {self.match_stats.match_id}</p>
                    <p>Date: {self.match_stats.date}</p>
                    <p>Location: {self.match_stats.location}</p>
                </div>
                
                <div class="section">
                    <h2>Match Summary</h2>
                    <p>Total Rounds: {self.match_stats.total_rounds}</p>
                    <p>Total Techniques: {self.match_stats.total_techniques}</p>
                    
                    <h3>Outcome</h3>
                    <p>
        """
        
        # Add outcome information
        if self.match_stats.winner_id is not None:
            winner = self.match_stats.fighter_stats.get(self.match_stats.winner_id)
            if winner:
                winner_class = "fighter-red" if winner.role == "Red Corner" else "fighter-blue"
                html_content += f"""
                        Winner: <span class="{winner_class}">{winner.name} ({winner.role})</span><br>
                        Outcome Type: {self.match_stats.outcome_type}
                """
            else:
                html_content += f"""
                        Winner: Fighter {self.match_stats.winner_id}<br>
                        Outcome Type: {self.match_stats.outcome_type}
                """
        else:
            html_content += "No winner declared"
        
        html_content += """
                    </p>
                </div>
                
                <div class="section">
                    <h2>Fighter Statistics</h2>
                    <table>
                        <tr>
                            <th>Fighter</th>
                            <th>Role</th>
                            <th>Total Techniques</th>
                            <th>Punches</th>
                            <th>Kicks</th>
                            <th>Knees</th>
                            <th>Elbows</th>
                            <th>Avg Speed</th>
                            <th>Max Speed</th>
                            <th>Damage Received</th>
                        </tr>
        """
        
        # Add fighter statistics
        for fighter_id, stats in self.match_stats.fighter_stats.items():
            fighter_class = "fighter-red" if stats.role == "Red Corner" else "fighter-blue"
            html_content += f"""
                        <tr class="{fighter_class}">
                            <td>{stats.name}</td>
                            <td>{stats.role}</td>
                            <td>{stats.total_techniques}</td>
                            <td>{stats.punches}</td>
                            <td>{stats.kicks}</td>
                            <td>{stats.knees}</td>
                            <td>{stats.elbows}</td>
                            <td>{stats.avg_speed:.2f}</td>
                            <td>{stats.max_speed:.2f}</td>
                            <td>{stats.damage_received:.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Round Summary</h2>
                    <table>
                        <tr>
                            <th>Round</th>
                            <th>Duration</th>
                            <th>Total Techniques</th>
                            <th>Intensity</th>
                        </tr>
        """
        
        # Add round statistics
        for round_num in range(1, self.match_stats.total_rounds + 1):
            round_summary = self._get_round_summary(round_num)
            
            duration_str = str(timedelta(seconds=round_summary.get('duration', 0)))[2:7]  # Format as MM:SS
            
            html_content += f"""
                        <tr>
                            <td>Round {round_num}</td>
                            <td>{duration_str}</td>
                            <td>{round_summary.get('total_techniques', 0)}</td>
                            <td>{round_summary.get('intensity', 0):.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
        """
        
        # Add charts if provided
        if chart_files:
            html_content += """
                <div class="section">
                    <h2>Visualizations</h2>
            """
            
            for chart_file in chart_files:
                # Get relative path to the chart file
                chart_filename = os.path.basename(chart_file)
                
                # Get chart title from filename
                chart_title = chart_filename.replace(f"{os.path.basename(output_path).split('.')[0]}_", "")
                chart_title = chart_title.replace(".png", "").replace("_", " ").title()
                
                html_content += f"""
                    <div class="chart-container">
                        <h3>{chart_title}</h3>
                        <img class="chart" src="{chart_filename}" alt="{chart_title}">
                    </div>
                """
            
            html_content += """
                </div>
            """
        
        # Add timeline section
        html_content += """
                <div class="section">
                    <h2>Technique Timeline</h2>
                    <table>
                        <tr>
                            <th>Time</th>
                            <th>Round</th>
                            <th>Fighter</th>
                            <th>Technique</th>
                            <th>Target</th>
                            <th>Speed</th>
                            <th>Impact</th>
                        </tr>
        """
        
        # Add timeline events
        technique_events = [event for event in self.match_stats.timeline 
                          if event['event_type'] == 'technique']
        
        # Sort by timestamp
        technique_events.sort(key=lambda x: x['timestamp'])
        
        for event in technique_events:
            fighter_id = event['fighter_id']
            fighter = self.match_stats.fighter_stats.get(fighter_id)
            fighter_name = fighter.name if fighter else f"Fighter {fighter_id}"
            fighter_role = fighter.role if fighter else "Unknown"
            
            fighter_class = "fighter-red" if fighter_role == "Red Corner" else "fighter-blue"
            
            # Format time as MM:SS
            time_str = str(timedelta(seconds=event['timestamp']))[2:7]
            
            html_content += f"""
                        <tr class="{fighter_class}">
                            <td>{time_str}</td>
                            <td>{event.get('round', '-')}</td>
                            <td>{fighter_name}</td>
                            <td>{event['technique_type']}</td>
                            <td>{event.get('target', 'UNKNOWN')}</td>
                            <td>{event.get('speed', 0):.2f}</td>
                            <td>{event.get('impact', 0):.2f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="footer">
                    <p>Generated by Thai Boxing Vision App</p>
                    <p>© 2025 All Rights Reserved</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created HTML report at {output_path}")
        
        # Copy chart files to the same directory as the HTML file if they're not already there
        if chart_files:
            html_dir = os.path.dirname(output_path)
            for chart_file in chart_files:
                if os.path.dirname(chart_file) != html_dir:
                    import shutil
                    shutil.copy(chart_file, html_dir)
        
        return output_path
    
    def _get_round_summary(self, round_num: int) -> Dict:
        """
        Get a summary of statistics for a specific round.
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary with round statistics
        """
        if round_num < 1 or round_num > self.match_stats.total_rounds:
            logger.warning(f"Invalid round number: {round_num}")
            return {}
        
        # Get round duration
        duration = self.match_stats.round_durations[round_num - 1] if round_num <= len(self.match_stats.round_durations) else 0.0
        
        # Get techniques in this round
        round_techniques = [event for event in self.match_stats.timeline 
                          if event['event_type'] == 'technique' and event.get('round') == round_num]
        
        # Count techniques by fighter
        techniques_by_fighter = {}
        for technique in round_techniques:
            fighter_id = technique['fighter_id']
            if fighter_id not in techniques_by_fighter:
                techniques_by_fighter[fighter_id] = 0
            techniques_by_fighter[fighter_id] += 1
        
        # Get round intensity
        intensity = self.match_stats.intensity_by_round.get(round_num, 0.0)
        
        return {
            'round_number': round_num,
            'duration': duration,
            'total_techniques': len(round_techniques),
            'techniques_by_fighter': techniques_by_fighter,
            'intensity': intensity
        }


class MatchStatisticsSystem:
    """
    Complete system for analyzing and visualizing Thai boxing match statistics.
    """
    
    def __init__(self, match_id: str = "match_001"):
        """
        Initialize the match statistics system.
        
        Args:
            match_id: Unique identifier for the match
        """
        self.match_analyzer = MatchAnalyzer(match_id=match_id)
        self.match_visualizer = None  # Will be initialized after analysis
        
        logger.info(f"MatchStatisticsSystem initialized for match {match_id}")
    
    def set_match_info(self, date: str, location: str) -> None:
        """
        Set basic match information.
        
        Args:
            date: Match date
            location: Match location
        """
        self.match_analyzer.set_match_info(date, location)
    
    def register_fighter(self, fighter_id: int, name: str = None, role: str = None) -> None:
        """
        Register a fighter for the match.
        
        Args:
            fighter_id: Unique identifier for the fighter
            name: Fighter name (optional)
            role: Fighter role (e.g., "Red Corner", "Blue Corner")
        """
        self.match_analyzer.register_fighter(fighter_id, name, role)
    
    def start_round(self, round_num: int, timestamp: float) -> None:
        """
        Start a new round.
        
        Args:
            round_num: Round number
            timestamp: Start time of the round in seconds
        """
        self.match_analyzer.start_round(round_num, timestamp)
    
    def end_round(self, round_num: int, timestamp: float) -> None:
        """
        End the current round.
        
        Args:
            round_num: Round number
            timestamp: End time of the round in seconds
        """
        self.match_analyzer.end_round(round_num, timestamp)
    
    def process_technique(self, technique: TechniqueEvent) -> None:
        """
        Process a detected technique and update statistics.
        
        Args:
            technique: Detected technique event
        """
        self.match_analyzer.process_technique(technique)
    
    def set_match_outcome(self, winner_id: int, outcome_type: str) -> None:
        """
        Set the match outcome.
        
        Args:
            winner_id: ID of the winning fighter
            outcome_type: Type of outcome (e.g., "Decision", "KO", "TKO")
        """
        self.match_analyzer.set_match_outcome(winner_id, outcome_type)
    
    def finalize_analysis(self) -> None:
        """
        Finalize the match analysis and initialize the visualizer.
        """
        # Initialize the visualizer with the match statistics
        self.match_visualizer = MatchVisualizer(self.match_analyzer.get_match_stats())
        
        logger.info("Match analysis finalized")
    
    def save_match_stats(self, filepath: str) -> None:
        """
        Save match statistics to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        self.match_analyzer.save_to_file(filepath)
    
    def load_match_stats(self, filepath: str) -> None:
        """
        Load match statistics from a JSON file.
        
        Args:
            filepath: Path to the JSON file
        """
        self.match_analyzer.load_from_file(filepath)
        
        # Initialize the visualizer with the loaded match statistics
        self.match_visualizer = MatchVisualizer(self.match_analyzer.get_match_stats())
    
    def create_match_summary_dashboard(self, 
                                     output_dir: str,
                                     base_filename: str = "match_summary") -> List[str]:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            output_dir: Directory to save the charts
            base_filename: Base name for the output files
            
        Returns:
            List of paths to the generated chart files
        """
        if self.match_visualizer is None:
            self.finalize_analysis()
        
        return self.match_visualizer.create_match_summary_dashboard(output_dir, base_filename)
    
    def create_html_report(self, 
                         output_path: str,
                         chart_files: List[str] = None) -> str:
        """
        Create an HTML report with match statistics and visualizations.
        
        Args:
            output_path: Path to save the HTML file
            chart_files: List of paths to chart images (optional)
            
        Returns:
            Path to the generated HTML file
        """
        if self.match_visualizer is None:
            self.finalize_analysis()
        
        return self.match_visualizer.create_html_report(output_path, chart_files)
    
    def compare_matches(self, 
                      other_match_stats: MatchStats,
                      output_dir: str,
                      base_filename: str = "match_comparison") -> List[str]:
        """
        Compare two matches and create visualization charts.
        
        Args:
            other_match_stats: Statistics for the other match to compare
            output_dir: Directory to save the charts
            base_filename: Base name for the output files
            
        Returns:
            List of paths to the generated chart files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current match stats
        current_match = self.match_analyzer.get_match_stats()
        
        # Create figure for technique distribution comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get technique distributions
        current_techniques = current_match.technique_distribution
        other_techniques = other_match_stats.technique_distribution
        
        # Get all technique types
        all_techniques = sorted(set(list(current_techniques.keys()) + list(other_techniques.keys())))
        
        # Set up bar positions
        x = np.arange(len(all_techniques))
        width = 0.35
        
        # Create bars
        current_counts = [current_techniques.get(t, 0) for t in all_techniques]
        other_counts = [other_techniques.get(t, 0) for t in all_techniques]
        
        ax.bar(x - width/2, current_counts, width, label=f'Match {current_match.match_id}')
        ax.bar(x + width/2, other_counts, width, label=f'Match {other_match_stats.match_id}')
        
        # Add labels and title
        ax.set_title('Technique Distribution Comparison', fontsize=16)
        ax.set_xlabel('Technique Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(all_techniques, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        technique_chart_path = os.path.join(output_dir, f"{base_filename}_techniques.png")
        plt.savefig(technique_chart_path)
        
        # Create figure for target distribution comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get target distributions
        current_targets = current_match.target_distribution
        other_targets = other_match_stats.target_distribution
        
        # Get all target types
        all_targets = sorted(set(list(current_targets.keys()) + list(other_targets.keys())))
        
        # Set up bar positions
        x = np.arange(len(all_targets))
        width = 0.35
        
        # Create bars
        current_counts = [current_targets.get(t, 0) for t in all_targets]
        other_counts = [other_targets.get(t, 0) for t in all_targets]
        
        ax.bar(x - width/2, current_counts, width, label=f'Match {current_match.match_id}')
        ax.bar(x + width/2, other_counts, width, label=f'Match {other_match_stats.match_id}')
        
        # Add labels and title
        ax.set_title('Target Distribution Comparison', fontsize=16)
        ax.set_xlabel('Target Area', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(all_targets)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        target_chart_path = os.path.join(output_dir, f"{base_filename}_targets.png")
        plt.savefig(target_chart_path)
        
        # Create figure for round intensity comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get round intensities
        current_intensity = current_match.intensity_by_round
        other_intensity = other_match_stats.intensity_by_round
        
        # Get all rounds
        max_rounds = max(
            max(current_intensity.keys()) if current_intensity else 0,
            max(other_intensity.keys()) if other_intensity else 0
        )
        
        rounds = list(range(1, max_rounds + 1))
        
        # Set up bar positions
        x = np.arange(len(rounds))
        width = 0.35
        
        # Create bars
        current_values = [current_intensity.get(r, 0) for r in rounds]
        other_values = [other_intensity.get(r, 0) for r in rounds]
        
        ax.bar(x - width/2, current_values, width, label=f'Match {current_match.match_id}')
        ax.bar(x + width/2, other_values, width, label=f'Match {other_match_stats.match_id}')
        
        # Add labels and title
        ax.set_title('Round Intensity Comparison', fontsize=16)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Round {r}' for r in rounds])
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        intensity_chart_path = os.path.join(output_dir, f"{base_filename}_intensity.png")
        plt.savefig(intensity_chart_path)
        
        # Return paths to all generated charts
        chart_files = [technique_chart_path, target_chart_path, intensity_chart_path]
        
        logger.info(f"Created match comparison charts at {output_dir}")
        
        return chart_files


if __name__ == "__main__":
    # Example usage
    import argparse
    import time
    from pathlib import Path
    
    # Add parent directory to path to import technique_recognition
    sys.path.append(str(Path(__file__).parent.parent))
    from technique_recognition.technique_classifier import TechniqueRecognitionSystem, TechniqueType, BodyTarget
    from fighter_tracking.fighter_tracker import FighterTrackingSystem, FighterRole
    
    parser = argparse.ArgumentParser(description="Match statistics and visualization for Thai boxing analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--match-id", type=str, default="match_001", help="Match ID")
    parser.add_argument("--save-stats", action="store_true", help="Save match statistics to JSON file")
    parser.add_argument("--create-report", action="store_true", help="Create HTML report")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        exit(1)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize tracking system
    tracking_system = FighterTrackingSystem()
    
    # Initialize technique recognition system
    technique_system = TechniqueRecognitionSystem()
    
    # Initialize match statistics system
    match_stats_system = MatchStatisticsSystem(match_id=args.match_id)
    
    # Set match info
    match_date = time.strftime("%Y-%m-%d")
    match_stats_system.set_match_info(date=match_date, location="Bangkok, Thailand")
    
    # Process video
    frame_count = 0
    current_round = 0
    round_start_time = 0
    
    # Assume standard Muay Thai timing: 3-minute rounds, 1-minute breaks
    round_duration = 3 * 60  # 3 minutes in seconds
    break_duration = 1 * 60  # 1 minute in seconds
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp
        timestamp = frame_count / fps
        
        # Check if we need to start/end a round
        expected_round = int(timestamp / (round_duration + break_duration)) + 1
        round_time = timestamp % (round_duration + break_duration)
        
        # Start a new round
        if expected_round > current_round and round_time < round_duration:
            current_round = expected_round
            round_start_time = timestamp
            match_stats_system.start_round(current_round, timestamp)
            print(f"Started round {current_round} at {timestamp:.2f}s")
        
        # End the current round
        elif current_round > 0 and round_time >= round_duration and round_time < round_duration + 1:
            match_stats_system.end_round(current_round, timestamp)
            print(f"Ended round {current_round} at {timestamp:.2f}s")
        
        # Process frame for fighter tracking
        result_frame, tracked_poses = tracking_system.process_frame(frame, timestamp)
        
        # Register fighters with roles
        for pose in tracked_poses:
            if pose.role == FighterRole.RED_CORNER:
                match_stats_system.register_fighter(pose.fighter_id, name=f"Fighter {pose.fighter_id}", role="Red Corner")
            elif pose.role == FighterRole.BLUE_CORNER:
                match_stats_system.register_fighter(pose.fighter_id, name=f"Fighter {pose.fighter_id}", role="Blue Corner")
            else:
                match_stats_system.register_fighter(pose.fighter_id)
        
        # Process fighters for technique recognition
        techniques = technique_system.process_fighters(tracked_poses, frame_count, timestamp)
        
        # Process detected techniques for match statistics
        for technique in techniques:
            match_stats_system.process_technique(technique)
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} FPS)")
    
    # Release resources
    cap.release()
    tracking_system.release()
    
    # Set match outcome (for demonstration, we'll set the winner to the fighter with the most techniques)
    match_stats = match_stats_system.match_analyzer.get_match_stats()
    if match_stats.fighter_stats:
        # Find fighter with most techniques
        winner_id = max(match_stats.fighter_stats.items(), 
                      key=lambda x: x[1].total_techniques)[0]
        match_stats_system.set_match_outcome(winner_id, "Decision")
    
    # Finalize analysis
    match_stats_system.finalize_analysis()
    
    # Save match statistics if requested
    if args.save_stats:
        stats_file = os.path.join(args.output_dir, f"{args.match_id}_stats.json")
        match_stats_system.save_match_stats(stats_file)
        print(f"Saved match statistics to {stats_file}")
    
    # Create visualizations
    charts = match_stats_system.create_match_summary_dashboard(
        output_dir=args.output_dir,
        base_filename=args.match_id
    )
    print(f"Created {len(charts)} visualization charts")
    
    # Create HTML report if requested
    if args.create_report:
        report_file = os.path.join(args.output_dir, f"{args.match_id}_report.html")
        match_stats_system.create_html_report(
            output_path=report_file,
            chart_files=charts
        )
        print(f"Created HTML report at {report_file}")
    
    print(f"Match analysis complete for {args.match_id}")
