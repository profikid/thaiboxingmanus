"""
Technique Recognition Module for Thai Boxing Vision App

This module provides components for recognizing and classifying Thai boxing techniques,
calculating strike speed, and assessing impact/damage.
"""

from .technique_classifier import (
    PoseFeatureExtractor,
    TechniqueRecognizer,
    ImpactAssessor,
    TechniqueRecognitionSystem,
    TechniqueEvent,
    TechniqueType,
    BodyTarget
)

__all__ = [
    'PoseFeatureExtractor',
    'TechniqueRecognizer',
    'ImpactAssessor',
    'TechniqueRecognitionSystem',
    'TechniqueEvent',
    'TechniqueType',
    'BodyTarget'
]
