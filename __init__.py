"""
Tello Explorer - Système d'Exploration Autonome
===============================================

Système complet pour l'exploration autonome de bâtiments endommagés
avec drone DJI Tello EDU.

Modules:
    - tello_controller: Contrôle du drone et navigation
    - vision: Traitement vidéo et détection
    - mapping: Cartographie duale (altitude + thermique)
    - obstacle_avoidance: Système d'évitement d'obstacles
    - exploration: Orchestration de mission complète

Exemple d'utilisation:
    >>> from tello_explorer import ExplorationMission, MissionConfig
    >>> config = MissionConfig(area_width=300, area_height=300)
    >>> mission = ExplorationMission(config, simulation_mode=True)
    >>> mission.prepare_mission()
    >>> mission.start_exploration()
"""

__version__ = "1.0.0"
__author__ = "Tello Explorer Team"
__license__ = "MIT"

# Imports principaux pour faciliter l'utilisation
from .tello_controller import TelloController
from .vision import VideoStream, ObstacleDetector, ThermalDetector
from .mapping import DualMap, ExplorationPlanner
from .obstacle_avoidance import (
    ObstacleAvoidanceSystem,
    ReactiveAvoidance,
    SafetyZone,
    AvoidanceStrategy
)
from .exploration import ExplorationMission, MissionConfig, SafetyScanner

__all__ = [
    # Controller
    'TelloController',

    # Vision
    'VideoStream',
    'ObstacleDetector',
    'ThermalDetector',

    # Mapping
    'DualMap',
    'ExplorationPlanner',

    # Obstacle Avoidance
    'ObstacleAvoidanceSystem',
    'ReactiveAvoidance',
    'SafetyZone',
    'AvoidanceStrategy',

    # Exploration
    'ExplorationMission',
    'MissionConfig',
    'SafetyScanner',
]
