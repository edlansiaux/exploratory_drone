#!/usr/bin/env python3
"""
Module d'exploration autonome optimisé pour Tello EDU
Exploration de bâtiments délabrés avec cartographie thermique

Réglages alignés sur le notebook de terrain (drone__1_.ipynb):
  - host drone "192.168.10.1" propagé au contrôleur
  - VideoStream relié à l'objet drone réel (streamon + sleep(2) + get_frame_read)
"""

import time
import math
import threading
import logging
from typing import Tuple, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass

from tello_controller import TelloController, DroneState, Position, DEFAULT_TELLO_HOST
from mapping import DualMap, ExplorationPlanner, Obstacle
from obstacle_avoidance import (
    ObstacleAvoidanceSystem,
    ReactiveAvoidance,
    AvoidanceStrategy,
    SafetyZone,
    ThreatLevel
)
from vision import VideoStream, ObstacleDetector, ThermalDetector, VisualObstacle

logger = logging.getLogger(__name__)


class MissionStatus(Enum):
    """État de la mission"""
    IDLE = "idle"
    PREPARING = "preparing"
    SCANNING = "scanning"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    AVOIDING = "avoiding"
    RETURNING = "returning"
    COMPLETED = "completed"
    ABORTED = "aborted"
    EMERGENCY = "emergency"


class ExplorationMode(Enum):
    """Mode d'exploration"""
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"


@dataclass
class MissionConfig:
    """Configuration de la mission"""
    # Zone d'exploration
    area_width: float = 500.0           # cm
    area_height: float = 500.0          # cm
    exploration_altitude: float = 100.0  # cm
    step_size: float = 50.0             # cm
    
    # Pattern
    pattern: str = "snake"  # snake, spiral, room_search
    
    # Limites
    max_duration: float = 600.0         # secondes
    min_battery: int = 15               # %
    
    # Sécurité
    scan_interval: float = 300.0        # cm (scan 360° tous les 3m)
    safety_margin: float = 80.0         # cm
    
    # Connexion drone (cf. notebook terrain)
    host: str = DEFAULT_TELLO_HOST
    
    # Fonctionnalités
    enable_mapping: bool = True
    enable_thermal: bool = True
    enable_avoidance: bool = True
    enable_scanning: bool = True


class SafetyScanner:
    """
    Scanner de sécurité 360°
    Vérifie régulièrement toutes les directions
    """
    
    def __init__(self, controller: TelloController, detector: ObstacleDetector,
                 thermal: ThermalDetector):
        self.controller = controller
        self.detector = detector
        self.thermal = thermal
        
        self.scan_results = {
            'front': {'clear': True, 'distance': 500, 'obstacles': []},
            'right': {'clear': True, 'distance': 500, 'obstacles': []},
            'back': {'clear': True, 'distance': 500, 'obstacles': []},
            'left': {'clear': True, 'distance': 500, 'obstacles': []},
            'up': {'clear': True, 'distance': 500},
            'down': {'clear': True, 'distance': 500},
        }
        
        self.thermal_readings = []
        self.last_scan_time = 0
        self.scan_in_progress = False
    
    def perform_360_scan(self, video_stream: VideoStream,
                         on_progress: Callable = None) -> dict:
        """
        Effectue un scan 360° complet
        
        Returns:
            Résultats du scan par direction
        """
        if self.scan_in_progress:
            return self.scan_results
        
        self.scan_in_progress = True
        logger.info("📡 DÉBUT SCAN 360°")
        
        directions = ['front', 'right', 'back', 'left']
        
        for i, direction in enumerate(directions):
            if on_progress:
                on_progress(i + 1, 4, direction)
            
            # Attendre stabilisation
            time.sleep(0.5)
            
            # Capturer et analyser
            frame = video_stream.get_frame()
            if frame is not None:
                # Détection obstacles
                obstacles = self.detector.detect(frame)
                
                # Détection thermique
                _, hotspots = self.thermal.detect(frame)
                
                # Mise à jour résultats
                self.scan_results[direction]['obstacles'] = obstacles
                self.scan_results[direction]['clear'] = len([
                    o for o in obstacles if o.distance_estimate < 100
                ]) == 0
                
                if obstacles:
                    self.scan_results[direction]['distance'] = min(
                        o.distance_estimate for o in obstacles
                    )
                else:
                    self.scan_results[direction]['distance'] = 500
                
                # Enregistrer lectures thermiques
                for hotspot in hotspots:
                    self.thermal_readings.append({
                        'direction': direction,
                        'temperature': hotspot.temperature,
                        'timestamp': time.time()
                    })
            
            # Rotation vers direction suivante (sauf dernière)
            if i < 3:
                self.controller.rotate_clockwise(90)
                time.sleep(0.8)
        
        # Retour position initiale
        self.controller.rotate_clockwise(90)
        time.sleep(0.5)
        
        self.last_scan_time = time.time()
        self.scan_in_progress = False
        
        logger.info("✅ SCAN 360° TERMINÉ")
        self._log_scan_results()
        
        return self.scan_results
    
    def _log_scan_results(self):
        """Log les résultats du scan"""
        for direction, data in self.scan_results.items():
            if direction in ['up', 'down']:
                continue
            status = "✓" if data['clear'] else "⚠️"
            logger.info(f"  {direction}: {status} dist={data['distance']:.0f}cm obs={len(data.get('obstacles', []))}")
    
    def quick_scan(self, video_stream: VideoStream) -> dict:
        """Scan rapide frontal uniquement"""
        frame = video_stream.get_frame()
        if frame is None:
            return self.scan_results
        
        obstacles = self.detector.detect(frame)
        _, hotspots = self.thermal.detect(frame)
        
        self.scan_results['front']['obstacles'] = obstacles
        self.scan_results['front']['clear'] = len([
            o for o in obstacles if o.distance_estimate < 80
        ]) == 0
        
        if obstacles:
            self.scan_results['front']['distance'] = min(
                o.distance_estimate for o in obstacles
            )
        
        return self.scan_results
    
    def get_safest_direction(self) -> str:
        """Retourne la direction la plus sûre"""
        safest = 'front'
        max_dist = 0
        
        for direction in ['front', 'right', 'back', 'left']:
            dist = self.scan_results[direction]['distance']
            if dist > max_dist and self.scan_results[direction]['clear']:
                max_dist = dist
                safest = direction
        
        return safest


class ExplorationMission:
    """
    Gestionnaire de mission d'exploration pour bâtiments délabrés
    Navigation sécurisée avec scans périodiques et cartographie thermique
    """
    
    def __init__(self, config: MissionConfig = None, simulation_mode: bool = False):
        self.config = config or MissionConfig()
        self.simulation_mode = simulation_mode
        
        # Composants principaux
        self.controller = TelloController(simulation_mode, host=self.config.host)
        self.dual_map = DualMap(
            resolution=self.config.step_size,
            size=(self.config.area_width, self.config.area_height)
        )
        self.planner = ExplorationPlanner(self.dual_map, self.config.step_size)
        
        # Système d'évitement
        self.avoidance = ObstacleAvoidanceSystem(SafetyZone(
            front=self.config.safety_margin,
            back=60, left=60, right=60, above=50, below=40,
            emergency_front=40, emergency_sides=25, emergency_vertical=25
        ))
        self.reactive = ReactiveAvoidance(emergency_distance=35)
        
        # Vision
        # Le drone réel sera relié au VideoStream lors de prepare_mission()
        self.video_stream = VideoStream(simulation_mode=simulation_mode)
        self.obstacle_detector = ObstacleDetector()
        self.thermal_detector = ThermalDetector()
        
        # Scanner de sécurité
        self.scanner = SafetyScanner(
            self.controller, self.obstacle_detector, self.thermal_detector
        )
        
        # État de la mission
        self.status = MissionStatus.IDLE
        self.mode = ExplorationMode.FULL_AUTO
        
        # Compteurs
        self.start_time: Optional[float] = None
        self.waypoints_completed = 0
        self.total_waypoints = 0
        self.distance_since_scan = 0.0
        
        # Threads
        self._exploration_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Callbacks
        self.on_status_change: Optional[Callable] = None
        self.on_waypoint_reached: Optional[Callable] = None
        self.on_obstacle_detected: Optional[Callable] = None
        self.on_thermal_alert: Optional[Callable] = None
        self.on_scan_complete: Optional[Callable] = None
        
        # Config callbacks internes
        self.avoidance.on_obstacle_detected = self._handle_obstacle_detected
        self.avoidance.on_collision_imminent = self._handle_collision_imminent
        
        logger.info(f"ExplorationMission initialisée (simulation: {simulation_mode})")
    
    def _set_status(self, new_status: MissionStatus):
        """Change le statut avec notification"""
        old = self.status
        self.status = new_status
        logger.info(f"Status: {old.value} → {new_status.value}")
        
        if self.on_status_change:
            self.on_status_change(old, new_status)
    
    def _handle_obstacle_detected(self, obstacle):
        """Callback obstacle détecté"""
        self.dual_map.add_obstacle(
            obstacle.x, obstacle.y, obstacle.z,
            radius=50, is_mobile=obstacle.is_mobile,
            obstacle_type=obstacle.obstacle_type
        )
        
        if self.on_obstacle_detected:
            self.on_obstacle_detected(obstacle)
    
    def _handle_collision_imminent(self, obstacle, threat_level):
        """Callback collision imminente"""
        logger.warning(f"⚠️ COLLISION IMMINENTE: {threat_level.value}")
        
        if threat_level == ThreatLevel.CRITICAL:
            # Réaction réflexe
            self.controller.move_back(30)
    
    def prepare_mission(self) -> bool:
        """Prépare la mission"""
        self._set_status(MissionStatus.PREPARING)
        
        # Connexion drone
        if not self.controller.connect():
            logger.error("Échec connexion drone")
            self._set_status(MissionStatus.ABORTED)
            return False
        
        # Relier le drone réel au flux vidéo (pour streamon/get_frame_read)
        if not self.simulation_mode:
            self.video_stream.drone = self.controller.drone
        
        # Vérification batterie
        telemetry = self.controller.get_telemetry()
        if telemetry.get('battery', 0) < self.config.min_battery:
            logger.error(f"Batterie insuffisante: {telemetry.get('battery')}%")
            self._set_status(MissionStatus.ABORTED)
            return False
        
        # Démarrage vidéo (streamon + sleep(2) + get_frame_read en mode réel)
        if self.config.enable_mapping or self.config.enable_thermal:
            self.video_stream.start()
            time.sleep(1)
        
        # Génération du plan
        if self.config.pattern == "snake":
            waypoints = self.planner.generate_snake_pattern(
                self.config.area_width, self.config.area_height
            )
        elif self.config.pattern == "spiral":
            waypoints = self.planner.generate_spiral_pattern(
                max(self.config.area_width, self.config.area_height) / 2
            )
        else:
            waypoints = self.planner.generate_room_search_pattern(
                self.config.area_width, self.config.area_height
            )
        
        self.total_waypoints = len(waypoints)
        logger.info(f"Plan: {self.total_waypoints} waypoints ({self.config.pattern})")
        
        self._set_status(MissionStatus.IDLE)
        return True
    
    def start_exploration(self) -> bool:
        """Démarre l'exploration"""
        if self.status not in [MissionStatus.IDLE, MissionStatus.PAUSED]:
            logger.warning(f"Impossible de démarrer: {self.status.value}")
            return False
        
        # Démarrage systèmes
        if self.config.enable_avoidance:
            self.avoidance.start_monitoring()
        
        # Décollage si nécessaire
        if self.controller.state != DroneState.FLYING:
            if not self.controller.takeoff():
                self._set_status(MissionStatus.ABORTED)
                return False
            
            # Altitude d'exploration
            alt_diff = self.config.exploration_altitude - self.controller.position.z
            if alt_diff > 20:
                self.controller.move_up(int(alt_diff))
        
        # Scan initial 360°
        if self.config.enable_scanning:
            self._set_status(MissionStatus.SCANNING)
            self.scanner.perform_360_scan(
                self.video_stream,
                on_progress=lambda c, t, d: logger.info(f"Scan {c}/{t}: {d}")
            )
            
            if self.on_scan_complete:
                self.on_scan_complete(self.scanner.scan_results)
        
        # Démarrage thread d'exploration
        self._stop_event.clear()
        self._pause_event.clear()
        self._exploration_thread = threading.Thread(
            target=self._exploration_loop,
            daemon=True
        )
        self._exploration_thread.start()
        
        self._set_status(MissionStatus.IN_PROGRESS)
        self.start_time = time.time()
        
        return True
    
    def pause_exploration(self):
        """Met en pause"""
        if self.status == MissionStatus.IN_PROGRESS:
            self._pause_event.set()
            self._set_status(MissionStatus.PAUSED)
    
    def resume_exploration(self):
        """Reprend l'exploration"""
        if self.status == MissionStatus.PAUSED:
            self._pause_event.clear()
            self._set_status(MissionStatus.IN_PROGRESS)
    
    def stop_exploration(self):
        """Arrête l'exploration"""
        self._stop_event.set()
        
        if self._exploration_thread:
            self._exploration_thread.join(timeout=5.0)
        
        self.avoidance.stop_monitoring()
        self.video_stream.stop()
        self.controller.land()
        
        self._set_status(MissionStatus.COMPLETED)
    
    def emergency_stop(self):
        """Arrêt d'urgence"""
        self._stop_event.set()
        self.avoidance.stop_monitoring()
        self.controller.emergency_stop()
        self._set_status(MissionStatus.EMERGENCY)
    
    def return_to_home(self):
        """Retour à la base"""
        self._set_status(MissionStatus.RETURNING)
        self._stop_event.set()
        
        if self._exploration_thread:
            self._exploration_thread.join(timeout=5.0)
        
        self.controller.return_to_home()
        self.controller.land()
        
        self._set_status(MissionStatus.COMPLETED)
    
    def _exploration_loop(self):
        """Boucle principale d'exploration"""
        logger.info("Démarrage boucle d'exploration")
        
        while not self._stop_event.is_set():
            # Pause
            if self._pause_event.is_set():
                time.sleep(0.1)
                continue
            
            # Vérifications
            if not self._check_conditions():
                break
            
            # Scan périodique
            if self.config.enable_scanning:
                if self.distance_since_scan >= self.config.scan_interval:
                    self._set_status(MissionStatus.SCANNING)
                    self.scanner.perform_360_scan(self.video_stream)
                    self.distance_since_scan = 0
                    
                    if self.on_scan_complete:
                        self.on_scan_complete(self.scanner.scan_results)
                    
                    self._set_status(MissionStatus.IN_PROGRESS)
            
            # Prochain waypoint
            waypoint = self.planner.get_next_waypoint()
            if waypoint is None:
                logger.info("Exploration terminée - tous les waypoints atteints")
                break
            
            # Navigation
            success = self._navigate_to_waypoint(waypoint)
            
            if success:
                self.waypoints_completed += 1
                self._record_data()
                
                if self.on_waypoint_reached:
                    self.on_waypoint_reached(waypoint, self.planner.get_progress())
            
            time.sleep(0.1)
        
        logger.info("Boucle d'exploration terminée")
    
    def _check_conditions(self) -> bool:
        """Vérifie les conditions de continuation"""
        # Durée
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.max_duration:
                logger.warning("Durée max atteinte")
                return False
        
        # Batterie
        telemetry = self.controller.get_telemetry()
        battery = telemetry.get('battery', 100)
        if battery < self.config.min_battery:
            logger.warning(f"Batterie faible: {battery}%")
            return False
        
        return True
    
    def _navigate_to_waypoint(self, waypoint: Tuple[float, float]) -> bool:
        """
        Navigation vers un waypoint avec vérifications continues
        """
        target_x, target_y = waypoint
        target_z = self.config.exploration_altitude
        
        current = self.controller.position
        
        # Calcul direction
        dx = target_x - current.x
        dy = target_y - current.y
        distance_total = math.sqrt(dx**2 + dy**2)
        
        if distance_total < 15:
            return True
        
        # Angle vers cible
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Rotation vers cible
        self.controller.rotate_to_angle(target_angle)
        time.sleep(0.3)
        
        # Navigation par pas
        step_size = self.config.step_size
        distance_covered = 0
        
        while distance_covered < distance_total:
            if self._stop_event.is_set():
                return False
            
            # Scan rapide frontal
            scan = self.scanner.quick_scan(self.video_stream)
            
            # Vérification réflexe
            front_dist = scan['front']['distance']
            reactive_action = self.reactive.check(
                front_dist, self.controller.position.z
            )
            
            if reactive_action:
                return self._execute_reactive_action(reactive_action)
            
            # Vérification évitement
            if self.config.enable_avoidance:
                current = self.controller.position
                risk, obstacle, threat = self.avoidance.check_collision_risk(
                    current.x, current.y, current.z,
                    target_x, target_y, target_z,
                    current.yaw
                )
                
                if risk and threat.value >= ThreatLevel.HIGH.value:
                    return self._handle_avoidance(
                        (current.x, current.y, current.z),
                        (target_x, target_y, target_z),
                        obstacle, threat
                    )
            
            # Mouvement
            remaining = distance_total - distance_covered
            move_dist = min(step_size, remaining)
            
            if self.controller.move_forward(int(move_dist)):
                distance_covered += move_dist
                self.distance_since_scan += move_dist
                
                # Enregistrement données
                self._record_data()
            else:
                logger.warning("Blocage détecté")
                return False
            
            time.sleep(0.2)
        
        return True
    
    def _execute_reactive_action(self, action: AvoidanceStrategy) -> bool:
        """Exécute une action réflexe"""
        logger.info(f"⚡ Action réflexe: {action.value}")
        
        if action == AvoidanceStrategy.BACKTRACK:
            self.controller.move_back(50)
        elif action == AvoidanceStrategy.GO_UP:
            self.controller.move_up(50)
        elif action == AvoidanceStrategy.GO_DOWN:
            self.controller.move_down(30)
        elif action == AvoidanceStrategy.GO_LEFT:
            self.controller.move_left(50)
        elif action == AvoidanceStrategy.GO_RIGHT:
            self.controller.move_right(50)
        
        return False  # Réessayer le waypoint
    
    def _handle_avoidance(self, current: Tuple, target: Tuple,
                         obstacle, threat: ThreatLevel) -> bool:
        """Gère l'évitement d'obstacle"""
        self._set_status(MissionStatus.AVOIDING)
        
        logger.info(f"Évitement obstacle: ({obstacle.x:.0f}, {obstacle.y:.0f})")
        
        # Stratégie
        strategy = self.avoidance.get_avoidance_strategy(current, target, obstacle)
        
        if strategy == AvoidanceStrategy.STOP:
            logger.info("Attente passage obstacle mobile...")
            time.sleep(2.0)
            self._set_status(MissionStatus.IN_PROGRESS)
            return False
        
        if strategy == AvoidanceStrategy.EMERGENCY_LAND:
            self.emergency_stop()
            return False
        
        # Chemin d'évitement
        path = self.avoidance.calculate_avoidance_path(current, target, obstacle, strategy)
        
        for wp in path:
            if self._stop_event.is_set():
                return False
            
            # Navigation simplifiée vers waypoint d'évitement
            self._navigate_simple(wp)
        
        self._set_status(MissionStatus.IN_PROGRESS)
        return True
    
    def _navigate_simple(self, target: Tuple[float, float, float]):
        """Navigation simplifiée (sans vérification complète)"""
        current = self.controller.position
        
        dx = target[0] - current.x
        dy = target[1] - current.y
        dz = target[2] - current.z
        
        # Altitude d'abord
        if abs(dz) > 20:
            if dz > 0:
                self.controller.move_up(min(int(dz), 100))
            else:
                self.controller.move_down(min(int(abs(dz)), 100))
        
        # Rotation
        angle = math.degrees(math.atan2(dy, dx))
        self.controller.rotate_to_angle(angle)
        
        # Avance
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 20:
            self.controller.move_forward(min(int(dist), 100))
    
    def _record_data(self):
        """Enregistre les données d'exploration"""
        if not self.config.enable_mapping:
            return
        
        pos = self.controller.position
        telemetry = self.controller.get_telemetry()
        
        # Distance au sol
        tof = telemetry.get('tof_distance', pos.z)
        
        # Température
        temp = 25.0
        if self.config.enable_thermal:
            frame = self.video_stream.get_frame()
            if frame is not None:
                _, hotspots = self.thermal_detector.detect(frame)
                if hotspots:
                    temp = max(h.temperature for h in hotspots)
                    
                    # Alerte si température élevée
                    if temp > 80 and self.on_thermal_alert:
                        self.on_thermal_alert(pos, temp, hotspots)
                    
                    # Zone thermique
                    if temp > 60:
                        self.dual_map.add_thermal_zone(
                            pos.x, pos.y, pos.z, 50, temp,
                            is_active=(temp > 100)
                        )
        
        # Enregistrement
        self.dual_map.add_point(pos.x, pos.y, pos.z, tof, temp)
    
    def add_simulated_obstacle(self, x: float, y: float, z: float,
                               is_mobile: bool = False,
                               velocity: Tuple = (0, 0, 0),
                               obstacle_type: str = "debris"):
        """Ajoute un obstacle simulé"""
        distance = math.sqrt(x**2 + y**2 + z**2)
        direction = (x/max(distance,1), y/max(distance,1), z/max(distance,1))
        
        obs = self.avoidance.add_obstacle(x, y, z, distance, direction, obstacle_type)
        
        if is_mobile:
            for i in range(5):
                time.sleep(0.1)
                obs.update_position(
                    x + velocity[0] * i * 0.1,
                    y + velocity[1] * i * 0.1,
                    z + velocity[2] * i * 0.1,
                    distance
                )
        
        return obs
    
    def get_mission_report(self) -> dict:
        """Génère le rapport de mission"""
        elapsed = 0
        if self.start_time:
            elapsed = time.time() - self.start_time
        
        telemetry = self.controller.get_telemetry()
        map_stats = self.dual_map.get_statistics()
        avoid_stats = self.avoidance.get_status_report()
        
        return {
            'status': self.status.value,
            'mode': self.mode.value,
            'duration_seconds': elapsed,
            'simulation': self.simulation_mode,
            'waypoints': {
                'completed': self.waypoints_completed,
                'total': self.total_waypoints,
                'progress': self.planner.get_progress()
            },
            'mapping': map_stats,
            'avoidance': avoid_stats,
            'thermal': {
                'max_temperature': self.thermal_detector.get_max_temperature(),
                'fire_detected': self.thermal_detector.has_fire_detected(),
                'zones': len(self.dual_map.thermal_zones)
            },
            'drone': {
                'position': self.controller.position.to_tuple(),
                'yaw': self.controller.position.yaw,
                'state': self.controller.state.value,
                'battery': telemetry.get('battery', 'N/A')
            }
        }
    
    def display_map(self, show_thermal: bool = False):
        """Affiche la carte"""
        drone_pos = (self.controller.position.x, self.controller.position.y)
        print(self.dual_map.to_ascii_map(drone_pos, show_thermal))
    
    def export_results(self, base_path: str = "exploration_results"):
        """Exporte les résultats"""
        import json
        
        # Carte
        self.dual_map.export_to_json(f"{base_path}_map.json")
        self.dual_map.export_grids(base_path)
        
        # Rapport
        report = self.get_mission_report()
        with open(f"{base_path}_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Résultats exportés: {base_path}_*")


# Test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("TEST EXPLORATION MISSION")
    print("=" * 60)
    
    # Configuration
    config = MissionConfig(
        area_width=300,
        area_height=300,
        exploration_altitude=100,
        step_size=50,
        pattern="snake",
        max_duration=60,
        scan_interval=200
    )
    
    # Mission
    mission = ExplorationMission(config, simulation_mode=True)
    
    # Callbacks
    def on_status(old, new):
        print(f"[STATUS] {old.value} → {new.value}")
    
    def on_waypoint(wp, progress):
        print(f"[WAYPOINT] ({wp[0]:.0f}, {wp[1]:.0f}) - {progress:.1f}%")
    
    def on_thermal(pos, temp, hotspots):
        print(f"[THERMAL] ⚠️ {temp:.0f}°C à ({pos.x:.0f}, {pos.y:.0f})")
    
    mission.on_status_change = on_status
    mission.on_waypoint_reached = on_waypoint
    mission.on_thermal_alert = on_thermal
    
    # Préparation
    print("\n--- Préparation ---")
    if mission.prepare_mission():
        
        # Obstacles simulés
        mission.add_simulated_obstacle(75, 50, 100, obstacle_type="debris")
        mission.add_simulated_obstacle(0, 100, 100, is_mobile=True, velocity=(10, 5, 0))
        
        # Zone thermique
        mission.dual_map.add_thermal_zone(100, 100, 50, 80, 120, True)
        
        # Démarrage
        print("\n--- Démarrage ---")
        mission.start_exploration()
        
        # Exploration
        time.sleep(8)
        
        # Cartes
        print("\n--- Carte Altitude ---")
        mission.display_map(show_thermal=False)
        
        print("\n--- Carte Thermique ---")
        mission.display_map(show_thermal=True)
        
        # Arrêt
        print("\n--- Arrêt ---")
        mission.stop_exploration()
        
        # Rapport
        print("\n--- Rapport ---")
        report = mission.get_mission_report()
        for k, v in report.items():
            print(f"  {k}: {v}")
        
        # Export
        mission.export_results("/tmp/test_exploration")
    
    print("\n" + "=" * 60)
    print("TEST TERMINÉ")
    print("=" * 60)
