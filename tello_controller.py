#!/usr/bin/env python3
"""
Module d'exploration autonome pour drone Tello EDU
CORRIGÉ: Suppression des méthodes dupliquées et logique cohérente "Nez en avant"
"""

import time
import math
import threading
import logging
from typing import Tuple, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass

from tello_controller import TelloController, DroneState, Position
from mapping import AltitudeMap, ExplorationPlanner, Obstacle
from obstacle_avoidance import (
    ObstacleAvoidanceSystem, 
    ReactiveAvoidance, 
    AvoidanceStrategy,
    SafetyZone
)

logger = logging.getLogger(__name__)

class ExplorationMode(Enum):
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"

class MissionStatus(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    RETURNING = "returning"
    COMPLETED = "completed"
    ABORTED = "aborted"
    EMERGENCY = "emergency"

@dataclass
class MissionConfig:
    area_width: float = 500.0
    area_height: float = 500.0
    exploration_altitude: float = 120.0
    step_size: float = 50.0
    pattern: str = "snake"
    max_duration: float = 600.0
    min_battery: int = 20
    enable_mapping: bool = True
    enable_avoidance: bool = True

class HazardDetector:
    def __init__(self):
        self.thermal_threshold = 50.0
        self.radiation_threshold = 0.5
        self.chemical_threshold = 100
        self.simulated_hotspots = []
        
    def add_simulated_hotspot(self, x, y, z, hazard_type, intensity):
        self.simulated_hotspots.append({
            'x': x, 'y': y, 'z': z, 'type': hazard_type, 'intensity': intensity
        })
    
    def check_hazards(self, x, y, z):
        hazards = {'thermal': 0.0, 'radiation': 0.0, 'chemical': 0.0, 'alerts': []}
        for hotspot in self.simulated_hotspots:
            dist = ((x - hotspot['x'])**2 + (y - hotspot['y'])**2 + (z - hotspot['z'])**2) ** 0.5
            if dist < 200:
                intensity = hotspot['intensity'] * (1 - dist/200)
                if hotspot['type'] in hazards: hazards[hotspot['type']] = max(hazards[hotspot['type']], intensity)
        return hazards

class ExplorationMission:
    def __init__(self, config: MissionConfig = None, simulation_mode: bool = False):
        self.config = config or MissionConfig()
        self.simulation_mode = simulation_mode
        self.controller = TelloController(simulation_mode)
        self.altitude_map = AltitudeMap(resolution=self.config.step_size, size=(self.config.area_width, self.config.area_height))
        self.planner = ExplorationPlanner(self.altitude_map, step_size=self.config.step_size)
        self.avoidance = ObstacleAvoidanceSystem(SafetyZone(front=80, back=50, left=50, right=50, above=50, below=40))
        self.reactive = ReactiveAvoidance(emergency_distance=40)
        self.hazard_detector = HazardDetector()
        
        self.status = MissionStatus.IDLE
        self.mode = ExplorationMode.FULL_AUTO
        self.distance_since_last_scan = 0.0
        self.scan_threshold = 500.0
        self.start_time: Optional[float] = None
        self.waypoints_completed = 0
        self.total_waypoints = 0
        
        self._exploration_thread: Optional[threading.Thread] = None
        self._stop_exploration = threading.Event()
        self._pause_exploration = threading.Event()
        
        self.on_status_change: Optional[Callable] = None
        self.on_hazard_detected: Optional[Callable] = None
        self.on_waypoint_reached: Optional[Callable] = None
        self.on_obstacle_detected: Optional[Callable] = None
        self.avoidance.on_obstacle_detected = self._handle_obstacle_detected
        
        logger.info(f"Mission initialisée (simulation: {simulation_mode})")
    
    def _perform_safety_scan(self):
        logger.info("📡 DÉBUT SCAN 360° DE SÉCURITÉ")
        for i in range(4):
            if self._stop_exploration.is_set(): break
            logger.info(f"Scan partie {i+1}/4...")
            self.controller.rotate_clockwise(90)
            time.sleep(1.0) 
            self._record_exploration_data()
        self.distance_since_last_scan = 0.0
        logger.info("✅ SCAN 360° TERMINÉ")

    def _set_status(self, new_status: MissionStatus):
        old_status = self.status
        self.status = new_status
        logger.info(f"Statut mission: {old_status.value} -> {new_status.value}")
        if self.on_status_change: self.on_status_change(old_status, new_status)
    
    def _handle_obstacle_detected(self, obstacle):
        self.altitude_map.add_obstacle(obstacle.x, obstacle.y, obstacle.z, radius=50, is_mobile=obstacle.is_mobile)
        if self.on_obstacle_detected: self.on_obstacle_detected(obstacle)
    
    def prepare_mission(self) -> bool:
        self._set_status(MissionStatus.PREPARING)
        if not self.controller.connect():
            self._set_status(MissionStatus.ABORTED)
            return False
        
        if self.config.pattern == "snake":
            waypoints = self.planner.generate_snake_pattern(self.config.area_width, self.config.area_height)
        else:
            waypoints = self.planner.generate_spiral_pattern(max(self.config.area_width, self.config.area_height) / 2)
        
        self.total_waypoints = len(waypoints)
        self._set_status(MissionStatus.IDLE)
        return True
    
    def start_exploration(self) -> bool:
        if self.status not in [MissionStatus.IDLE, MissionStatus.PAUSED]: return False
        
        if self.config.enable_avoidance: self.avoidance.start_monitoring()
        if self.controller.state != DroneState.FLYING:
            if not self.controller.takeoff():
                self._set_status(MissionStatus.ABORTED)
                return False
            self.controller.move_up(int(self.config.exploration_altitude - self.controller.position.z))
        
        logger.info("Exécution du scan initial avant exploration...")
        self._perform_safety_scan()
        
        self._stop_exploration.clear()
        self._pause_exploration.clear()
        self._exploration_thread = threading.Thread(target=self._exploration_loop, daemon=True)
        self._exploration_thread.start()
        
        self._set_status(MissionStatus.IN_PROGRESS)
        self.start_time = time.time()
        return True
    
    def pause_exploration(self):
        if self.status == MissionStatus.IN_PROGRESS:
            self._pause_exploration.set()
            self._set_status(MissionStatus.PAUSED)
    
    def resume_exploration(self):
        if self.status == MissionStatus.PAUSED:
            self._pause_exploration.clear()
            self._set_status(MissionStatus.IN_PROGRESS)
    
    def stop_exploration(self):
        self._stop_exploration.set()
        if self._exploration_thread: self._exploration_thread.join(timeout=5.0)
        self.avoidance.stop_monitoring()
        self.controller.land()
        self._set_status(MissionStatus.COMPLETED)
    
    def emergency_stop(self):
        self._stop_exploration.set()
        self.avoidance.stop_monitoring()
        self.controller.emergency_stop()
        self._set_status(MissionStatus.EMERGENCY)
    
    def return_to_home(self):
        self._set_status(MissionStatus.RETURNING)
        self._stop_exploration.set()
        self.controller.return_to_home()
        self.controller.land()
        self._set_status(MissionStatus.COMPLETED)
    
    def _exploration_loop(self):
        logger.info("Démarrage boucle exploration")
        while not self._stop_exploration.is_set():
            if self._pause_exploration.is_set():
                time.sleep(0.1)
                continue
            if not self._check_mission_conditions(): break
            
            waypoint = self.planner.get_next_waypoint()
            if waypoint is None:
                logger.info("Waypoints terminés")
                break
            
            if self._navigate_to_waypoint(waypoint):
                self.waypoints_completed += 1
                self._record_exploration_data()
                if self.on_waypoint_reached: self.on_waypoint_reached(waypoint, self.planner.get_progress())
            time.sleep(0.2)
    
    def _check_mission_conditions(self) -> bool:
        if self.start_time and (time.time() - self.start_time > self.config.max_duration): return False
        if self.controller.get_telemetry().get('battery', 100) < self.config.min_battery: return False
        return True

    def _navigate_to_waypoint(self, waypoint: Tuple[float, float]) -> bool:
        """
        Navigation principale : Rotation vers la cible, puis avance par pas.
        """
        target_x, target_y = waypoint
        target_z = self.config.exploration_altitude
        
        # Boucle de mouvement vers la cible
        while True:
            if self._stop_exploration.is_set(): return False
            
            current_pos = self.controller.position
            dx = target_x - current_pos.x
            dy = target_y - current_pos.y
            dist_total = math.sqrt(dx**2 + dy**2)
            
            if dist_total < 10: return True
            
            # Calcul angle vers cible (0° = +X, 90° = +Y)
            target_angle = math.degrees(math.atan2(dy, dx))
            
            # Rotation
            self.controller.rotate_to_angle(target_angle)
            time.sleep(0.2)
            
            # Distance pour ce pas (limité à step_size)
            step = min(self.config.step_size, dist_total)
            
            # Vérification scan périodique
            if self.distance_since_last_scan >= self.scan_threshold:
                self._perform_safety_scan()
                # Réalignement après scan
                self.controller.rotate_to_angle(target_angle)
            
            # Vérification obstacle
            if self.config.enable_avoidance:
                collision, obs = self.avoidance.check_collision_risk(
                    current_pos.x, current_pos.y, current_pos.z,
                    target_x, target_y, target_z
                )
                if collision and obs:
                    return self._handle_obstacle_avoidance((current_pos.x, current_pos.y, current_pos.z), (target_x, target_y, target_z), obs)
            
            # Mouvement AVANT
            if self.controller.move_forward(int(step)):
                self.distance_since_last_scan += step
                self._record_exploration_data()
            else:
                return False

    def _move_to_position(self, x, y, z):
        """Wrapper pour compatibilité avec l'évitement d'obstacle"""
        return self._navigate_to_waypoint((x, y))

    def _handle_obstacle_avoidance(self, current, target, obstacle) -> bool:
        logger.info(f"Évitement obstacle à ({obstacle.x:.0f}, {obstacle.y:.0f})")
        strategy = self.avoidance.get_avoidance_strategy(current, target, obstacle)
        
        if strategy == AvoidanceStrategy.STOP:
            time.sleep(2.0)
            return False
        elif strategy == AvoidanceStrategy.EMERGENCY_LAND:
            self.emergency_stop()
            return False
        
        waypoints = self.avoidance.calculate_avoidance_waypoints(current, target, obstacle, strategy)
        for wp in waypoints:
            if self._stop_exploration.is_set(): return False
            # On utilise navigate_to_waypoint pour l'évitement aussi (nose-first)
            if not self._navigate_to_waypoint((wp[0], wp[1])): return False
            # Ajustement altitude si nécessaire
            z_diff = wp[2] - self.controller.position.z
            if abs(z_diff) > 10:
                if z_diff > 0: self.controller.move_up(int(z_diff))
                else: self.controller.move_down(int(abs(z_diff)))
        return True

    def _record_exploration_data(self):
        if not self.config.enable_mapping: return
        pos = self.controller.position
        self.altitude_map.add_point(pos.x, pos.y, pos.z, self.controller.get_telemetry().get('tof_distance', pos.z))
        
        hazards = self.hazard_detector.check_hazards(pos.x, pos.y, pos.z)
        if hazards['alerts'] and self.on_hazard_detected:
            self.on_hazard_detected(pos, hazards)

    def add_simulated_obstacle(self, x, y, z, is_mobile=False, velocity=(0,0,0)):
        from obstacle_avoidance import DetectedObstacle
        obs = self.avoidance.add_obstacle(x, y, z, distance=((x**2+y**2+z**2)**0.5), direction=(x,y,z))
        if is_mobile:
            for i in range(5):
                obs.update_position(x + velocity[0]*i*0.1, y + velocity[1]*i*0.1, z + velocity[2]*i*0.1, obs.distance)

    def get_mission_report(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'status': self.status.value,
            'duration_seconds': elapsed,
            'waypoints': {'completed': self.waypoints_completed, 'total': self.total_waypoints},
            'mapping': {'coverage': self.altitude_map.get_exploration_coverage()},
            'obstacles': self.avoidance.get_status_report(),
            'drone': self.controller.get_telemetry()
        }

    def display_map(self):
        print(self.altitude_map.to_ascii_map((self.controller.position.x, self.controller.position.y)))

    def export_results(self, base_path: str):
        import json
        self.altitude_map.export_to_json(f"{base_path}_map.json")
        self.altitude_map.export_to_csv(f"{base_path}_points.csv")
        with open(f"{base_path}_report.json", 'w') as f:
            json.dump(self.get_mission_report(), f, indent=2, default=str)
