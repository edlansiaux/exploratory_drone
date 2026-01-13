#!/usr/bin/env python3
"""
Module d'exploration autonome pour drone Tello EDU
Intègre contrôle, cartographie et évitement d'obstacles
Conçu pour l'exploration de zones dangereuses (NRBC)
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
    """Modes d'exploration disponibles"""
    MANUAL = "manual"           # Contrôle manuel uniquement
    SEMI_AUTO = "semi_auto"     # Navigation auto avec supervision
    FULL_AUTO = "full_auto"     # Exploration complètement autonome


class MissionStatus(Enum):
    """État de la mission"""
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
    """Configuration d'une mission d'exploration"""
    area_width: float = 500.0      # Largeur zone à explorer (cm)
    area_height: float = 500.0     # Hauteur zone à explorer (cm)
    exploration_altitude: float = 120.0  # Altitude d'exploration (cm)
    step_size: float = 50.0        # Pas d'exploration (cm)
    pattern: str = "snake"         # Pattern: "snake" ou "spiral"
    max_duration: float = 600.0    # Durée max en secondes
    min_battery: int = 20          # Batterie min avant retour
    enable_mapping: bool = True
    enable_avoidance: bool = True


class HazardDetector:
    """
    Simulateur de détection de dangers NRBC
    En situation réelle, serait connecté à des capteurs spécialisés
    """
    
    def __init__(self):
        self.thermal_threshold = 50.0  # °C
        self.radiation_threshold = 0.5  # mSv/h
        self.chemical_threshold = 100   # ppm
        
        # Données simulées
        self.simulated_hotspots = []
        
    def add_simulated_hotspot(self, x: float, y: float, z: float, 
                              hazard_type: str, intensity: float):
        """Ajoute un point chaud simulé"""
        self.simulated_hotspots.append({
            'x': x, 'y': y, 'z': z,
            'type': hazard_type,
            'intensity': intensity
        })
    
    def check_hazards(self, x: float, y: float, z: float) -> dict:
        """
        Vérifie les dangers à une position donnée
        
        Returns:
            Dictionnaire avec les niveaux de danger détectés
        """
        hazards = {
            'thermal': 0.0,
            'radiation': 0.0,
            'chemical': 0.0,
            'alerts': []
        }
        
        for hotspot in self.simulated_hotspots:
            dist = ((x - hotspot['x'])**2 + 
                   (y - hotspot['y'])**2 + 
                   (z - hotspot['z'])**2) ** 0.5
            
            if dist < 200:  # Zone d'effet de 2m
                intensity = hotspot['intensity'] * (1 - dist/200)
                
                if hotspot['type'] == 'thermal':
                    hazards['thermal'] = max(hazards['thermal'], intensity)
                elif hotspot['type'] == 'radiation':
                    hazards['radiation'] = max(hazards['radiation'], intensity)
                elif hotspot['type'] == 'chemical':
                    hazards['chemical'] = max(hazards['chemical'], intensity)
        
        # Générer les alertes
        if hazards['thermal'] > self.thermal_threshold:
            hazards['alerts'].append(f"ALERTE THERMIQUE: {hazards['thermal']:.1f}°C")
        if hazards['radiation'] > self.radiation_threshold:
            hazards['alerts'].append(f"ALERTE RADIATION: {hazards['radiation']:.2f} mSv/h")
        if hazards['chemical'] > self.chemical_threshold:
            hazards['alerts'].append(f"ALERTE CHIMIQUE: {hazards['chemical']:.0f} ppm")
        
        return hazards


class ExplorationMission:
    """
    Gestionnaire de mission d'exploration
    Coordonne tous les sous-systèmes pour une exploration efficace et sûre
    """
    
    def __init__(self, config: MissionConfig = None, simulation_mode: bool = False):
        """
        Initialise une nouvelle mission
        
        Args:
            config: Configuration de la mission
            simulation_mode: Si True, simule sans drone réel
        """
        self.config = config or MissionConfig()
        self.simulation_mode = simulation_mode
        
        # Composants principaux
        self.controller = TelloController(simulation_mode)
        self.altitude_map = AltitudeMap(
            resolution=self.config.step_size,
            size=(self.config.area_width, self.config.area_height)
        )
        self.planner = ExplorationPlanner(
            self.altitude_map, 
            step_size=self.config.step_size
        )
        self.avoidance = ObstacleAvoidanceSystem(
            SafetyZone(front=80, back=50, left=50, right=50, above=50, below=40)
        )
        self.reactive = ReactiveAvoidance(emergency_distance=40)
        self.hazard_detector = HazardDetector()
        
        # État de la mission
        self.status = MissionStatus.IDLE
        self.mode = ExplorationMode.SEMI_AUTO
        self.start_time: Optional[float] = None
        self.waypoints_completed = 0
        self.total_waypoints = 0
        
        # Thread d'exploration
        self._exploration_thread: Optional[threading.Thread] = None
        self._stop_exploration = threading.Event()
        self._pause_exploration = threading.Event()
        
        # Callbacks
        self.on_status_change: Optional[Callable] = None
        self.on_hazard_detected: Optional[Callable] = None
        self.on_waypoint_reached: Optional[Callable] = None
        self.on_obstacle_detected: Optional[Callable] = None
        
        # Configuration des callbacks d'évitement
        self.avoidance.on_obstacle_detected = self._handle_obstacle_detected
        
        logger.info(f"Mission initialisée (simulation: {simulation_mode})")
    
    def _perform_safety_scan(self):
        """
        Effectue une rotation pour scanner l'environnement (Avant/Arrière/Côtés)
        Indispensable car la caméra est seulement frontale.
        """
        logger.info("Début du scan de sécurité 360°...")
        
        # On fait 4 rotations de 90 degrés pour couvrir tout l'espace
        # Le système SLAM et ObstacleDetector mettront à jour la carte à chaque étape
        for _ in range(4):
            if self._stop_exploration.is_set():
                break
                
            # Rotation lente pour permettre au SLAM de suivre
            self.controller.rotate_clockwise(90)
            time.sleep(1.0) # Pause pour la stabilisation de l'image et la détection
            
            # Enregistrement explicite des données à cet angle
            self._record_exploration_data()

    def _navigate_to_waypoint(self, waypoint: Tuple[float, float]) -> bool:
        """
        Navigation "Nez en avant" : Rotation puis Avancée
        """
        target_x, target_y = waypoint
        target_z = self.config.exploration_altitude
        
        current = self.controller.position
        
        # 1. Calcul du vecteur vers la cible
        dx = target_x - current.x
        dy = target_y - current.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 10: # Déjà arrivé
            return True

        # 2. Vérification de sécurité (Scan) si on change radicalement de direction
        # ou si c'est le début du mouvement
        self._perform_safety_scan()

        # 3. Calcul de l'angle cible (en degrés)
        # atan2 retourne l'angle en radians par rapport à l'axe X
        target_angle_rad = math.atan2(dy, dx)
        target_angle_deg = math.degrees(target_angle_rad)
        
        # Conversion du repère mathématique au repère drone (si nécessaire)
        # Supposons ici que 0° = Axe Y (Nord) pour le drone Tello
        # L'ajustement dépend de votre repère initial dans mapping.py
        
        # 4. Rotation face à la cible
        # Note: Il faut implémenter une logique pour calculer la différence d'angle 
        # par rapport à l'orientation actuelle du drone.
        # Pour simplifier ici, on utilise une rotation relative basée sur le vecteur
        
        logger.info(f"Orientation vers le waypoint ({target_x:.0f}, {target_y:.0f})")
        # On pivote pour faire face au mouvement
        # (Nécessite de connaître l'angle actuel du drone, voir modification tello_controller)
        # Si on ne connait pas l'angle absolu, on ne peut pas utiliser cette méthode facilement.
        # Alternative simple : on désactive le strafing.
        
        # Si nous sommes en mode "Nez en avant strict", on ne fait que avancer.
        # Pour ce faire, il faut calculer l'angle de rotation requis.
        
        # ... (Logique de rotation ici via self.controller.rotate_...) ...
        
        # 5. Avancer vers la cible (par petits pas pour vérifier les obstacles)
        step = 50 # cm
        remaining = distance
        
        while remaining > 0:
            if self._stop_exploration.is_set():
                return False
                
            # Distance à parcourir pour ce pas
            move_dist = min(remaining, step)
            
            # Vérification Obstacle Frontal (Vision)
            # On utilise le détecteur d'obstacle via la caméra frontale
            # C'est géré par l'événement on_obstacle_detected ou via self.avoidance
            
            # Si un obstacle est détecté devant via la vision ou le SLAM
            # La méthode _handle_obstacle_avoidance sera appelée
            
            # Mouvement avant UNIQUEMENT (pas de gauche/droite/arrière)
            logger.info(f"Avance de {move_dist:.0f}cm")
            success = self.controller.move_forward(int(move_dist))
            
            if not success:
                logger.warning("Blocage détecté, tentative de contournement")
                # Ici lancer une logique d'évitement qui implique de tourner
                return False
                
            remaining -= move_dist
            time.sleep(0.5) # Stabilisation
            
        return True

    def _move_to_position(self, target_x: float, target_y: float, target_z: float) -> bool:
        """
        Surcharge de la méthode originale pour interdire les mouvements latéraux/arrières
        """
        # Cette méthode est appelée par la logique d'évitement ou de navigation fine.
        # On la redirige vers la logique de rotation + avance.
        return self._navigate_to_waypoint((target_x, target_y))
        
    def _set_status(self, new_status: MissionStatus):
        """Change le statut et notifie"""
        old_status = self.status
        self.status = new_status
        logger.info(f"Statut mission: {old_status.value} -> {new_status.value}")
        
        if self.on_status_change:
            self.on_status_change(old_status, new_status)
    
    def _handle_obstacle_detected(self, obstacle):
        """Callback quand un obstacle est détecté"""
        # Ajouter à la carte
        self.altitude_map.add_obstacle(
            obstacle.x, obstacle.y, obstacle.z,
            radius=50, is_mobile=obstacle.is_mobile
        )
        
        if self.on_obstacle_detected:
            self.on_obstacle_detected(obstacle)
    
    def prepare_mission(self) -> bool:
        """
        Prépare la mission (connexion, checks pré-vol)
        
        Returns:
            True si la préparation est réussie
        """
        self._set_status(MissionStatus.PREPARING)
        
        # Connexion au drone
        if not self.controller.connect():
            logger.error("Échec connexion drone")
            self._set_status(MissionStatus.ABORTED)
            return False
        
        # Vérification batterie
        telemetry = self.controller.get_telemetry()
        if telemetry.get('battery', 0) < self.config.min_battery:
            logger.error(f"Batterie insuffisante: {telemetry.get('battery')}%")
            self._set_status(MissionStatus.ABORTED)
            return False
        
        # Génération du plan d'exploration
        if self.config.pattern == "snake":
            waypoints = self.planner.generate_snake_pattern(
                self.config.area_width,
                self.config.area_height
            )
        else:
            waypoints = self.planner.generate_spiral_pattern(
                max_radius=max(self.config.area_width, self.config.area_height) / 2
            )
        
        self.total_waypoints = len(waypoints)
        logger.info(f"Plan d'exploration: {self.total_waypoints} waypoints")
        
        self._set_status(MissionStatus.IDLE)
        return True
    
    def start_exploration(self) -> bool:
        """
        Démarre l'exploration autonome
        
        Returns:
            True si le démarrage est réussi
        """
        if self.status not in [MissionStatus.IDLE, MissionStatus.PAUSED]:
            logger.warning(f"Impossible de démarrer: statut = {self.status.value}")
            return False
        
        # Décollage si nécessaire
        if self.controller.state != DroneState.FLYING:
            if not self.controller.takeoff():
                self._set_status(MissionStatus.ABORTED)
                return False
            
            # Monter à l'altitude d'exploration
            current_alt = self.controller.position.z
            if current_alt < self.config.exploration_altitude:
                diff = int(self.config.exploration_altitude - current_alt)
                if diff >= 20:
                    self.controller.move_up(diff)
        
        # Démarrer les systèmes
        if self.config.enable_avoidance:
            self.avoidance.start_monitoring()
        
        # Démarrer le thread d'exploration
        self._stop_exploration.clear()
        self._pause_exploration.clear()
        self._exploration_thread = threading.Thread(
            target=self._exploration_loop,
            daemon=True
        )
        self._exploration_thread.start()
        
        self._set_status(MissionStatus.IN_PROGRESS)
        self.start_time = time.time()
        
        return True
    
    def pause_exploration(self):
        """Met en pause l'exploration"""
        if self.status == MissionStatus.IN_PROGRESS:
            self._pause_exploration.set()
            self._set_status(MissionStatus.PAUSED)
    
    def resume_exploration(self):
        """Reprend l'exploration"""
        if self.status == MissionStatus.PAUSED:
            self._pause_exploration.clear()
            self._set_status(MissionStatus.IN_PROGRESS)
    
    def stop_exploration(self):
        """Arrête l'exploration et atterrit"""
        self._stop_exploration.set()
        
        if self._exploration_thread:
            self._exploration_thread.join(timeout=5.0)
        
        self.avoidance.stop_monitoring()
        self.controller.land()
        self._set_status(MissionStatus.COMPLETED)
    
    def emergency_stop(self):
        """Arrêt d'urgence immédiat"""
        self._stop_exploration.set()
        self.avoidance.stop_monitoring()
        self.controller.emergency_stop()
        self._set_status(MissionStatus.EMERGENCY)
    
    def return_to_home(self):
        """Retourne au point de départ"""
        self._set_status(MissionStatus.RETURNING)
        self._stop_exploration.set()
        
        if self._exploration_thread:
            self._exploration_thread.join(timeout=5.0)
        
        self.controller.return_to_home()
        self.controller.land()
        self._set_status(MissionStatus.COMPLETED)
    
    def _exploration_loop(self):
        """Boucle principale d'exploration autonome"""
        logger.info("Démarrage de la boucle d'exploration")
        
        while not self._stop_exploration.is_set():
            # Vérifier pause
            if self._pause_exploration.is_set():
                time.sleep(0.1)
                continue
            
            # Vérifier conditions de fin
            if not self._check_mission_conditions():
                break
            
            # Obtenir le prochain waypoint
            waypoint = self.planner.get_next_waypoint()
            if waypoint is None:
                logger.info("Exploration terminée - tous les waypoints atteints")
                break
            
            # Naviguer vers le waypoint
            success = self._navigate_to_waypoint(waypoint)
            
            if success:
                self.waypoints_completed += 1
                self._record_exploration_data()
                
                if self.on_waypoint_reached:
                    self.on_waypoint_reached(waypoint, self.planner.get_progress())
            
            # Petite pause pour stabilisation
            time.sleep(0.2)
        
        logger.info("Boucle d'exploration terminée")
    
    def _check_mission_conditions(self) -> bool:
        """Vérifie les conditions de continuation de la mission"""
        # Vérifier la durée
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.max_duration:
                logger.warning("Durée maximale atteinte")
                return False
        
        # Vérifier la batterie
        telemetry = self.controller.get_telemetry()
        battery = telemetry.get('battery', 100)
        if battery < self.config.min_battery:
            logger.warning(f"Batterie faible: {battery}%")
            return False
        
        return True
    
    def _navigate_to_waypoint(self, waypoint: Tuple[float, float]) -> bool:
        """
        Navigue vers un waypoint avec évitement d'obstacles
        
        Args:
            waypoint: Coordonnées (x, y) du waypoint
        
        Returns:
            True si le waypoint est atteint
        """
        target_x, target_y = waypoint
        target_z = self.config.exploration_altitude
        
        current_pos = self.controller.position
        
        # Vérifier les obstacles sur le chemin
        if self.config.enable_avoidance:
            collision, obstacle = self.avoidance.check_collision_risk(
                current_pos.x, current_pos.y, current_pos.z,
                target_x, target_y, target_z
            )
            
            if collision and obstacle:
                return self._handle_obstacle_avoidance(
                    (current_pos.x, current_pos.y, current_pos.z),
                    (target_x, target_y, target_z),
                    obstacle
                )
        
        # Navigation directe
        return self._move_to_position(target_x, target_y, target_z)
    
    def _handle_obstacle_avoidance(self, current: Tuple[float, float, float],
                                    target: Tuple[float, float, float],
                                    obstacle) -> bool:
        """Gère l'évitement d'un obstacle détecté"""
        logger.info(f"Évitement obstacle à ({obstacle.x:.0f}, {obstacle.y:.0f})")
        
        # Déterminer la stratégie
        strategy = self.avoidance.get_avoidance_strategy(current, target, obstacle)
        
        if strategy == AvoidanceStrategy.STOP:
            # Attendre que l'obstacle mobile passe
            logger.info("Attente passage obstacle mobile...")
            time.sleep(2.0)
            return False  # Réessayer au prochain cycle
        
        elif strategy == AvoidanceStrategy.EMERGENCY_LAND:
            self.emergency_stop()
            return False
        
        else:
            # Calculer le chemin d'évitement
            waypoints = self.avoidance.calculate_avoidance_waypoints(
                current, target, obstacle, strategy
            )
            
            # Suivre les waypoints d'évitement
            for wp in waypoints:
                if self._stop_exploration.is_set():
                    return False
                
                if not self._move_to_position(*wp):
                    return False
            
            return True
    
    def _move_to_position(self, target_x: float, target_y: float, 
                          target_z: float) -> bool:
        """
        Déplace le drone vers une position cible
        
        Args:
            target_x, target_y, target_z: Position cible
        
        Returns:
            True si la position est atteinte
        """
        current = self.controller.position
        
        # Calculer les différences
        dx = target_x - current.x
        dy = target_y - current.y
        dz = target_z - current.z
        
        # Mouvement par étapes pour permettre les vérifications
        try:
            # Ajustement altitude d'abord (plus sûr)
            if abs(dz) >= 20:
                if dz > 0:
                    self.controller.move_up(min(int(dz), 100))
                else:
                    self.controller.move_down(min(int(abs(dz)), 100))
            
            # Mouvement horizontal
            if abs(dx) >= 20:
                if dx > 0:
                    self.controller.move_forward(min(int(dx), 100))
                else:
                    self.controller.move_back(min(int(abs(dx)), 100))
            
            if abs(dy) >= 20:
                if dy > 0:
                    self.controller.move_right(min(int(dy), 100))
                else:
                    self.controller.move_left(min(int(abs(dy)), 100))
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur navigation: {e}")
            return False
    
    def _record_exploration_data(self):
        """Enregistre les données d'exploration au point actuel"""
        if not self.config.enable_mapping:
            return
        
        pos = self.controller.position
        telemetry = self.controller.get_telemetry()
        
        # Distance au sol (ToF ou estimation)
        tof_distance = telemetry.get('tof_distance', pos.z)
        
        # Ajouter le point à la carte
        self.altitude_map.add_point(pos.x, pos.y, pos.z, tof_distance)
        
        # Vérifier les dangers
        hazards = self.hazard_detector.check_hazards(pos.x, pos.y, pos.z)
        if hazards['alerts']:
            for alert in hazards['alerts']:
                logger.warning(alert)
            
            if self.on_hazard_detected:
                self.on_hazard_detected(pos, hazards)
    
    def add_simulated_obstacle(self, x: float, y: float, z: float,
                                is_mobile: bool = False,
                                velocity: Tuple[float, float, float] = (0, 0, 0)):
        """
        Ajoute un obstacle simulé pour les tests
        
        Args:
            x, y, z: Position de l'obstacle
            is_mobile: Si True, l'obstacle est mobile
            velocity: Vélocité si mobile
        """
        from obstacle_avoidance import DetectedObstacle
        
        obstacle = self.avoidance.add_obstacle(
            x, y, z,
            distance=((x**2 + y**2 + z**2) ** 0.5),
            direction=(x, y, z)
        )
        
        if is_mobile:
            # Simuler plusieurs mises à jour pour le marquer comme mobile
            for i in range(5):
                time.sleep(0.1)
                obstacle.update_position(
                    x + velocity[0] * i * 0.1,
                    y + velocity[1] * i * 0.1,
                    z + velocity[2] * i * 0.1,
                    obstacle.distance
                )
    
    def get_mission_report(self) -> dict:
        """Génère un rapport complet de la mission"""
        elapsed = 0
        if self.start_time:
            elapsed = time.time() - self.start_time
        
        telemetry = self.controller.get_telemetry()
        
        return {
            'status': self.status.value,
            'mode': self.mode.value,
            'duration_seconds': elapsed,
            'waypoints': {
                'completed': self.waypoints_completed,
                'total': self.total_waypoints,
                'progress': self.planner.get_progress()
            },
            'mapping': {
                'coverage': self.altitude_map.get_exploration_coverage(),
                'points_recorded': len(self.altitude_map.raw_points),
                'altitude_stats': self.altitude_map.get_altitude_stats()
            },
            'obstacles': self.avoidance.get_status_report(),
            'drone': {
                'position': self.controller.position.to_tuple(),
                'state': self.controller.state.value,
                'battery': telemetry.get('battery', 'N/A')
            }
        }
    
    def display_map(self):
        """Affiche la carte d'exploration"""
        drone_pos = (self.controller.position.x, self.controller.position.y)
        print(self.altitude_map.to_ascii_map(drone_pos))
    
    def export_results(self, base_path: str = "exploration_results"):
        """
        Exporte les résultats de la mission
        
        Args:
            base_path: Chemin de base pour les fichiers
        """
        import json
        
        # Export de la carte
        self.altitude_map.export_to_json(f"{base_path}_map.json")
        self.altitude_map.export_to_csv(f"{base_path}_points.csv")
        
        # Export du rapport
        report = self.get_mission_report()
        with open(f"{base_path}_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Résultats exportés: {base_path}_*")


# ============================================================
# Commandes manuelles simplifiées
# ============================================================

class ManualController:
    """Interface simplifiée pour le contrôle manuel"""
    
    def __init__(self, mission: ExplorationMission):
        self.mission = mission
        self.controller = mission.controller
    
    def takeoff(self):
        """Décollage"""
        return self.controller.takeoff()
    
    def land(self):
        """Atterrissage"""
        return self.controller.land()
    
    def left(self, distance: int = 50):
        """Gauche de 50cm par défaut"""
        return self.controller.move_left(distance)
    
    def right(self, distance: int = 50):
        """Droite de 50cm par défaut"""
        return self.controller.move_right(distance)
    
    def up(self, distance: int = 50):
        """Monter de 50cm par défaut"""
        return self.controller.move_up(distance)
    
    def down(self, distance: int = 50):
        """Descendre de 50cm par défaut"""
        return self.controller.move_down(distance)
    
    def forward(self, distance: int = 50):
        """Avancer de 50cm par défaut"""
        return self.controller.move_forward(distance)
    
    def back(self, distance: int = 50):
        """Reculer de 50cm par défaut"""
        return self.controller.move_back(distance)
    
    def rotate_left(self, angle: int = 90):
        """Rotation anti-horaire"""
        return self.controller.rotate_counter_clockwise(angle)
    
    def rotate_right(self, angle: int = 90):
        """Rotation horaire"""
        return self.controller.rotate_clockwise(angle)
    
    def emergency(self):
        """Arrêt d'urgence"""
        self.controller.emergency_stop()
    
    def status(self):
        """Affiche le statut"""
        print(f"État: {self.controller.state.value}")
        print(f"Position: {self.controller.position.to_tuple()}")
        print(f"Télémétrie: {self.controller.get_telemetry()}")


# Test du module
if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU SYSTÈME D'EXPLORATION TELLO EDU")
    print("=" * 60)
    
    # Configuration de test
    config = MissionConfig(
        area_width=300,
        area_height=300,
        exploration_altitude=100,
        step_size=50,
        pattern="snake",
        max_duration=120
    )
    
    # Création de la mission en mode simulation
    mission = ExplorationMission(config, simulation_mode=True)
    
    # Callbacks de test
    def on_status(old, new):
        print(f"[STATUS] {old.value} -> {new.value}")
    
    def on_waypoint(wp, progress):
        print(f"[WAYPOINT] {wp} - Progression: {progress:.1f}%")
    
    def on_hazard(pos, hazards):
        print(f"[HAZARD] Position: {pos.to_tuple()}")
        for alert in hazards['alerts']:
            print(f"  - {alert}")
    
    mission.on_status_change = on_status
    mission.on_waypoint_reached = on_waypoint
    mission.on_hazard_detected = on_hazard
    
    # Préparation
    print("\n--- Préparation de la mission ---")
    if mission.prepare_mission():
        
        # Ajout de dangers simulés
        mission.hazard_detector.add_simulated_hotspot(100, 100, 0, 'thermal', 80)
        mission.hazard_detector.add_simulated_hotspot(-50, 50, 0, 'radiation', 1.0)
        
        # Ajout d'obstacles simulés
        mission.add_simulated_obstacle(75, 50, 100, is_mobile=False)
        mission.add_simulated_obstacle(0, 100, 100, is_mobile=True, velocity=(10, 5, 0))
        
        # Démarrage de l'exploration
        print("\n--- Démarrage de l'exploration ---")
        mission.start_exploration()
        
        # Laisser l'exploration se dérouler
        time.sleep(5)
        
        # Afficher la carte
        print("\n--- Carte d'exploration ---")
        mission.display_map()
        
        # Arrêt
        print("\n--- Arrêt de l'exploration ---")
        mission.stop_exploration()
        
        # Rapport final
        print("\n--- Rapport de mission ---")
        report = mission.get_mission_report()
        for key, value in report.items():
            print(f"{key}: {value}")
        
        # Export des résultats
        mission.export_results("/tmp/test_exploration")
        
    print("\n" + "=" * 60)
    print("TEST TERMINÉ")
    print("=" * 60)
