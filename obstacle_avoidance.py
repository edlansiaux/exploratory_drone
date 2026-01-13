#!/usr/bin/env python3
"""
Système d'évitement d'obstacles optimisé pour Tello EDU
Conçu pour réaction rapide en environnement délabré
"""

import numpy as np
import math
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AvoidanceStrategy(Enum):
    """Stratégies d'évitement"""
    NONE = "none"
    STOP = "stop"
    GO_LEFT = "go_left"
    GO_RIGHT = "go_right"
    GO_UP = "go_up"
    GO_DOWN = "go_down"
    BACKTRACK = "backtrack"
    GO_AROUND_LEFT = "go_around_left"
    GO_AROUND_RIGHT = "go_around_right"
    EMERGENCY_LAND = "emergency_land"


class ThreatLevel(Enum):
    """Niveaux de menace"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DetectedObstacle:
    """Obstacle détecté par les capteurs"""
    x: float
    y: float
    z: float
    distance: float
    direction: Tuple[float, float, float]  # Vecteur depuis drone
    obstacle_type: str = "unknown"
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    # Historique pour détection de mouvement
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def __post_init__(self):
        self.position_history.append((self.x, self.y, self.z, self.timestamp))
    
    @property
    def is_mobile(self) -> bool:
        """Détermine si l'obstacle est mobile"""
        if len(self.position_history) < 3:
            return False
        
        positions = list(self.position_history)
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dz = positions[i][2] - positions[i-1][2]
            total_movement += math.sqrt(dx**2 + dy**2 + dz**2)
        
        return total_movement > 15  # Seuil de mouvement
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Calcule la vélocité estimée"""
        if len(self.position_history) < 2:
            return (0, 0, 0)
        
        positions = list(self.position_history)
        vx, vy, vz = 0, 0, 0
        count = 0
        
        for i in range(1, len(positions)):
            dt = positions[i][3] - positions[i-1][3]
            if dt > 0:
                vx += (positions[i][0] - positions[i-1][0]) / dt
                vy += (positions[i][1] - positions[i-1][1]) / dt
                vz += (positions[i][2] - positions[i-1][2]) / dt
                count += 1
        
        if count > 0:
            return (vx/count, vy/count, vz/count)
        return (0, 0, 0)
    
    def predict_position(self, dt: float) -> Tuple[float, float, float]:
        """Prédit la position future"""
        vel = self.get_velocity()
        return (
            self.x + vel[0] * dt,
            self.y + vel[1] * dt,
            self.z + vel[2] * dt
        )
    
    def update_position(self, x: float, y: float, z: float, distance: float):
        """Met à jour la position"""
        self.x, self.y, self.z = x, y, z
        self.distance = distance
        self.timestamp = time.time()
        self.position_history.append((x, y, z, self.timestamp))


@dataclass
class SafetyZone:
    """Zone de sécurité configurable autour du drone"""
    front: float = 100.0
    back: float = 60.0
    left: float = 60.0
    right: float = 60.0
    above: float = 50.0
    below: float = 40.0
    
    # Zones d'urgence (réaction immédiate)
    emergency_front: float = 50.0
    emergency_sides: float = 30.0
    emergency_vertical: float = 30.0


class ObstacleAvoidanceSystem:
    """
    Système d'évitement d'obstacles à réaction rapide
    Optimisé pour environnements dégradés
    """
    
    # Temps de réaction cible (ms)
    TARGET_REACTION_TIME = 100
    
    def __init__(self, safety_zone: SafetyZone = None):
        self.safety_zone = safety_zone or SafetyZone()
        self.detected_obstacles: List[DetectedObstacle] = []
        
        # État
        self.is_active = False
        self.avoidance_in_progress = False
        self.current_strategy = AvoidanceStrategy.NONE
        
        # Paramètres
        self.detection_range = 250  # cm
        self.prediction_time = 1.5  # secondes
        self.obstacle_timeout = 3.0  # secondes
        
        # Thread de monitoring haute fréquence
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._monitor_rate = 50  # Hz (20ms)
        
        # Dernière décision
        self.last_decision_time = 0
        self.decision_cooldown = 0.1  # 100ms
        
        # Callbacks
        self.on_obstacle_detected: Optional[Callable] = None
        self.on_collision_imminent: Optional[Callable] = None
        self.on_strategy_selected: Optional[Callable] = None
        
        # Statistiques
        self.stats = {
            'obstacles_detected': 0,
            'collisions_avoided': 0,
            'avg_reaction_time_ms': 0,
            'reaction_times': deque(maxlen=100)
        }
        
        logger.info("ObstacleAvoidanceSystem initialisé")
    
    def start_monitoring(self):
        """Démarre le monitoring haute fréquence"""
        if self.is_active:
            return
        
        self.is_active = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"Monitoring démarré ({self._monitor_rate}Hz)")
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.is_active = False
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Monitoring arrêté")
    
    def _monitor_loop(self):
        """Boucle de monitoring haute fréquence"""
        interval = 1.0 / self._monitor_rate
        
        while not self._stop_monitoring.is_set():
            start = time.time()
            
            # Nettoyage obstacles anciens
            self._cleanup_old_obstacles()
            
            # Mise à jour prédictions mobiles
            self._update_mobile_predictions()
            
            # Calcul temps écoulé
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    def _cleanup_old_obstacles(self):
        """Supprime les obstacles non vus récemment"""
        current = time.time()
        self.detected_obstacles = [
            obs for obs in self.detected_obstacles
            if current - obs.timestamp < self.obstacle_timeout
        ]
    
    def _update_mobile_predictions(self):
        """Met à jour les prédictions pour obstacles mobiles"""
        for obs in self.detected_obstacles:
            if obs.is_mobile:
                # Log pour debug si nécessaire
                pass
    
    def add_obstacle(self, x: float, y: float, z: float,
                     distance: float, direction: Tuple[float, float, float],
                     obstacle_type: str = "unknown") -> DetectedObstacle:
        """
        Ajoute ou met à jour un obstacle
        
        Args:
            x, y, z: Position mondiale
            distance: Distance au drone
            direction: Vecteur direction depuis le drone
            obstacle_type: Type d'obstacle
            
        Returns:
            L'obstacle détecté
        """
        start_time = time.time()
        
        # Chercher obstacle existant proche
        for obs in self.detected_obstacles:
            dist_to_existing = math.sqrt(
                (obs.x - x)**2 + (obs.y - y)**2 + (obs.z - z)**2
            )
            if dist_to_existing < 60:  # Même obstacle
                obs.update_position(x, y, z, distance)
                return obs
        
        # Nouvel obstacle
        new_obs = DetectedObstacle(x, y, z, distance, direction, obstacle_type)
        self.detected_obstacles.append(new_obs)
        
        self.stats['obstacles_detected'] += 1
        
        if self.on_obstacle_detected:
            self.on_obstacle_detected(new_obs)
        
        # Mesurer temps de réaction
        reaction_time = (time.time() - start_time) * 1000
        self.stats['reaction_times'].append(reaction_time)
        if len(self.stats['reaction_times']) > 0:
            self.stats['avg_reaction_time_ms'] = sum(self.stats['reaction_times']) / len(self.stats['reaction_times'])
        
        logger.debug(f"Obstacle détecté: ({x:.0f},{y:.0f},{z:.0f}) dist={distance:.0f}cm [{obstacle_type}]")
        return new_obs
    
    def check_collision_risk(self, drone_x: float, drone_y: float, drone_z: float,
                             target_x: float, target_y: float, target_z: float,
                             drone_yaw: float = 0) -> Tuple[bool, Optional[DetectedObstacle], ThreatLevel]:
        """
        Vérifie le risque de collision sur une trajectoire
        
        Args:
            drone_x, drone_y, drone_z: Position du drone
            target_x, target_y, target_z: Position cible
            drone_yaw: Orientation du drone (degrés)
            
        Returns:
            (risque, obstacle, niveau_menace)
        """
        dx = target_x - drone_x
        dy = target_y - drone_y
        dz = target_z - drone_z
        
        trajectory_length = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if trajectory_length == 0:
            return False, None, ThreatLevel.NONE
        
        # Normaliser
        dx /= trajectory_length
        dy /= trajectory_length
        dz /= trajectory_length
        
        closest_obstacle = None
        min_distance = float('inf')
        threat_level = ThreatLevel.NONE
        
        for obs in self.detected_obstacles:
            # Position (prédite si mobile)
            if obs.is_mobile:
                time_to_reach = trajectory_length / 50  # ~50cm/s
                ox, oy, oz = obs.predict_position(time_to_reach)
            else:
                ox, oy, oz = obs.x, obs.y, obs.z
            
            # Vecteur drone -> obstacle
            vx = ox - drone_x
            vy = oy - drone_y
            vz = oz - drone_z
            
            # Projection sur trajectoire
            proj = vx*dx + vy*dy + vz*dz
            
            # Point le plus proche sur la trajectoire
            if proj < 0:
                closest = (drone_x, drone_y, drone_z)
            elif proj > trajectory_length:
                closest = (target_x, target_y, target_z)
            else:
                closest = (
                    drone_x + proj * dx,
                    drone_y + proj * dy,
                    drone_z + proj * dz
                )
            
            # Distance à l'obstacle
            dist = math.sqrt(
                (ox - closest[0])**2 +
                (oy - closest[1])**2 +
                (oz - closest[2])**2
            )
            
            # Déterminer zone de sécurité selon direction
            safety = self._get_safety_margin(vx, vy, vz, drone_yaw)
            
            if dist < safety:
                # Évaluer niveau de menace
                current_threat = self._evaluate_threat(dist, obs)
                
                if dist < min_distance:
                    min_distance = dist
                    closest_obstacle = obs
                    threat_level = current_threat
        
        collision_risk = closest_obstacle is not None
        
        if collision_risk and threat_level.value >= ThreatLevel.HIGH.value:
            if self.on_collision_imminent:
                self.on_collision_imminent(closest_obstacle, threat_level)
        
        return collision_risk, closest_obstacle, threat_level
    
    def _get_safety_margin(self, vx: float, vy: float, vz: float, 
                          drone_yaw: float) -> float:
        """Calcule la marge de sécurité selon la direction"""
        # Transformer vecteur en repère drone
        yaw_rad = math.radians(drone_yaw)
        local_x = vx * math.cos(yaw_rad) + vy * math.sin(yaw_rad)
        local_y = -vx * math.sin(yaw_rad) + vy * math.cos(yaw_rad)
        
        # Déterminer direction principale
        if abs(local_x) > abs(local_y) and abs(local_x) > abs(vz):
            # Avant/arrière
            return self.safety_zone.front if local_x > 0 else self.safety_zone.back
        elif abs(local_y) > abs(vz):
            # Gauche/droite
            return self.safety_zone.left if local_y < 0 else self.safety_zone.right
        else:
            # Haut/bas
            return self.safety_zone.above if vz > 0 else self.safety_zone.below
    
    def _evaluate_threat(self, distance: float, obstacle: DetectedObstacle) -> ThreatLevel:
        """Évalue le niveau de menace d'un obstacle"""
        # Distance critique
        if distance < self.safety_zone.emergency_front:
            return ThreatLevel.CRITICAL
        
        # Obstacle mobile rapide
        if obstacle.is_mobile:
            vel = obstacle.get_velocity()
            speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            if speed > 30:  # >30cm/s
                return ThreatLevel.CRITICAL if distance < 100 else ThreatLevel.HIGH
        
        # Types dangereux
        if obstacle.obstacle_type in ["fire", "hole"]:
            return ThreatLevel.CRITICAL
        
        # Par distance
        if distance < self.safety_zone.emergency_front:
            return ThreatLevel.CRITICAL
        elif distance < self.safety_zone.front * 0.5:
            return ThreatLevel.HIGH
        elif distance < self.safety_zone.front:
            return ThreatLevel.MEDIUM
        
        return ThreatLevel.LOW
    
    def get_avoidance_strategy(self, drone_pos: Tuple[float, float, float],
                               target_pos: Tuple[float, float, float],
                               obstacle: DetectedObstacle,
                               available_space: dict = None) -> AvoidanceStrategy:
        """
        Détermine la meilleure stratégie d'évitement
        
        Args:
            drone_pos: Position du drone
            target_pos: Position cible
            obstacle: Obstacle à éviter
            available_space: Espace disponible dans chaque direction
            
        Returns:
            Stratégie d'évitement
        """
        # Vérifier cooldown
        if time.time() - self.last_decision_time < self.decision_cooldown:
            return self.current_strategy
        
        self.last_decision_time = time.time()
        
        # Vecteur drone -> obstacle
        ox = obstacle.x - drone_pos[0]
        oy = obstacle.y - drone_pos[1]
        oz = obstacle.z - drone_pos[2]
        
        # Vecteur drone -> cible
        tx = target_pos[0] - drone_pos[0]
        ty = target_pos[1] - drone_pos[1]
        tz = target_pos[2] - drone_pos[2]
        
        # Obstacle mobile rapide : attendre
        if obstacle.is_mobile:
            vel = obstacle.get_velocity()
            speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            if speed > 30:
                logger.info("Obstacle mobile rapide - STOP")
                self.current_strategy = AvoidanceStrategy.STOP
                return AvoidanceStrategy.STOP
        
        # Distance critique : reculer
        if obstacle.distance < self.safety_zone.emergency_front:
            logger.info("Distance critique - BACKTRACK")
            self.stats['collisions_avoided'] += 1
            self.current_strategy = AvoidanceStrategy.BACKTRACK
            return AvoidanceStrategy.BACKTRACK
        
        # Type dangereux (feu, trou)
        if obstacle.obstacle_type in ["fire", "hole"]:
            self.stats['collisions_avoided'] += 1
            if abs(oz) > abs(ox) and abs(oz) > abs(oy):
                strategy = AvoidanceStrategy.GO_UP if oz > 0 else AvoidanceStrategy.GO_DOWN
            else:
                strategy = AvoidanceStrategy.GO_AROUND_RIGHT
            self.current_strategy = strategy
            return strategy
        
        # Choisir la meilleure direction
        # Produit vectoriel pour déterminer le côté
        cross_z = tx * oy - ty * ox
        
        if abs(oz) > abs(ox) and abs(oz) > abs(oy):
            # Obstacle principalement au-dessus/dessous
            strategy = AvoidanceStrategy.GO_DOWN if oz > 0 else AvoidanceStrategy.GO_UP
        else:
            # Obstacle sur le côté
            if cross_z > 0:
                strategy = AvoidanceStrategy.GO_AROUND_RIGHT
            else:
                strategy = AvoidanceStrategy.GO_AROUND_LEFT
        
        self.current_strategy = strategy
        
        if self.on_strategy_selected:
            self.on_strategy_selected(strategy, obstacle)
        
        return strategy
    
    def calculate_avoidance_path(self, drone_pos: Tuple[float, float, float],
                                 target_pos: Tuple[float, float, float],
                                 obstacle: DetectedObstacle,
                                 strategy: AvoidanceStrategy) -> List[Tuple[float, float, float]]:
        """
        Calcule le chemin d'évitement
        
        Returns:
            Liste de waypoints
        """
        waypoints = []
        avoidance_dist = 100  # cm
        
        if strategy == AvoidanceStrategy.STOP:
            return [drone_pos]
        
        elif strategy == AvoidanceStrategy.BACKTRACK:
            # Reculer de 50cm
            back_x = drone_pos[0] - 50 * (obstacle.x - drone_pos[0]) / max(obstacle.distance, 1)
            back_y = drone_pos[1] - 50 * (obstacle.y - drone_pos[1]) / max(obstacle.distance, 1)
            waypoints = [(back_x, back_y, drone_pos[2])]
        
        elif strategy in [AvoidanceStrategy.GO_AROUND_LEFT, AvoidanceStrategy.GO_AROUND_RIGHT]:
            # Point intermédiaire perpendiculaire
            mid_x = (drone_pos[0] + obstacle.x) / 2
            mid_y = (drone_pos[1] + obstacle.y) / 2
            
            dx = obstacle.x - drone_pos[0]
            dy = obstacle.y - drone_pos[1]
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                if strategy == AvoidanceStrategy.GO_AROUND_LEFT:
                    perp_x = -dy / length * avoidance_dist
                    perp_y = dx / length * avoidance_dist
                else:
                    perp_x = dy / length * avoidance_dist
                    perp_y = -dx / length * avoidance_dist
                
                waypoints = [
                    (mid_x + perp_x, mid_y + perp_y, drone_pos[2]),
                    target_pos
                ]
            else:
                waypoints = [target_pos]
        
        elif strategy == AvoidanceStrategy.GO_UP:
            waypoints = [
                (drone_pos[0], drone_pos[1], drone_pos[2] + avoidance_dist),
                (obstacle.x, obstacle.y, obstacle.z + avoidance_dist + 50),
                target_pos
            ]
        
        elif strategy == AvoidanceStrategy.GO_DOWN:
            safe_alt = max(40, drone_pos[2] - avoidance_dist)
            waypoints = [
                (drone_pos[0], drone_pos[1], safe_alt),
                (obstacle.x, obstacle.y, safe_alt),
                target_pos
            ]
        
        else:
            waypoints = [target_pos]
        
        return waypoints
    
    def check_immediate_danger(self, front_dist: float, height: float,
                               left_clear: bool = True, right_clear: bool = True,
                               up_clear: bool = True, down_clear: bool = True) -> Optional[AvoidanceStrategy]:
        """
        Vérification rapide de danger immédiat (réflexe)
        
        Args:
            front_dist: Distance frontale (capteur ToF)
            height: Hauteur actuelle
            left_clear, right_clear, up_clear, down_clear: Directions libres
            
        Returns:
            Action d'urgence ou None
        """
        # Obstacle frontal critique
        if front_dist < self.safety_zone.emergency_front:
            logger.warning(f"⚠️ DANGER FRONTAL: {front_dist:.0f}cm")
            self.stats['collisions_avoided'] += 1
            return AvoidanceStrategy.BACKTRACK
        
        # Trop bas
        if height < 30:
            logger.warning(f"⚠️ ALTITUDE CRITIQUE: {height:.0f}cm")
            return AvoidanceStrategy.GO_UP
        
        # Trop haut (plafond)
        if not up_clear and height > 250:
            logger.warning("⚠️ PLAFOND DÉTECTÉ")
            return AvoidanceStrategy.GO_DOWN
        
        return None
    
    def get_safe_direction(self, drone_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Trouve une direction sûre pour s'éloigner
        
        Returns:
            Vecteur direction normalisé ou None
        """
        if not self.detected_obstacles:
            return None
        
        # Force de répulsion combinée
        repulsion = [0.0, 0.0, 0.0]
        
        for obs in self.detected_obstacles:
            dx = drone_pos[0] - obs.x
            dy = drone_pos[1] - obs.y
            dz = drone_pos[2] - obs.z
            
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            if dist > 0:
                # Force inversement proportionnelle à la distance
                force = 1.0 / (dist + 1) ** 2
                repulsion[0] += dx * force / dist
                repulsion[1] += dy * force / dist
                repulsion[2] += dz * force / dist
        
        # Normaliser
        length = math.sqrt(sum(r**2 for r in repulsion))
        if length > 0:
            return tuple(r / length for r in repulsion)
        
        return None
    
    def get_status_report(self) -> dict:
        """Rapport d'état du système"""
        return {
            'is_active': self.is_active,
            'total_obstacles': len(self.detected_obstacles),
            'mobile_obstacles': sum(1 for o in self.detected_obstacles if o.is_mobile),
            'current_strategy': self.current_strategy.value,
            'avoidance_in_progress': self.avoidance_in_progress,
            'stats': {
                'obstacles_detected': self.stats['obstacles_detected'],
                'collisions_avoided': self.stats['collisions_avoided'],
                'avg_reaction_time_ms': self.stats['avg_reaction_time_ms']
            },
            'obstacles': [
                {
                    'position': (o.x, o.y, o.z),
                    'distance': o.distance,
                    'is_mobile': o.is_mobile,
                    'type': o.obstacle_type
                }
                for o in self.detected_obstacles
            ]
        }


class ReactiveAvoidance:
    """
    Couche d'évitement réactif ultra-rapide
    Réflexes immédiats pour dangers critiques
    """
    
    def __init__(self, emergency_distance: float = 40.0):
        self.emergency_distance = emergency_distance
        self.last_reaction = 0
        self.cooldown = 0.2  # 200ms
        
        # Compteurs
        self.reactions_count = 0
        self.last_reaction_type = None
    
    def check(self, front_dist: float, height: float,
              left_dist: float = 1000, right_dist: float = 1000,
              up_dist: float = 1000, down_dist: float = 1000) -> Optional[AvoidanceStrategy]:
        """
        Vérification réflexe ultra-rapide
        
        Returns:
            Action d'urgence ou None
        """
        now = time.time()
        if now - self.last_reaction < self.cooldown:
            return None
        
        action = None
        
        # Frontal
        if front_dist < self.emergency_distance:
            action = AvoidanceStrategy.BACKTRACK
            logger.warning(f"⚡ RÉFLEXE: Obstacle frontal {front_dist:.0f}cm")
        
        # Sol
        elif height < 25:
            action = AvoidanceStrategy.GO_UP
            logger.warning(f"⚡ RÉFLEXE: Sol proche {height:.0f}cm")
        
        # Latéral
        elif left_dist < self.emergency_distance * 0.7:
            action = AvoidanceStrategy.GO_RIGHT
            logger.warning(f"⚡ RÉFLEXE: Obstacle gauche {left_dist:.0f}cm")
        
        elif right_dist < self.emergency_distance * 0.7:
            action = AvoidanceStrategy.GO_LEFT
            logger.warning(f"⚡ RÉFLEXE: Obstacle droit {right_dist:.0f}cm")
        
        if action:
            self.last_reaction = now
            self.reactions_count += 1
            self.last_reaction_type = action
        
        return action


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Test ObstacleAvoidanceSystem ===\n")
    
    avoidance = ObstacleAvoidanceSystem()
    avoidance.start_monitoring()
    
    # Test ajout obstacles
    obs1 = avoidance.add_obstacle(100, 50, 100, 112, (0.89, 0.45, 0), "debris")
    obs2 = avoidance.add_obstacle(0, 100, 100, 100, (0, 1, 0), "person")
    
    # Simuler mouvement
    for i in range(5):
        time.sleep(0.1)
        obs2.update_position(obs2.x + 10, obs2.y + 5, obs2.z, obs2.distance)
    
    print(f"Obstacle 2 mobile: {obs2.is_mobile}")
    print(f"Vélocité: {obs2.get_velocity()}")
    
    # Test collision
    drone_pos = (0, 0, 100)
    target_pos = (150, 75, 100)
    
    risk, obstacle, threat = avoidance.check_collision_risk(*drone_pos, *target_pos)
    
    print(f"\nRisque collision: {risk}")
    print(f"Niveau menace: {threat.value}")
    
    if risk and obstacle:
        strategy = avoidance.get_avoidance_strategy(drone_pos, target_pos, obstacle)
        print(f"Stratégie: {strategy.value}")
        
        path = avoidance.calculate_avoidance_path(drone_pos, target_pos, obstacle, strategy)
        print(f"Chemin évitement: {path}")
    
    print(f"\nStatistiques: {avoidance.get_status_report()}")
    
    avoidance.stop_monitoring()
    print("\n✓ Test terminé")
