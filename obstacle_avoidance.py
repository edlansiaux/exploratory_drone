#!/usr/bin/env python3
"""
Module d'évitement d'obstacles pour drone Tello EDU
Gère les obstacles statiques et mobiles avec prédiction de trajectoire
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
import time
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)


class AvoidanceStrategy(Enum):
    """Stratégies d'évitement disponibles"""
    STOP = "stop"                    # Arrêt et attente
    GO_AROUND_LEFT = "go_around_left"
    GO_AROUND_RIGHT = "go_around_right"
    GO_ABOVE = "go_above"
    GO_BELOW = "go_below"
    BACKTRACK = "backtrack"          # Recule et recalcule
    EMERGENCY_LAND = "emergency_land"


@dataclass
class DetectedObstacle:
    """Obstacle détecté par les capteurs"""
    x: float
    y: float
    z: float
    distance: float  # Distance au drone
    direction: Tuple[float, float, float]  # Vecteur direction depuis le drone
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    # Historique pour suivi de mouvement
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def __post_init__(self):
        self.position_history.append((self.x, self.y, self.z, self.timestamp))
    
    @property
    def is_mobile(self) -> bool:
        """Détermine si l'obstacle est mobile basé sur son historique"""
        if len(self.position_history) < 3:
            return False
        
        # Calcul du déplacement total
        positions = list(self.position_history)
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dz = positions[i][2] - positions[i-1][2]
            total_movement += np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Seuil de mouvement: 10cm
        return total_movement > 10
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Calcule la vélocité estimée de l'obstacle"""
        if len(self.position_history) < 2:
            return (0, 0, 0)
        
        positions = list(self.position_history)
        # Moyenne des dernières vélocités
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
        """Prédit la position future de l'obstacle"""
        vel = self.get_velocity()
        return (
            self.x + vel[0] * dt,
            self.y + vel[1] * dt,
            self.z + vel[2] * dt
        )
    
    def update_position(self, x: float, y: float, z: float, distance: float):
        """Met à jour la position de l'obstacle"""
        self.x = x
        self.y = y
        self.z = z
        self.distance = distance
        self.timestamp = time.time()
        self.position_history.append((x, y, z, self.timestamp))


@dataclass
class SafetyZone:
    """Zone de sécurité autour du drone"""
    front: float = 100.0   # Distance de sécurité avant (cm)
    back: float = 50.0     # Distance de sécurité arrière
    left: float = 50.0     # Distance de sécurité gauche
    right: float = 50.0    # Distance de sécurité droite
    above: float = 50.0    # Distance de sécurité dessus
    below: float = 50.0    # Distance de sécurité dessous


class ObstacleAvoidanceSystem:
    """
    Système d'évitement d'obstacles intelligent
    Gère les obstacles statiques et mobiles avec prédiction
    """
    
    def __init__(self, safety_zone: SafetyZone = None):
        """
        Initialise le système d'évitement
        
        Args:
            safety_zone: Zones de sécurité personnalisées
        """
        self.safety_zone = safety_zone or SafetyZone()
        self.detected_obstacles: List[DetectedObstacle] = []
        self.is_active = False
        self.avoidance_in_progress = False
        
        # Callbacks
        self.on_obstacle_detected: Optional[Callable] = None
        self.on_collision_imminent: Optional[Callable] = None
        self.on_avoidance_complete: Optional[Callable] = None
        
        # Paramètres de détection
        self.detection_range = 200  # cm
        self.prediction_time = 2.0  # secondes pour la prédiction
        self.obstacle_timeout = 5.0  # secondes avant d'oublier un obstacle
        
        # Thread de monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        logger.info("Système d'évitement initialisé")
    
    def start_monitoring(self):
        """Démarre le monitoring continu des obstacles"""
        if self.is_active:
            return
        
        self.is_active = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Monitoring des obstacles démarré")
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.is_active = False
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Monitoring des obstacles arrêté")
    
    def _monitor_loop(self):
        """Boucle de monitoring des obstacles"""
        while not self._stop_monitoring.is_set():
            self._cleanup_old_obstacles()
            self._check_mobile_obstacles()
            time.sleep(0.1)  # 10 Hz
    

    
    def _cleanup_old_obstacles(self):
        """Supprime les obstacles non détectés récemment"""
        current_time = time.time()
        self.detected_obstacles = [
            obs for obs in self.detected_obstacles 
            if current_time - obs.timestamp < self.obstacle_timeout
        ]
    
    def _check_mobile_obstacles(self):
        """Vérifie les obstacles mobiles et met à jour les prédictions"""
        for obs in self.detected_obstacles:
            if obs.is_mobile:
                # Prédire la trajectoire
                future_pos = obs.predict_position(self.prediction_time)
                logger.debug(f"Obstacle mobile prédit à {future_pos}")
    
    def add_obstacle(self, x: float, y: float, z: float, 
                     distance: float, direction: Tuple[float, float, float]) -> DetectedObstacle:
        """
        Ajoute ou met à jour un obstacle détecté
        
        Args:
            x, y, z: Position de l'obstacle
            distance: Distance au drone
            direction: Direction depuis le drone
        
        Returns:
            L'obstacle détecté ou mis à jour
        """
        # Chercher un obstacle existant proche
        for obs in self.detected_obstacles:
            dist_to_existing = np.sqrt(
                (obs.x - x)**2 + (obs.y - y)**2 + (obs.z - z)**2
            )
            if dist_to_existing < 50:  # Même obstacle si <50cm
                obs.update_position(x, y, z, distance)
                return obs
        
        # Nouvel obstacle
        new_obs = DetectedObstacle(x, y, z, distance, direction)
        self.detected_obstacles.append(new_obs)
        
        if self.on_obstacle_detected:
            self.on_obstacle_detected(new_obs)
        
        logger.info(f"Nouvel obstacle détecté à ({x:.0f}, {y:.0f}, {z:.0f}), distance: {distance:.0f}cm")
        return new_obs
    
    def check_collision_risk(self, drone_x: float, drone_y: float, drone_z: float,
                             target_x: float, target_y: float, target_z: float) -> Tuple[bool, Optional[DetectedObstacle]]:
        """
        Vérifie le risque de collision sur une trajectoire
        
        Args:
            drone_x, drone_y, drone_z: Position actuelle du drone
            target_x, target_y, target_z: Position cible
        
        Returns:
            (risque_collision, obstacle_le_plus_proche)
        """
        # Vecteur de trajectoire
        dx = target_x - drone_x
        dy = target_y - drone_y
        dz = target_z - drone_z
        trajectory_length = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if trajectory_length == 0:
            return False, None
        
        # Normaliser
        dx /= trajectory_length
        dy /= trajectory_length
        dz /= trajectory_length
        
        closest_obstacle = None
        min_distance = float('inf')
        
        for obs in self.detected_obstacles:
            # Pour les obstacles mobiles, utiliser la position prédite
            if obs.is_mobile:
                # Temps estimé pour atteindre cette zone
                time_to_reach = trajectory_length / 50  # ~50cm/s vitesse moyenne
                ox, oy, oz = obs.predict_position(time_to_reach)
            else:
                ox, oy, oz = obs.x, obs.y, obs.z
            
            # Distance point-droite (trajectoire)
            # Vecteur drone -> obstacle
            vx = ox - drone_x
            vy = oy - drone_y
            vz = oz - drone_z
            
            # Projection sur la trajectoire
            proj = vx*dx + vy*dy + vz*dz
            
            # Point le plus proche sur la trajectoire
            if proj < 0:
                closest_x, closest_y, closest_z = drone_x, drone_y, drone_z
            elif proj > trajectory_length:
                closest_x, closest_y, closest_z = target_x, target_y, target_z
            else:
                closest_x = drone_x + proj * dx
                closest_y = drone_y + proj * dy
                closest_z = drone_z + proj * dz
            
            # Distance à l'obstacle
            dist = np.sqrt(
                (ox - closest_x)**2 + 
                (oy - closest_y)**2 + 
                (oz - closest_z)**2
            )
            
            # Marge de sécurité
            safety_margin = max(
                self.safety_zone.front,
                self.safety_zone.left,
                self.safety_zone.right
            )
            
            if dist < safety_margin and dist < min_distance:
                min_distance = dist
                closest_obstacle = obs
        
        return closest_obstacle is not None, closest_obstacle
    
    def get_avoidance_strategy(self, drone_pos: Tuple[float, float, float],
                               target_pos: Tuple[float, float, float],
                               obstacle: DetectedObstacle) -> AvoidanceStrategy:
        """
        Détermine la meilleure stratégie d'évitement
        
        Args:
            drone_pos: Position actuelle du drone
            target_pos: Position cible
            obstacle: Obstacle à éviter
        
        Returns:
            Stratégie d'évitement recommandée
        """
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        dz = target_pos[2] - drone_pos[2]
        
        # Vecteur vers l'obstacle
        ox = obstacle.x - drone_pos[0]
        oy = obstacle.y - drone_pos[1]
        oz = obstacle.z - drone_pos[2]
        
        # Si l'obstacle est mobile et rapide, stratégie d'attente
        if obstacle.is_mobile:
            vel = obstacle.get_velocity()
            speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            if speed > 20:  # >20cm/s
                logger.info("Obstacle mobile rapide détecté - stratégie STOP")
                return AvoidanceStrategy.STOP
        
        # Calculer le produit vectoriel pour déterminer le côté
        cross_z = dx * oy - dy * ox
        
        # Déterminer la meilleure direction
        if abs(oz) > abs(ox) and abs(oz) > abs(oy):
            # Obstacle principalement au-dessus ou en-dessous
            if oz > 0:
                return AvoidanceStrategy.GO_BELOW
            else:
                return AvoidanceStrategy.GO_ABOVE
        else:
            # Obstacle sur le côté
            if cross_z > 0:
                return AvoidanceStrategy.GO_AROUND_RIGHT
            else:
                return AvoidanceStrategy.GO_AROUND_LEFT
    
    def calculate_avoidance_waypoints(self, drone_pos: Tuple[float, float, float],
                                       target_pos: Tuple[float, float, float],
                                       obstacle: DetectedObstacle,
                                       strategy: AvoidanceStrategy) -> List[Tuple[float, float, float]]:
        """
        Calcule les waypoints pour contourner un obstacle
        
        Args:
            drone_pos: Position actuelle
            target_pos: Position cible
            obstacle: Obstacle à éviter
            strategy: Stratégie choisie
        
        Returns:
            Liste de waypoints pour l'évitement
        """
        waypoints = []
        avoidance_distance = 120  # Distance de contournement en cm
        
        if strategy == AvoidanceStrategy.STOP:
            # Rester sur place et attendre
            return [drone_pos]
        
        elif strategy == AvoidanceStrategy.GO_AROUND_LEFT:
            # Contournement par la gauche
            mid_x = (drone_pos[0] + obstacle.x) / 2
            mid_y = (drone_pos[1] + obstacle.y) / 2
            
            # Perpendiculaire à la direction de l'obstacle
            dx = obstacle.x - drone_pos[0]
            dy = obstacle.y - drone_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                perp_x = -dy / length * avoidance_distance
                perp_y = dx / length * avoidance_distance
            else:
                perp_x, perp_y = avoidance_distance, 0
            
            waypoints = [
                (mid_x + perp_x, mid_y + perp_y, drone_pos[2]),
                target_pos
            ]
        
        elif strategy == AvoidanceStrategy.GO_AROUND_RIGHT:
            # Contournement par la droite
            mid_x = (drone_pos[0] + obstacle.x) / 2
            mid_y = (drone_pos[1] + obstacle.y) / 2
            
            dx = obstacle.x - drone_pos[0]
            dy = obstacle.y - drone_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                perp_x = dy / length * avoidance_distance
                perp_y = -dx / length * avoidance_distance
            else:
                perp_x, perp_y = -avoidance_distance, 0
            
            waypoints = [
                (mid_x + perp_x, mid_y + perp_y, drone_pos[2]),
                target_pos
            ]
        
        elif strategy == AvoidanceStrategy.GO_ABOVE:
            # Passer au-dessus
            waypoints = [
                (drone_pos[0], drone_pos[1], obstacle.z + avoidance_distance),
                (obstacle.x, obstacle.y, obstacle.z + avoidance_distance),
                target_pos
            ]
        
        elif strategy == AvoidanceStrategy.GO_BELOW:
            # Passer en-dessous (avec limite de sécurité)
            safe_altitude = max(30, obstacle.z - avoidance_distance)
            waypoints = [
                (drone_pos[0], drone_pos[1], safe_altitude),
                (obstacle.x, obstacle.y, safe_altitude),
                target_pos
            ]
        
        elif strategy == AvoidanceStrategy.BACKTRACK:
            # Reculer et recalculer
            back_x = drone_pos[0] - (obstacle.x - drone_pos[0]) * 0.5
            back_y = drone_pos[1] - (obstacle.y - drone_pos[1]) * 0.5
            waypoints = [(back_x, back_y, drone_pos[2])]
        
        return waypoints
    
    def get_safe_direction(self, drone_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Trouve une direction sûre pour s'éloigner des obstacles
        
        Args:
            drone_pos: Position actuelle du drone
        
        Returns:
            Vecteur direction sûre normalisé ou None
        """
        if not self.detected_obstacles:
            return None
        
        # Calculer le vecteur résultant des obstacles
        repulsion_x, repulsion_y, repulsion_z = 0, 0, 0
        
        for obs in self.detected_obstacles:
            dx = drone_pos[0] - obs.x
            dy = drone_pos[1] - obs.y
            dz = drone_pos[2] - obs.z
            
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist > 0:
                # Force de répulsion inversement proportionnelle à la distance
                force = 1.0 / (dist + 1)
                repulsion_x += dx * force / dist
                repulsion_y += dy * force / dist
                repulsion_z += dz * force / dist
        
        # Normaliser
        length = np.sqrt(repulsion_x**2 + repulsion_y**2 + repulsion_z**2)
        if length > 0:
            return (repulsion_x/length, repulsion_y/length, repulsion_z/length)
        
        return None
    
    def simulate_obstacle_movement(self, obstacle_id: int, 
                                    velocity: Tuple[float, float, float]):
        """
        Simule le mouvement d'un obstacle (pour tests)
        
        Args:
            obstacle_id: Index de l'obstacle
            velocity: Vélocité (vx, vy, vz) en cm/s
        """
        if obstacle_id < len(self.detected_obstacles):
            obs = self.detected_obstacles[obstacle_id]
            obs.update_position(
                obs.x + velocity[0] * 0.1,
                obs.y + velocity[1] * 0.1,
                obs.z + velocity[2] * 0.1,
                obs.distance
            )
    
    def get_status_report(self) -> dict:
        """Retourne un rapport d'état du système"""
        return {
            'is_active': self.is_active,
            'total_obstacles': len(self.detected_obstacles),
            'mobile_obstacles': sum(1 for o in self.detected_obstacles if o.is_mobile),
            'avoidance_in_progress': self.avoidance_in_progress,
            'obstacles': [
                {
                    'position': (o.x, o.y, o.z),
                    'distance': o.distance,
                    'is_mobile': o.is_mobile,
                    'velocity': o.get_velocity() if o.is_mobile else (0, 0, 0)
                }
                for o in self.detected_obstacles
            ]
        }


class ReactiveAvoidance:
    """
    Couche d'évitement réactif pour réponse immédiate aux obstacles proches
    Complémente le système principal avec des réflexes rapides
    """
    
    def __init__(self, emergency_distance: float = 50.0):
        """
        Args:
            emergency_distance: Distance déclenchant une réaction d'urgence (cm)
        """
        self.emergency_distance = emergency_distance
        self.last_reaction_time = 0
        self.cooldown = 0.5  # Secondes entre réactions
    
    def check_immediate_danger(self, front_distance: float, 
                                height: float) -> Optional[AvoidanceStrategy]:
        """
        Vérifie les dangers immédiats et retourne une action réflexe
        
        Args:
            front_distance: Distance mesurée devant (capteur ToF)
            height: Hauteur actuelle
        
        Returns:
            Action d'urgence ou None si pas de danger
        """
        current_time = time.time()
        if current_time - self.last_reaction_time < self.cooldown:
            return None
        
        # Obstacle très proche devant
        if front_distance < self.emergency_distance:
            self.last_reaction_time = current_time
            logger.warning(f"DANGER: Obstacle à {front_distance}cm!")
            return AvoidanceStrategy.BACKTRACK
        
        # Trop proche du sol
        if height < 30:
            self.last_reaction_time = current_time
            logger.warning(f"DANGER: Altitude trop basse ({height}cm)")
            return AvoidanceStrategy.GO_ABOVE
        
        return None


# Test du module
if __name__ == "__main__":
    print("=== Test du système d'évitement d'obstacles ===\n")
    
    # Création du système
    avoidance = ObstacleAvoidanceSystem()
    
    # Ajout d'obstacles de test
    obs1 = avoidance.add_obstacle(100, 50, 100, 112, (0.89, 0.45, 0))
    obs2 = avoidance.add_obstacle(0, 100, 100, 100, (0, 1, 0))
    
    # Simulation de mouvement pour obs2 (le rendre mobile)
    for i in range(5):
        time.sleep(0.1)
        avoidance.simulate_obstacle_movement(1, (10, 5, 0))
    
    # Test de détection de collision
    drone_pos = (0, 0, 100)
    target_pos = (150, 75, 100)
    
    collision, obstacle = avoidance.check_collision_risk(
        *drone_pos, *target_pos
    )
    
    print(f"Position drone: {drone_pos}")
    print(f"Position cible: {target_pos}")
    print(f"Risque de collision: {collision}")
    
    if collision and obstacle:
        print(f"Obstacle: ({obstacle.x:.0f}, {obstacle.y:.0f}, {obstacle.z:.0f})")
        print(f"Mobile: {obstacle.is_mobile}")
        
        strategy = avoidance.get_avoidance_strategy(drone_pos, target_pos, obstacle)
        print(f"Stratégie: {strategy.value}")
        
        waypoints = avoidance.calculate_avoidance_waypoints(
            drone_pos, target_pos, obstacle, strategy
        )
        print(f"Waypoints d'évitement: {waypoints}")
    
    print(f"\nRapport d'état: {avoidance.get_status_report()}")
