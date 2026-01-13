#!/usr/bin/env python3
"""
Module de cartographie pour drone Tello EDU
Crée des cartes d'altitude et de l'environnement exploré
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MapPoint:
    """Point de données cartographiques"""
    x: float
    y: float
    z: float  # Altitude du drone
    ground_distance: float  # Distance au sol (capteur ToF)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def ground_altitude(self) -> float:
        """Calcule l'altitude du sol à ce point"""
        return self.z - self.ground_distance
    
    def to_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'ground_distance': self.ground_distance,
            'ground_altitude': self.ground_altitude,
            'timestamp': self.timestamp
        }


@dataclass
class Obstacle:
    """Représente un obstacle détecté"""
    x: float
    y: float
    z: float
    radius: float  # Rayon estimé
    is_mobile: bool = False
    velocity: Tuple[float, float, float] = (0, 0, 0)
    last_seen: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'radius': self.radius,
            'is_mobile': self.is_mobile,
            'velocity': self.velocity,
            'confidence': self.confidence
        }
    
    def predict_position(self, dt: float) -> Tuple[float, float, float]:
        """Prédit la position future de l'obstacle mobile"""
        if not self.is_mobile:
            return (self.x, self.y, self.z)
        
        return (
            self.x + self.velocity[0] * dt,
            self.y + self.velocity[1] * dt,
            self.z + self.velocity[2] * dt
        )


class AltitudeMap:
    """
    Carte d'altitude de la zone explorée
    Utilise une grille pour stocker les données
    """
    
    def __init__(self, resolution: float = 50.0, size: Tuple[float, float] = (1000, 1000)):
        """
        Initialise la carte d'altitude
        
        Args:
            resolution: Résolution de la grille en cm
            size: Taille de la zone (largeur, hauteur) en cm
        """
        self.resolution = resolution
        self.size = size
        
        # Calcul des dimensions de la grille
        self.grid_width = int(size[0] / resolution) + 1
        self.grid_height = int(size[1] / resolution) + 1
        
        # Grille d'altitude (NaN = non exploré)
        self.altitude_grid = np.full((self.grid_height, self.grid_width), np.nan)
        
        # Grille de comptage (pour moyenner les mesures)
        self.count_grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        # Liste des points bruts
        self.raw_points: List[MapPoint] = []
        
        # Obstacles détectés
        self.obstacles: List[Obstacle] = []
        
        # Offset pour centrer la carte (origine au centre)
        self.offset_x = size[0] / 2
        self.offset_y = size[1] / 2
        
        # Métadonnées
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        logger.info(f"Carte initialisée: {self.grid_width}x{self.grid_height} cellules")
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convertit les coordonnées monde en indices de grille"""
        grid_x = int((x + self.offset_x) / self.resolution)
        grid_y = int((y + self.offset_y) / self.resolution)
        
        # Clamp aux limites
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convertit les indices de grille en coordonnées monde"""
        x = grid_x * self.resolution - self.offset_x + self.resolution / 2
        y = grid_y * self.resolution - self.offset_y + self.resolution / 2
        return x, y
    
    def add_point(self, x: float, y: float, z: float, ground_distance: float):
        """
        Ajoute un point de mesure à la carte
        
        Args:
            x, y: Position horizontale du drone
            z: Altitude du drone
            ground_distance: Distance au sol (capteur ToF)
        """
        point = MapPoint(x, y, z, ground_distance)
        self.raw_points.append(point)
        
        # Mise à jour de la grille
        gx, gy = self._world_to_grid(x, y)
        
        ground_altitude = point.ground_altitude
        
        if np.isnan(self.altitude_grid[gy, gx]):
            self.altitude_grid[gy, gx] = ground_altitude
            self.count_grid[gy, gx] = 1
        else:
            # Moyenne pondérée
            n = self.count_grid[gy, gx]
            self.altitude_grid[gy, gx] = (self.altitude_grid[gy, gx] * n + ground_altitude) / (n + 1)
            self.count_grid[gy, gx] += 1
        
        self.last_updated = datetime.now()
    
    def add_obstacle(self, x: float, y: float, z: float, 
                     radius: float = 50.0, is_mobile: bool = False):
        """
        Ajoute un obstacle à la carte
        
        Args:
            x, y, z: Position de l'obstacle
            radius: Rayon estimé en cm
            is_mobile: True si l'obstacle est mobile
        """
        obstacle = Obstacle(x, y, z, radius, is_mobile)
        
        # Vérifier si c'est une mise à jour d'un obstacle existant
        for i, existing in enumerate(self.obstacles):
            dist = np.sqrt((existing.x - x)**2 + (existing.y - y)**2 + (existing.z - z)**2)
            if dist < radius * 2:  # Probablement le même obstacle
                if is_mobile:
                    # Calculer la vélocité
                    dt = time.time() - existing.last_seen
                    if dt > 0:
                        vx = (x - existing.x) / dt
                        vy = (y - existing.y) / dt
                        vz = (z - existing.z) / dt
                        obstacle.velocity = (vx, vy, vz)
                
                self.obstacles[i] = obstacle
                logger.info(f"Obstacle mis à jour: ({x:.1f}, {y:.1f}, {z:.1f})")
                return
        
        self.obstacles.append(obstacle)
        logger.info(f"Nouvel obstacle détecté: ({x:.1f}, {y:.1f}, {z:.1f})")
    
    def get_altitude_at(self, x: float, y: float) -> Optional[float]:
        """
        Récupère l'altitude du sol à une position donnée
        
        Args:
            x, y: Coordonnées monde
        
        Returns:
            Altitude du sol ou None si non exploré
        """
        gx, gy = self._world_to_grid(x, y)
        alt = self.altitude_grid[gy, gx]
        return None if np.isnan(alt) else alt
    
    def get_exploration_coverage(self) -> float:
        """
        Calcule le pourcentage de la zone explorée
        
        Returns:
            Pourcentage de couverture (0-100)
        """
        explored = np.sum(~np.isnan(self.altitude_grid))
        total = self.grid_width * self.grid_height
        return (explored / total) * 100
    
    def get_altitude_stats(self) -> Dict:
        """
        Calcule les statistiques d'altitude
        
        Returns:
            Dictionnaire avec min, max, moyenne, écart-type
        """
        valid_altitudes = self.altitude_grid[~np.isnan(self.altitude_grid)]
        
        if len(valid_altitudes) == 0:
            return {'min': None, 'max': None, 'mean': None, 'std': None}
        
        return {
            'min': float(np.min(valid_altitudes)),
            'max': float(np.max(valid_altitudes)),
            'mean': float(np.mean(valid_altitudes)),
            'std': float(np.std(valid_altitudes))
        }
    
    def get_obstacles_in_area(self, x: float, y: float, z: float, 
                              radius: float = 100.0) -> List[Obstacle]:
        """
        Trouve les obstacles dans une zone donnée
        
        Args:
            x, y, z: Centre de la zone
            radius: Rayon de recherche
        
        Returns:
            Liste des obstacles dans la zone
        """
        nearby = []
        for obs in self.obstacles:
            dist = np.sqrt((obs.x - x)**2 + (obs.y - y)**2 + (obs.z - z)**2)
            if dist < radius + obs.radius:
                nearby.append(obs)
        return nearby
    
    def get_mobile_obstacles(self) -> List[Obstacle]:
        """Retourne la liste des obstacles mobiles"""
        return [obs for obs in self.obstacles if obs.is_mobile]
    
    def to_ascii_map(self, drone_pos: Tuple[float, float] = None) -> str:
        """
        Génère une représentation ASCII de la carte
        
        Args:
            drone_pos: Position actuelle du drone (optionnel)
        
        Returns:
            Chaîne représentant la carte en ASCII
        """
        # Sous-échantillonnage pour affichage
        display_width = min(60, self.grid_width)
        display_height = min(30, self.grid_height)
        
        scale_x = self.grid_width / display_width
        scale_y = self.grid_height / display_height
        
        lines = []
        lines.append("=" * (display_width + 2))
        lines.append("CARTE D'ALTITUDE")
        lines.append("=" * (display_width + 2))
        
        # Symboles: · = non exploré, 0-9 = altitude relative, X = obstacle, D = drone
        stats = self.get_altitude_stats()
        alt_range = (stats['max'] - stats['min']) if stats['min'] is not None and stats['max'] is not None else 1
        
        for dy in range(display_height):
            row = ""
            for dx in range(display_width):
                gx = int(dx * scale_x)
                gy = int(dy * scale_y)
                
                # Vérifier si c'est la position du drone
                if drone_pos:
                    dgx, dgy = self._world_to_grid(drone_pos[0], drone_pos[1])
                    if abs(gx - dgx) < scale_x and abs(gy - dgy) < scale_y:
                        row += "D"
                        continue
                
                # Vérifier obstacles
                wx, wy = self._grid_to_world(gx, gy)
                is_obstacle = False
                for obs in self.obstacles:
                    if abs(obs.x - wx) < self.resolution and abs(obs.y - wy) < self.resolution:
                        row += "X" if not obs.is_mobile else "M"
                        is_obstacle = True
                        break
                
                if is_obstacle:
                    continue
                
                # Altitude
                alt = self.altitude_grid[gy, gx]
                if np.isnan(alt):
                    row += "·"
                else:
                    if alt_range > 0:
                        level = int((alt - stats['min']) / alt_range * 9)
                        level = max(0, min(9, level))
                    else:
                        level = 5
                    row += str(level)
            
            lines.append("|" + row + "|")
        
        lines.append("=" * (display_width + 2))
        lines.append(f"Couverture: {self.get_exploration_coverage():.1f}%")
        lines.append(f"Obstacles: {len(self.obstacles)} (mobiles: {len(self.get_mobile_obstacles())})")
        if stats['min'] is not None:
            lines.append(f"Altitude: {stats['min']:.0f} - {stats['max']:.0f} cm")
        lines.append("Légende: · non exploré | 0-9 altitude | X obstacle | M mobile | D drone")
        
        return "\n".join(lines)
    
    def export_to_json(self, filepath: str):
        """
        Exporte la carte au format JSON
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        data = {
            'metadata': {
                'created_at': self.created_at.isoformat(),
                'last_updated': self.last_updated.isoformat(),
                'resolution': self.resolution,
                'size': self.size,
                'coverage': self.get_exploration_coverage(),
                'stats': self.get_altitude_stats()
            },
            'raw_points': [p.to_dict() for p in self.raw_points],
            'obstacles': [o.to_dict() for o in self.obstacles],
            'altitude_grid': self.altitude_grid.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Carte exportée vers {filepath}")
    
    def export_to_csv(self, filepath: str):
        """
        Exporte les points bruts au format CSV
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        with open(filepath, 'w') as f:
            f.write("x,y,z,ground_distance,ground_altitude,timestamp\n")
            for p in self.raw_points:
                f.write(f"{p.x},{p.y},{p.z},{p.ground_distance},{p.ground_altitude},{p.timestamp}\n")
        
        logger.info(f"Points exportés vers {filepath}")


class ExplorationPlanner:
    """
    Planificateur d'exploration pour couvrir une zone efficacement
    """
    
    def __init__(self, altitude_map: AltitudeMap, step_size: float = 50.0):
        """
        Initialise le planificateur
        
        Args:
            altitude_map: Carte à remplir
            step_size: Distance entre les points d'exploration (cm)
        """
        self.map = altitude_map
        self.step_size = step_size
        self.exploration_path: List[Tuple[float, float]] = []
        self.current_index = 0
    
    def generate_snake_pattern(self, width: float, height: float, 
                                start_x: float = 0, start_y: float = 0) -> List[Tuple[float, float]]:
        """
        Génère un pattern de serpent pour couvrir une zone rectangulaire
        
        Args:
            width, height: Dimensions de la zone à explorer
            start_x, start_y: Point de départ
        
        Returns:
            Liste de waypoints (x, y)
        """
        waypoints = []
        
        # Calcul du nombre de passes
        num_passes_x = int(width / self.step_size)
        num_passes_y = int(height / self.step_size)
        
        direction = 1  # 1 = vers la droite, -1 = vers la gauche
        
        for i in range(num_passes_y + 1):
            y = start_y - height/2 + i * self.step_size
            
            if direction == 1:
                x_start = start_x - width/2
                x_end = start_x + width/2
            else:
                x_start = start_x + width/2
                x_end = start_x - width/2
            
            for j in range(num_passes_x + 1):
                x = x_start + direction * j * self.step_size
                waypoints.append((x, y))
            
            direction *= -1
        
        self.exploration_path = waypoints
        self.current_index = 0
        
        logger.info(f"Pattern généré: {len(waypoints)} waypoints")
        return waypoints
    
    def generate_spiral_pattern(self, max_radius: float, 
                                 center_x: float = 0, center_y: float = 0) -> List[Tuple[float, float]]:
        """
        Génère un pattern en spirale depuis le centre
        
        Args:
            max_radius: Rayon maximum de la spirale
            center_x, center_y: Centre de la spirale
        
        Returns:
            Liste de waypoints (x, y)
        """
        waypoints = [(center_x, center_y)]
        
        angle = 0
        radius = 0
        angle_step = np.pi / 4  # 45 degrés
        
        while radius < max_radius:
            radius += self.step_size / (2 * np.pi)  # Incrément progressif
            angle += angle_step
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            waypoints.append((x, y))
        
        self.exploration_path = waypoints
        self.current_index = 0
        
        logger.info(f"Spirale générée: {len(waypoints)} waypoints")
        return waypoints
    
    def get_next_waypoint(self) -> Optional[Tuple[float, float]]:
        """
        Récupère le prochain waypoint à atteindre
        
        Returns:
            Coordonnées (x, y) ou None si terminé
        """
        if self.current_index >= len(self.exploration_path):
            return None
        
        waypoint = self.exploration_path[self.current_index]
        self.current_index += 1
        return waypoint
    
    def get_progress(self) -> float:
        """Retourne le pourcentage de progression"""
        if len(self.exploration_path) == 0:
            return 0.0
        return (self.current_index / len(self.exploration_path)) * 100
    
    def reset(self):
        """Remet le planificateur au début"""
        self.current_index = 0


# Test du module
if __name__ == "__main__":
    print("=== Test du module de cartographie ===")
    
    # Création d'une carte
    altitude_map = AltitudeMap(resolution=50, size=(500, 500))
    
    # Ajout de points simulés
    for x in range(-200, 250, 50):
        for y in range(-200, 250, 50):
            # Simulation d'un terrain vallonné
            ground_dist = 100 + 20 * np.sin(x/100) * np.cos(y/100)
            altitude_map.add_point(x, y, 150, ground_dist)
    
    # Ajout d'obstacles
    altitude_map.add_obstacle(100, 100, 100, radius=30, is_mobile=False)
    altitude_map.add_obstacle(-50, 50, 80, radius=25, is_mobile=True)
    
    # Affichage
    print(altitude_map.to_ascii_map(drone_pos=(0, 0)))
    print(f"\nStatistiques: {altitude_map.get_altitude_stats()}")
    
    # Test du planificateur
    planner = ExplorationPlanner(altitude_map, step_size=50)
    waypoints = planner.generate_snake_pattern(400, 400)
    print(f"\nWaypoints générés: {len(waypoints)}")
