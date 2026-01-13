#!/usr/bin/env python3
"""
Module de cartographie optimisé pour Tello EDU
Cartographie spatiale et thermique simultanée
"""

import numpy as np
import json
import time
import math
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
    z: float
    ground_distance: float
    temperature: float = 25.0  # °C
    timestamp: float = field(default_factory=time.time)
    
    @property
    def ground_altitude(self) -> float:
        return self.z - self.ground_distance
    
    def to_dict(self) -> dict:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'ground_distance': self.ground_distance,
            'ground_altitude': self.ground_altitude,
            'temperature': self.temperature,
            'timestamp': self.timestamp
        }


@dataclass
class Obstacle:
    """Obstacle détecté"""
    x: float
    y: float
    z: float
    radius: float
    is_mobile: bool = False
    velocity: Tuple[float, float, float] = (0, 0, 0)
    obstacle_type: str = "unknown"
    threat_level: int = 0  # 0-4
    last_seen: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'radius': self.radius,
            'is_mobile': self.is_mobile,
            'velocity': self.velocity,
            'type': self.obstacle_type,
            'threat_level': self.threat_level,
            'confidence': self.confidence
        }
    
    def predict_position(self, dt: float) -> Tuple[float, float, float]:
        if not self.is_mobile:
            return (self.x, self.y, self.z)
        return (
            self.x + self.velocity[0] * dt,
            self.y + self.velocity[1] * dt,
            self.z + self.velocity[2] * dt
        )


@dataclass
class ThermalZone:
    """Zone thermique détectée"""
    x: float
    y: float
    z: float
    radius: float
    temperature: float
    is_active: bool = True  # Feu actif?
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'radius': self.radius,
            'temperature': self.temperature,
            'is_active': self.is_active,
            'timestamp': self.timestamp
        }


class DualMap:
    """
    Carte double: spatiale (altitude) + thermique
    """
    
    def __init__(self, resolution: float = 50.0, size: Tuple[float, float] = (1000, 1000)):
        self.resolution = resolution
        self.size = size
        
        self.grid_width = int(size[0] / resolution) + 1
        self.grid_height = int(size[1] / resolution) + 1
        
        # Grille d'altitude (NaN = non exploré)
        self.altitude_grid = np.full((self.grid_height, self.grid_width), np.nan)
        
        # Grille thermique (température)
        self.thermal_grid = np.full((self.grid_height, self.grid_width), 25.0)  # 25°C par défaut
        
        # Grille d'occupation (0=libre, 1=occupé, -1=inconnu)
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)
        
        # Comptage
        self.count_grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        # Données brutes
        self.raw_points: List[MapPoint] = []
        self.obstacles: List[Obstacle] = []
        self.thermal_zones: List[ThermalZone] = []
        
        # Offset (origine au centre)
        self.offset_x = size[0] / 2
        self.offset_y = size[1] / 2
        
        # Métadonnées
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        logger.info(f"DualMap initialisée: {self.grid_width}x{self.grid_height}")
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Coordonnées monde -> grille"""
        gx = int((x + self.offset_x) / self.resolution)
        gy = int((y + self.offset_y) / self.resolution)
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        return gx, gy
    
    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Grille -> coordonnées monde"""
        x = gx * self.resolution - self.offset_x + self.resolution / 2
        y = gy * self.resolution - self.offset_y + self.resolution / 2
        return x, y
    
    def add_point(self, x: float, y: float, z: float, 
                  ground_distance: float, temperature: float = 25.0):
        """Ajoute un point avec données thermiques"""
        point = MapPoint(x, y, z, ground_distance, temperature)
        self.raw_points.append(point)
        
        gx, gy = self._world_to_grid(x, y)
        
        # Altitude
        ground_alt = point.ground_altitude
        if np.isnan(self.altitude_grid[gy, gx]):
            self.altitude_grid[gy, gx] = ground_alt
            self.count_grid[gy, gx] = 1
        else:
            n = self.count_grid[gy, gx]
            self.altitude_grid[gy, gx] = (self.altitude_grid[gy, gx] * n + ground_alt) / (n + 1)
            self.count_grid[gy, gx] += 1
        
        # Thermique (moyenne pondérée)
        alpha = 0.3
        self.thermal_grid[gy, gx] = alpha * temperature + (1 - alpha) * self.thermal_grid[gy, gx]
        
        # Occupation (libre car le drone y est passé)
        self.occupancy_grid[gy, gx] = 0
        
        self.last_updated = datetime.now()
    
    def add_obstacle(self, x: float, y: float, z: float,
                     radius: float = 50.0, is_mobile: bool = False,
                     obstacle_type: str = "unknown", threat_level: int = 0):
        """Ajoute un obstacle"""
        # Vérifier si mise à jour
        for i, existing in enumerate(self.obstacles):
            dist = math.sqrt((existing.x - x)**2 + (existing.y - y)**2)
            if dist < radius * 2:
                if is_mobile:
                    dt = time.time() - existing.last_seen
                    if dt > 0:
                        existing.velocity = (
                            (x - existing.x) / dt,
                            (y - existing.y) / dt,
                            (z - existing.z) / dt
                        )
                
                existing.x, existing.y, existing.z = x, y, z
                existing.is_mobile = is_mobile
                existing.last_seen = time.time()
                existing.obstacle_type = obstacle_type
                existing.threat_level = threat_level
                
                # MAJ grille occupation
                gx, gy = self._world_to_grid(x, y)
                self.occupancy_grid[gy, gx] = 100
                
                return existing
        
        # Nouvel obstacle
        obs = Obstacle(x, y, z, radius, is_mobile, obstacle_type=obstacle_type, threat_level=threat_level)
        self.obstacles.append(obs)
        
        gx, gy = self._world_to_grid(x, y)
        self.occupancy_grid[gy, gx] = 100
        
        logger.info(f"Obstacle ajouté: ({x:.0f}, {y:.0f}) - {obstacle_type}")
        return obs
    
    def add_thermal_zone(self, x: float, y: float, z: float,
                         radius: float, temperature: float, is_active: bool = True):
        """Ajoute une zone thermique"""
        zone = ThermalZone(x, y, z, radius, temperature, is_active)
        self.thermal_zones.append(zone)
        
        # MAJ grille thermique (zone circulaire)
        gx, gy = self._world_to_grid(x, y)
        grid_radius = int(radius / self.resolution) + 1
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if dx**2 + dy**2 <= grid_radius**2:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.thermal_grid[ny, nx] = max(self.thermal_grid[ny, nx], temperature)
        
        logger.info(f"Zone thermique: ({x:.0f}, {y:.0f}) - {temperature:.0f}°C")
        return zone
    
    def get_altitude_at(self, x: float, y: float) -> Optional[float]:
        """Altitude à une position"""
        gx, gy = self._world_to_grid(x, y)
        alt = self.altitude_grid[gy, gx]
        return None if np.isnan(alt) else alt
    
    def get_temperature_at(self, x: float, y: float) -> float:
        """Température à une position"""
        gx, gy = self._world_to_grid(x, y)
        return self.thermal_grid[gy, gx]
    
    def is_occupied(self, x: float, y: float) -> bool:
        """Vérifie si une position est occupée"""
        gx, gy = self._world_to_grid(x, y)
        return self.occupancy_grid[gy, gx] > 50
    
    def get_exploration_coverage(self) -> float:
        """Pourcentage de zone explorée"""
        explored = np.sum(~np.isnan(self.altitude_grid))
        total = self.grid_width * self.grid_height
        return (explored / total) * 100
    
    def get_hot_zones_count(self, threshold: float = 50.0) -> int:
        """Nombre de zones chaudes"""
        return int(np.sum(self.thermal_grid > threshold))
    
    def get_statistics(self) -> Dict:
        """Statistiques complètes"""
        valid_alt = self.altitude_grid[~np.isnan(self.altitude_grid)]
        
        return {
            'coverage': self.get_exploration_coverage(),
            'points_recorded': len(self.raw_points),
            'obstacles_count': len(self.obstacles),
            'mobile_obstacles': sum(1 for o in self.obstacles if o.is_mobile),
            'thermal_zones': len(self.thermal_zones),
            'altitude': {
                'min': float(np.min(valid_alt)) if len(valid_alt) > 0 else None,
                'max': float(np.max(valid_alt)) if len(valid_alt) > 0 else None,
                'mean': float(np.mean(valid_alt)) if len(valid_alt) > 0 else None,
            },
            'temperature': {
                'min': float(np.min(self.thermal_grid)),
                'max': float(np.max(self.thermal_grid)),
                'mean': float(np.mean(self.thermal_grid)),
                'hot_cells': self.get_hot_zones_count()
            }
        }
    
    def get_obstacles_near(self, x: float, y: float, radius: float = 100.0) -> List[Obstacle]:
        """Obstacles dans un rayon donné"""
        nearby = []
        for obs in self.obstacles:
            dist = math.sqrt((obs.x - x)**2 + (obs.y - y)**2)
            if dist < radius + obs.radius:
                nearby.append(obs)
        return nearby
    
    def get_mobile_obstacles(self) -> List[Obstacle]:
        """Liste des obstacles mobiles"""
        return [o for o in self.obstacles if o.is_mobile]
    
    def get_danger_zones(self) -> List[Tuple[float, float, str]]:
        """Liste des zones dangereuses"""
        dangers = []
        
        # Obstacles critiques
        for obs in self.obstacles:
            if obs.threat_level >= 3:
                dangers.append((obs.x, obs.y, f"obstacle_{obs.obstacle_type}"))
        
        # Zones thermiques
        for zone in self.thermal_zones:
            if zone.temperature > 100:
                dangers.append((zone.x, zone.y, "fire"))
            elif zone.temperature > 60:
                dangers.append((zone.x, zone.y, "hot_zone"))
        
        return dangers
    
    def to_ascii_map(self, drone_pos: Tuple[float, float] = None, 
                     show_thermal: bool = False) -> str:
        """Génère une représentation ASCII"""
        display_w = min(60, self.grid_width)
        display_h = min(30, self.grid_height)
        
        scale_x = self.grid_width / display_w
        scale_y = self.grid_height / display_h
        
        lines = []
        lines.append("=" * (display_w + 2))
        title = "CARTE THERMIQUE" if show_thermal else "CARTE D'ALTITUDE"
        lines.append(title.center(display_w + 2))
        lines.append("=" * (display_w + 2))
        
        # Grille à afficher
        grid = self.thermal_grid if show_thermal else self.altitude_grid
        
        # Calcul des limites
        if show_thermal:
            vmin, vmax = 20, 150
        else:
            valid = grid[~np.isnan(grid)] if not show_thermal else grid.flatten()
            if len(valid) > 0:
                vmin, vmax = np.min(valid), np.max(valid)
            else:
                vmin, vmax = 0, 1
        
        value_range = max(vmax - vmin, 1)
        
        for dy in range(display_h):
            row = ""
            for dx in range(display_w):
                gx = int(dx * scale_x)
                gy = int(dy * scale_y)
                
                # Position drone
                if drone_pos:
                    dgx, dgy = self._world_to_grid(drone_pos[0], drone_pos[1])
                    if abs(gx - dgx) < scale_x and abs(gy - dgy) < scale_y:
                        row += "D"
                        continue
                
                # Obstacles
                wx, wy = self._grid_to_world(gx, gy)
                is_obs = False
                for obs in self.obstacles:
                    if abs(obs.x - wx) < self.resolution and abs(obs.y - wy) < self.resolution:
                        if obs.is_mobile:
                            row += "M"
                        elif obs.obstacle_type == "fire":
                            row += "F"
                        else:
                            row += "X"
                        is_obs = True
                        break
                
                if is_obs:
                    continue
                
                # Valeur
                val = grid[gy, gx]
                if np.isnan(val) and not show_thermal:
                    row += "·"
                else:
                    level = int((val - vmin) / value_range * 9)
                    level = max(0, min(9, level))
                    row += str(level)
            
            lines.append("|" + row + "|")
        
        lines.append("=" * (display_w + 2))
        stats = self.get_statistics()
        lines.append(f"Couverture: {stats['coverage']:.1f}%")
        lines.append(f"Obstacles: {stats['obstacles_count']} | Thermique: {stats['thermal_zones']}")
        
        if show_thermal:
            lines.append(f"Temp: {stats['temperature']['min']:.0f}-{stats['temperature']['max']:.0f}°C")
        else:
            if stats['altitude']['min'] is not None:
                lines.append(f"Altitude: {stats['altitude']['min']:.0f}-{stats['altitude']['max']:.0f}cm")
        
        lines.append("Légende: D=drone | X=obstacle | M=mobile | F=feu | ·=inexploré")
        
        return "\n".join(lines)
    
    def export_to_json(self, filepath: str):
        """Export JSON complet"""
        data = {
            'metadata': {
                'created_at': self.created_at.isoformat(),
                'last_updated': self.last_updated.isoformat(),
                'resolution': self.resolution,
                'size': self.size,
                'statistics': self.get_statistics()
            },
            'raw_points': [p.to_dict() for p in self.raw_points[-1000:]],  # Limite
            'obstacles': [o.to_dict() for o in self.obstacles],
            'thermal_zones': [z.to_dict() for z in self.thermal_zones],
            'danger_zones': self.get_danger_zones()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Carte exportée: {filepath}")
    
    def export_grids(self, base_path: str):
        """Export des grilles en fichiers numpy"""
        np.save(f"{base_path}_altitude.npy", self.altitude_grid)
        np.save(f"{base_path}_thermal.npy", self.thermal_grid)
        np.save(f"{base_path}_occupancy.npy", self.occupancy_grid)
        logger.info(f"Grilles exportées: {base_path}_*.npy")


class ExplorationPlanner:
    """Planificateur d'exploration optimisé"""
    
    def __init__(self, dual_map: DualMap, step_size: float = 50.0):
        self.map = dual_map
        self.step_size = step_size
        self.exploration_path: List[Tuple[float, float]] = []
        self.current_index = 0
        
    def generate_snake_pattern(self, width: float, height: float,
                               start_x: float = 0, start_y: float = 0) -> List[Tuple[float, float]]:
        """Génère un pattern serpent"""
        waypoints = []
        
        num_x = int(width / self.step_size)
        num_y = int(height / self.step_size)
        
        direction = 1
        
        for i in range(num_y + 1):
            y = start_y - height/2 + i * self.step_size
            
            if direction == 1:
                x_range = range(num_x + 1)
            else:
                x_range = range(num_x, -1, -1)
            
            for j in x_range:
                x = start_x - width/2 + j * self.step_size
                waypoints.append((x, y))
            
            direction *= -1
        
        self.exploration_path = waypoints
        self.current_index = 0
        
        logger.info(f"Pattern snake: {len(waypoints)} waypoints")
        return waypoints
    
    def generate_spiral_pattern(self, max_radius: float,
                                center_x: float = 0, center_y: float = 0) -> List[Tuple[float, float]]:
        """Génère un pattern spirale"""
        waypoints = [(center_x, center_y)]
        
        angle = 0
        radius = 0
        angle_step = math.pi / 4
        
        while radius < max_radius:
            radius += self.step_size / (2 * math.pi)
            angle += angle_step
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            waypoints.append((x, y))
        
        self.exploration_path = waypoints
        self.current_index = 0
        
        logger.info(f"Pattern spirale: {len(waypoints)} waypoints")
        return waypoints
    
    def generate_room_search_pattern(self, width: float, height: float) -> List[Tuple[float, float]]:
        """Pattern optimisé pour recherche dans une pièce"""
        waypoints = []
        
        # Périmètre d'abord
        for x in np.arange(-width/2, width/2, self.step_size):
            waypoints.append((x, -height/2))
        for y in np.arange(-height/2, height/2, self.step_size):
            waypoints.append((width/2, y))
        for x in np.arange(width/2, -width/2, -self.step_size):
            waypoints.append((x, height/2))
        for y in np.arange(height/2, -height/2, -self.step_size):
            waypoints.append((-width/2, y))
        
        # Puis intérieur en spirale
        inner_waypoints = self.generate_spiral_pattern(
            max(width, height) / 2 - self.step_size
        )
        waypoints.extend(inner_waypoints)
        
        self.exploration_path = waypoints
        self.current_index = 0
        
        logger.info(f"Pattern room search: {len(waypoints)} waypoints")
        return waypoints
    
    def get_next_waypoint(self) -> Optional[Tuple[float, float]]:
        """Prochain waypoint"""
        if self.current_index >= len(self.exploration_path):
            return None
        
        wp = self.exploration_path[self.current_index]
        self.current_index += 1
        return wp
    
    def get_progress(self) -> float:
        """Progression en %"""
        if len(self.exploration_path) == 0:
            return 0.0
        return (self.current_index / len(self.exploration_path)) * 100
    
    def skip_to_waypoint(self, index: int):
        """Saute à un waypoint spécifique"""
        self.current_index = max(0, min(index, len(self.exploration_path)))
    
    def reset(self):
        """Remet au début"""
        self.current_index = 0
    
    def replan_avoiding(self, obstacle_x: float, obstacle_y: float, radius: float = 100):
        """Replanifie en évitant une zone"""
        new_path = []
        
        for wp in self.exploration_path[self.current_index:]:
            dist = math.sqrt((wp[0] - obstacle_x)**2 + (wp[1] - obstacle_y)**2)
            if dist > radius:
                new_path.append(wp)
        
        self.exploration_path = self.exploration_path[:self.current_index] + new_path
        logger.info(f"Chemin replanifié, {len(new_path)} waypoints restants")


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Test DualMap ===\n")
    
    dual_map = DualMap(resolution=50, size=(500, 500))
    
    # Points d'exploration
    for x in range(-200, 250, 50):
        for y in range(-200, 250, 50):
            ground_dist = 100 + 20 * math.sin(x/100) * math.cos(y/100)
            temp = 25 + 30 * math.exp(-((x-100)**2 + (y-100)**2) / 10000)
            dual_map.add_point(x, y, 120, ground_dist, temp)
    
    # Obstacles
    dual_map.add_obstacle(100, 100, 100, 30, False, "debris", 2)
    dual_map.add_obstacle(-50, 50, 80, 25, True, "person", 1)
    
    # Zone thermique
    dual_map.add_thermal_zone(100, 100, 50, 80, 120, True)
    
    # Affichage
    print("CARTE ALTITUDE:")
    print(dual_map.to_ascii_map((0, 0), show_thermal=False))
    
    print("\nCARTE THERMIQUE:")
    print(dual_map.to_ascii_map((0, 0), show_thermal=True))
    
    print("\nSTATISTIQUES:")
    stats = dual_map.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Test planificateur
    planner = ExplorationPlanner(dual_map)
    planner.generate_snake_pattern(400, 400)
    print(f"\nWaypoints: {len(planner.exploration_path)}")
    
    print("\n✓ Test terminé")
