#!/usr/bin/env python3
"""
Contrôleur principal pour drone Tello EDU
Exploration de zones dangereuses (thermique/nucléaire/chimique)
"""

import time

# Import conditionnel de djitellopy (requis uniquement pour le mode réel)
try:
    from djitellopy import Tello
    DJITELLOPY_AVAILABLE = True
except ImportError:
    Tello = None
    DJITELLOPY_AVAILABLE = False
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DroneState(Enum):
    """États possibles du drone"""
    IDLE = "idle"
    FLYING = "flying"
    EXPLORING = "exploring"
    AVOIDING = "avoiding"
    EMERGENCY = "emergency"
    LANDED = "landed"


@dataclass
class Position:
    """Position 3D du drone"""
    x: float = 0.0  # cm
    y: float = 0.0  # cm
    z: float = 0.0  # cm (altitude)
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class TelloController:
    """
    Contrôleur principal du drone Tello EDU
    Gère les commandes de base et le suivi de position
    """
    
    # Constantes de mouvement
    DEFAULT_MOVE_DISTANCE = 50  # cm
    MIN_MOVE_DISTANCE = 20  # cm (limite Tello)
    MAX_MOVE_DISTANCE = 500  # cm
    DEFAULT_SPEED = 30  # cm/s
    
    def __init__(self, simulation_mode: bool = False):
        """
        Initialise le contrôleur
        
        Args:
            simulation_mode: Si True, simule les commandes sans drone réel
        """
        self.simulation_mode = simulation_mode
        self.drone: Optional[Tello] = None
        self.state = DroneState.IDLE
        self.position = Position()
        self.start_position = Position()
        self.battery_level = 100
        self.is_connected = False
        
        logger.info(f"Contrôleur initialisé (mode simulation: {simulation_mode})")
    
    def connect(self) -> bool:
        """
        Établit la connexion avec le drone
        
        Returns:
            True si la connexion est établie
        """
        try:
            if self.simulation_mode:
                logger.info("Mode simulation - connexion simulée")
                self.is_connected = True
                return True
            
            if not DJITELLOPY_AVAILABLE:
                logger.error("djitellopy non installé. Utilisez 'pip install djitellopy' ou le mode simulation.")
                return False
            
            self.drone = Tello()
            self.drone.connect()
            self.battery_level = self.drone.get_battery()
            self.is_connected = True
            
            logger.info(f"Connecté au drone - Batterie: {self.battery_level}%")
            return True
            
        except Exception as e:
            logger.error(f"Erreur de connexion: {e}")
            return False
    
    def disconnect(self):
        """Déconnecte proprement du drone"""
        if self.state == DroneState.FLYING:
            self.land()
        
        if self.drone and not self.simulation_mode:
            self.drone.end()
        
        self.is_connected = False
        logger.info("Drone déconnecté")
    
    def takeoff(self) -> bool:
        """
        Fait décoller le drone
        
        Returns:
            True si le décollage est réussi
        """
        if not self.is_connected:
            logger.error("Drone non connecté")
            return False
        
        if self.state == DroneState.FLYING:
            logger.warning("Le drone est déjà en vol")
            return True
        
        try:
            logger.info("Décollage en cours...")
            
            if not self.simulation_mode:
                self.drone.takeoff()
                time.sleep(2)  # Stabilisation
            
            self.state = DroneState.FLYING
            self.position.z = 80  # Altitude de décollage approximative
            self.start_position = Position(0, 0, self.position.z)
            
            logger.info(f"Décollage réussi - Altitude: {self.position.z}cm")
            return True
            
        except Exception as e:
            logger.error(f"Erreur de décollage: {e}")
            self.state = DroneState.EMERGENCY
            return False
    
    def land(self) -> bool:
        """
        Fait atterrir le drone
        
        Returns:
            True si l'atterrissage est réussi
        """
        if self.state not in [DroneState.FLYING, DroneState.EXPLORING, DroneState.AVOIDING]:
            logger.warning("Le drone n'est pas en vol")
            return True
        
        try:
            logger.info("Atterrissage en cours...")
            
            if not self.simulation_mode:
                self.drone.land()
                time.sleep(2)
            
            self.state = DroneState.LANDED
            self.position.z = 0
            
            logger.info("Atterrissage réussi")
            return True
            
        except Exception as e:
            logger.error(f"Erreur d'atterrissage: {e}")
            self.emergency_stop()
            return False
    
    def emergency_stop(self):
        """Arrêt d'urgence - coupe les moteurs immédiatement"""
        logger.warning("ARRÊT D'URGENCE ACTIVÉ")
        
        if not self.simulation_mode and self.drone:
            try:
                self.drone.emergency()
            except:
                pass
        
        self.state = DroneState.EMERGENCY
    
    def _validate_move(self, distance: int) -> int:
        """Valide et ajuste la distance de mouvement"""
        distance = abs(distance)
        if distance < self.MIN_MOVE_DISTANCE:
            logger.warning(f"Distance trop faible, ajustée à {self.MIN_MOVE_DISTANCE}cm")
            return self.MIN_MOVE_DISTANCE
        if distance > self.MAX_MOVE_DISTANCE:
            logger.warning(f"Distance trop grande, ajustée à {self.MAX_MOVE_DISTANCE}cm")
            return self.MAX_MOVE_DISTANCE
        return distance
    
    def move_left(self, distance: int = DEFAULT_MOVE_DISTANCE) -> bool:
        """
        Déplace le drone vers la gauche
        
        Args:
            distance: Distance en cm (défaut: 50cm)
        
        Returns:
            True si le mouvement est réussi
        """
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            logger.error("Le drone doit être en vol")
            return False
        
        distance = self._validate_move(distance)
        
        try:
            logger.info(f"Déplacement gauche: {distance}cm")
            
            if not self.simulation_mode:
                self.drone.move_left(distance)
            
            self.position.y -= distance
            logger.info(f"Position: {self.position.to_tuple()}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur mouvement gauche: {e}")
            return False
    
    # Ajoutez ceci dans la classe TelloController
    
    def get_heading(self) -> float:
        """
        Retourne l'orientation actuelle du drone en degrés (0 = Nord/Départ, + = Horaire)
        Ceci est une estimation basée sur l'accumulation des commandes de rotation.
        """
        # Note: Pour une vraie précision, il faudrait le capteur IMU du drone (yaw)
        # Ici on simule ou on récupère l'état estimé si disponible
        if self.drone and not self.simulation_mode:
            return self.drone.get_yaw()
        return 0.0 # En simulation, il faudrait tracker l'angle accumulé

    def rotate_to_angle(self, target_angle: float):
        """
        Pivote le drone vers un angle cible absolu
        """
        current_yaw = self.get_heading()
        diff = target_angle - current_yaw
        
        # Normalisation entre -180 et 180
        diff = (diff + 180) % 360 - 180
        
        if abs(diff) > 5: # Zone morte de 5 degrés
            if diff > 0:
                self.rotate_clockwise(int(diff))
            else:
                self.rotate_counter_clockwise(int(abs(diff)))
                
    def move_right(self, distance: int = DEFAULT_MOVE_DISTANCE) -> bool:
        """
        Déplace le drone vers la droite
        
        Args:
            distance: Distance en cm (défaut: 50cm)
        
        Returns:
            True si le mouvement est réussi
        """
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            logger.error("Le drone doit être en vol")
            return False
        
        distance = self._validate_move(distance)
        
        try:
            logger.info(f"Déplacement droite: {distance}cm")
            
            if not self.simulation_mode:
                self.drone.move_right(distance)
            
            self.position.y += distance
            logger.info(f"Position: {self.position.to_tuple()}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur mouvement droite: {e}")
            return False
    
    def move_forward(self, distance: int = DEFAULT_MOVE_DISTANCE) -> bool:
        """
        Déplace le drone vers l'avant
        
        Args:
            distance: Distance en cm (défaut: 50cm)
        
        Returns:
            True si le mouvement est réussi
        """
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            logger.error("Le drone doit être en vol")
            return False
        
        distance = self._validate_move(distance)
        
        try:
            logger.info(f"Déplacement avant: {distance}cm")
            
            if not self.simulation_mode:
                self.drone.move_forward(distance)
            
            self.position.x += distance
            logger.info(f"Position: {self.position.to_tuple()}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur mouvement avant: {e}")
            return False
    
    def move_back(self, distance: int = DEFAULT_MOVE_DISTANCE) -> bool:
        """
        Déplace le drone vers l'arrière
        
        Args:
            distance: Distance en cm (défaut: 50cm)
        
        Returns:
            True si le mouvement est réussi
        """
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            logger.error("Le drone doit être en vol")
            return False
        
        distance = self._validate_move(distance)
        
        try:
            logger.info(f"Déplacement arrière: {distance}cm")
            
            if not self.simulation_mode:
                self.drone.move_back(distance)
            
            self.position.x -= distance
            logger.info(f"Position: {self.position.to_tuple()}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur mouvement arrière: {e}")
            return False
    
    def move_up(self, distance: int = DEFAULT_MOVE_DISTANCE) -> bool:
        """
        Monte le drone
        
        Args:
            distance: Distance en cm (défaut: 50cm)
        
        Returns:
            True si le mouvement est réussi
        """
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            logger.error("Le drone doit être en vol")
            return False
        
        distance = self._validate_move(distance)
        
        try:
            logger.info(f"Montée: {distance}cm")
            
            if not self.simulation_mode:
                self.drone.move_up(distance)
            
            self.position.z += distance
            logger.info(f"Position: {self.position.to_tuple()}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur montée: {e}")
            return False
    
    def move_down(self, distance: int = DEFAULT_MOVE_DISTANCE) -> bool:
        """
        Descend le drone
        
        Args:
            distance: Distance en cm (défaut: 50cm)
        
        Returns:
            True si le mouvement est réussi
        """
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            logger.error("Le drone doit être en vol")
            return False
        
        distance = self._validate_move(distance)
        
        # Vérification de sécurité pour ne pas toucher le sol
        if self.position.z - distance < 30:
            logger.warning("Altitude trop basse - mouvement ajusté")
            distance = max(0, int(self.position.z - 30))
            if distance < self.MIN_MOVE_DISTANCE:
                logger.error("Impossible de descendre plus bas")
                return False
        
        try:
            logger.info(f"Descente: {distance}cm")
            
            if not self.simulation_mode:
                self.drone.move_down(distance)
            
            self.position.z -= distance
            logger.info(f"Position: {self.position.to_tuple()}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur descente: {e}")
            return False
    
    def rotate_clockwise(self, angle: int = 90) -> bool:
        """Rotation horaire"""
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            return False
        
        try:
            logger.info(f"Rotation horaire: {angle}°")
            if not self.simulation_mode:
                self.drone.rotate_clockwise(angle)
            return True
        except Exception as e:
            logger.error(f"Erreur rotation: {e}")
            return False
    
    def rotate_counter_clockwise(self, angle: int = 90) -> bool:
        """Rotation anti-horaire"""
        if self.state != DroneState.FLYING and self.state != DroneState.EXPLORING:
            return False
        
        try:
            logger.info(f"Rotation anti-horaire: {angle}°")
            if not self.simulation_mode:
                self.drone.rotate_counter_clockwise(angle)
            return True
        except Exception as e:
            logger.error(f"Erreur rotation: {e}")
            return False
    
    def get_telemetry(self) -> dict:
        """
        Récupère les données de télémétrie du drone
        
        Returns:
            Dictionnaire avec les données de télémétrie
        """
        if self.simulation_mode:
            return {
                'battery': self.battery_level,
                'height': int(self.position.z),
                'temperature': 25,
                'flight_time': 0,
                'tof_distance': 100,
                'position': self.position.to_tuple()
            }
        
        if not self.drone:
            return {}
        
        try:
            return {
                'battery': self.drone.get_battery(),
                'height': self.drone.get_height(),
                'temperature': self.drone.get_temperature(),
                'flight_time': self.drone.get_flight_time(),
                'tof_distance': self.drone.get_distance_tof(),
                'position': self.position.to_tuple()
            }
        except Exception as e:
            logger.error(f"Erreur télémétrie: {e}")
            return {}
    
    def return_to_home(self) -> bool:
        """
        Retourne à la position de départ
        
        Returns:
            True si le retour est réussi
        """
        logger.info("Retour à la base...")
        
        # Calcul du chemin retour (simplifié)
        dx = -self.position.x
        dy = -self.position.y
        
        try:
            # Retour sur l'axe X
            if dx > 0:
                while dx >= self.MIN_MOVE_DISTANCE:
                    move = min(dx, self.MAX_MOVE_DISTANCE)
                    self.move_forward(int(move))
                    dx -= move
            elif dx < 0:
                while abs(dx) >= self.MIN_MOVE_DISTANCE:
                    move = min(abs(dx), self.MAX_MOVE_DISTANCE)
                    self.move_back(int(move))
                    dx += move
            
            # Retour sur l'axe Y
            if dy > 0:
                while dy >= self.MIN_MOVE_DISTANCE:
                    move = min(dy, self.MAX_MOVE_DISTANCE)
                    self.move_left(int(move))
                    dy -= move
            elif dy < 0:
                while abs(dy) >= self.MIN_MOVE_DISTANCE:
                    move = min(abs(dy), self.MAX_MOVE_DISTANCE)
                    self.move_right(int(move))
                    dy += move
            
            logger.info("Retour à la base effectué")
            return True
            
        except Exception as e:
            logger.error(f"Erreur retour base: {e}")
            return False


# Test basique si exécuté directement
if __name__ == "__main__":
    print("=== Test du contrôleur Tello ===")
    
    controller = TelloController(simulation_mode=True)
    
    if controller.connect():
        controller.takeoff()
        
        # Test des mouvements de base
        controller.move_forward(50)
        controller.move_right(50)
        controller.move_up(50)
        controller.move_down(50)
        controller.move_left(50)
        controller.move_back(50)
        
        print(f"Télémétrie: {controller.get_telemetry()}")
        
        controller.land()
        controller.disconnect()
