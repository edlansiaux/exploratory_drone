#!/usr/bin/env python3
"""
Contrôleur optimisé pour drone Tello EDU - Exploration de bâtiments
Version avec gestion complète de l'orientation et mouvements sécurisés

Réglages alignés sur le notebook de terrain (drone__1_.ipynb):
  - Connexion par host explicite "192.168.10.1"
  - Temporisation de stabilisation après streamon (2s)
  - djitellopy >= 2.5.0
"""

import time
import math
import logging
from typing import Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)

# Adresse IP par défaut du Tello en mode point d'accès (cf. notebook terrain)
DEFAULT_TELLO_HOST = "192.168.10.1"
# Délai de stabilisation après streamon avant lecture des frames (cf. notebook: "TRÈS IMPORTANT")
STREAM_WARMUP_DELAY = 2.0


class DroneState(Enum):
    """États possibles du drone"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    FLYING = "flying"
    LANDING = "landing"
    EMERGENCY = "emergency"


@dataclass
class Position:
    """Position 3D avec orientation"""
    x: float = 0.0      # cm (axe avant/arrière)
    y: float = 0.0      # cm (axe gauche/droite)
    z: float = 0.0      # cm (altitude)
    yaw: float = 0.0    # degrés (orientation, 0=Nord)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )
    
    def copy(self) -> 'Position':
        return Position(self.x, self.y, self.z, self.yaw)


@dataclass
class TelemetryData:
    """Données de télémétrie du drone"""
    battery: int = 100
    temperature: float = 25.0
    height: float = 0.0
    tof_distance: float = 0.0
    barometer: float = 0.0
    flight_time: int = 0
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp: float = field(default_factory=time.time)


class TelloController:
    """
    Contrôleur optimisé pour Tello EDU
    Gère les mouvements, l'orientation et la télémétrie
    """
    
    # Constantes de sécurité
    MIN_ALTITUDE = 30       # cm
    MAX_ALTITUDE = 300      # cm
    MIN_MOVE_DIST = 20      # cm
    MAX_MOVE_DIST = 500     # cm
    MOVE_TIMEOUT = 15       # secondes
    
    def __init__(self, simulation_mode: bool = False,
                 host: Optional[str] = None, drone=None):
        """
        Initialise le contrôleur
        
        Args:
            simulation_mode: Si True, simule le drone sans connexion réelle
            host: Adresse IP du drone. Si None (recommandé en mode point d'accès),
                  djitellopy utilise son défaut (192.168.10.1). Ne renseigner que
                  pour un Tello en mode station sur un réseau domestique.
            drone: Instance djitellopy.Tello DÉJÀ connectée à réutiliser. Évite
                   d'ouvrir un second socket UDP (le Tello n'accepte qu'un client).
        """
        self.simulation_mode = simulation_mode
        self.host = host
        # Réutilisation d'une instance Tello existante (déjà connectée)
        self.drone = drone
        self._external_drone = drone is not None
        
        # État
        self.state = DroneState.DISCONNECTED
        self.position = Position()
        self.home_position = Position()
        self.telemetry = TelemetryData()
        
        # Thread safety
        self._lock = Lock()
        self._move_in_progress = False
        
        # Historique des mouvements (pour retour sécurisé)
        self.movement_history = []
        self.max_history = 100
        
        # Callbacks
        self.on_low_battery = None
        self.on_position_update = None
        
        host_label = self.host if self.host else "défaut djitellopy (192.168.10.1)"
        reuse = " [instance réutilisée]" if self._external_drone else ""
        logger.info(f"TelloController initialisé (simulation: {simulation_mode}, host: {host_label}){reuse}")
    
    def connect(self) -> bool:
        """Connexion au drone"""
        with self._lock:
            try:
                if self.simulation_mode:
                    logger.info("Mode simulation - connexion virtuelle")
                    self.state = DroneState.CONNECTED
                    self._init_simulation()
                    return True
                
                from djitellopy import Tello
                if self.drone is None:
                    # Créer l'instance. host=None => défaut djitellopy (192.168.10.1)
                    if self.host:
                        self.drone = Tello(host=self.host)
                    else:
                        self.drone = Tello()
                    self.drone.connect()
                else:
                    # Instance déjà fournie (réutilisée). On vérifie qu'elle répond
                    # sans rouvrir de socket ; un get_battery sert de ping applicatif.
                    logger.info("Réutilisation d'une instance Tello existante")
                
                # Vérifier la batterie (sert aussi de test de communication)
                battery = self.drone.get_battery()
                logger.info(f"Niveau de batterie : {battery}%")
                if battery < 10:
                    logger.error(f"Batterie critique: {battery}%")
                    return False
                
                self.state = DroneState.CONNECTED
                self._update_telemetry()
                
                conn_host = self.host if self.host else "192.168.10.1"
                logger.info(f"Connecté ({conn_host}) - Batterie: {battery}%")
                return True
                
            except Exception as e:
                logger.error(f"Erreur connexion: {e}")
                return False
    
    def start_video_stream(self) -> bool:
        """
        Active le flux vidéo du drone avec la temporisation de stabilisation.
        Reproduit la séquence du notebook terrain:
            drone.streamon(); time.sleep(2)  # TRÈS IMPORTANT
        """
        if self.simulation_mode:
            return True
        if not self.drone:
            logger.warning("Drone non connecté, impossible de démarrer le flux")
            return False
        try:
            self.drone.streamon()
            time.sleep(STREAM_WARMUP_DELAY)  # Stabilisation flux (cf. notebook)
            logger.info("Flux vidéo drone activé")
            return True
        except Exception as e:
            logger.error(f"Erreur streamon: {e}")
            return False
    
    def stop_video_stream(self):
        """Coupe le flux vidéo du drone proprement."""
        if self.simulation_mode or not self.drone:
            return
        try:
            self.drone.streamoff()
            logger.info("Flux vidéo drone coupé")
        except Exception as e:
            logger.warning(f"Erreur streamoff: {e}")
    
    def get_frame_reader(self):
        """
        Retourne le frame_reader djitellopy (BackgroundFrameRead).
        À utiliser après start_video_stream(). Les frames sont en BGR.
        """
        if self.simulation_mode or not self.drone:
            return None
        try:
            return self.drone.get_frame_read()
        except Exception as e:
            logger.error(f"Erreur get_frame_read: {e}")
            return None
    
    def disconnect(self):
        """Déconnexion propre du drone"""
        with self._lock:
            if self.state == DroneState.FLYING:
                self.land()
            
            if self.drone and not self.simulation_mode:
                try:
                    # Couper le flux avant de fermer (cf. notebook: streamoff puis end)
                    try:
                        self.drone.streamoff()
                    except Exception:
                        pass
                    # Ne fermer le socket que si nous l'avons ouvert nous-mêmes.
                    # Une instance fournie de l'extérieur reste gérée par l'appelant.
                    if not self._external_drone:
                        self.drone.end()
                except Exception:
                    pass
            
            self.state = DroneState.DISCONNECTED
            logger.info("Déconnecté")
    
    def _init_simulation(self):
        """Initialise les données de simulation"""
        self.telemetry = TelemetryData(
            battery=100,
            temperature=25.0,
            height=0,
            tof_distance=200
        )
        self.position = Position()
    
    def _update_telemetry(self):
        """Met à jour la télémétrie depuis le drone"""
        if self.simulation_mode:
            # Simulation de décharge batterie
            elapsed = time.time() - self.telemetry.timestamp
            if elapsed > 60:  # Chaque minute
                self.telemetry.battery = max(0, self.telemetry.battery - 1)
                self.telemetry.timestamp = time.time()
            return
        
        if self.drone:
            try:
                self.telemetry = TelemetryData(
                    battery=self.drone.get_battery(),
                    temperature=self.drone.get_temperature(),
                    height=self.drone.get_height(),
                    tof_distance=self.drone.get_distance_tof(),
                    barometer=self.drone.get_barometer(),
                    flight_time=self.drone.get_flight_time(),
                    timestamp=time.time()
                )
            except Exception as e:
                logger.warning(f"Erreur télémétrie: {e}")
    
    def get_telemetry(self) -> Dict:
        """Retourne les données de télémétrie"""
        self._update_telemetry()
        return {
            'battery': self.telemetry.battery,
            'temperature': self.telemetry.temperature,
            'height': self.telemetry.height,
            'tof_distance': self.telemetry.tof_distance,
            'position': self.position.to_tuple(),
            'yaw': self.position.yaw,
            'state': self.state.value
        }

    def is_imu_ready(self) -> bool:
        """
        Vérifie sommairement que le positionnement (IMU + caméra ventrale) est
        exploitable. Un Tello au-dessus d'un sol non texturé ou mal éclairé
        renvoie 'error No valid imu' sur les déplacements; cette vérification
        tente de l'anticiper en lisant l'accélération rapportée par le drone.

        Retourne True en simulation. En réel, retourne False si les valeurs
        d'accélération sont toutes nulles/indisponibles (signe d'un état non prêt).
        """
        if self.simulation_mode:
            return True
        if not self.drone:
            return False
        try:
            # djitellopy expose get_acceleration_x/y/z (mg) à partir du state.
            ax = self.drone.get_acceleration_x()
            ay = self.drone.get_acceleration_y()
            az = self.drone.get_acceleration_z()
            # Au repos, az ~ -1000 mg (gravité). Si tout est exactement 0, le
            # state n'est pas alimenté => positionnement non prêt.
            if ax == 0 and ay == 0 and az == 0:
                logger.warning("IMU non prêt: accélérations nulles (positionnement indisponible)")
                return False
            return True
        except Exception as e:
            logger.warning(f"Lecture état IMU impossible: {e}")
            # En cas de doute, ne pas bloquer la mission sur cette base seule.
            return True
    
    def takeoff(self) -> bool:
        """Décollage sécurisé"""
        with self._lock:
            if self.state != DroneState.CONNECTED:
                logger.warning(f"Impossible de décoller: état = {self.state}")
                return False
            
            try:
                if not self.simulation_mode:
                    self.drone.takeoff()
                    time.sleep(2)  # Stabilisation
                
                self.position.z = 80  # Altitude par défaut Tello
                self.home_position = self.position.copy()
                self.state = DroneState.FLYING
                
                logger.info("Décollage réussi")
                return True
                
            except Exception as e:
                logger.error(f"Erreur décollage: {e}")
                return False
    
    def land(self, retries: int = 3) -> bool:
        """
        Atterrissage sécurisé avec plusieurs tentatives.

        En cas de liaison Wi-Fi instable (WinError 10051), on retente le 'land'
        plusieurs fois. Note de sécurité: le firmware Tello déclenche de toute
        façon un auto-land après ~15s sans commande reçue.

        Args:
            retries: nombre de tentatives d'envoi de la commande land
        """
        with self._lock:
            if self.state != DroneState.FLYING:
                return True

            self.state = DroneState.LANDING

            if self.simulation_mode:
                self.position.z = 0
                self.state = DroneState.CONNECTED
                logger.info("Atterrissage réussi")
                return True

            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    self.drone.land()
                    self.position.z = 0
                    self.state = DroneState.CONNECTED
                    logger.info(f"Atterrissage réussi (tentative {attempt})")
                    return True
                except Exception as e:
                    last_err = e
                    logger.error(f"Erreur atterrissage (tentative {attempt}/{retries}): {e}")
                    time.sleep(1.0)

            logger.critical(
                "ÉCHEC ATTERRISSAGE après %d tentatives (%s). "
                "Le Tello devrait auto-atterrir après ~15s sans commande.",
                retries, last_err
            )
            return False
    
    def emergency_stop(self):
        """Arrêt d'urgence - coupe les moteurs"""
        self.state = DroneState.EMERGENCY
        
        if not self.simulation_mode and self.drone:
            try:
                self.drone.emergency()
            except:
                pass
        
        logger.critical("ARRÊT D'URGENCE ACTIVÉ")
    
    def _validate_move(self, distance: int) -> bool:
        """Valide un mouvement avant exécution"""
        if self.state != DroneState.FLYING:
            logger.warning("Drone pas en vol")
            return False
        
        if not (self.MIN_MOVE_DIST <= abs(distance) <= self.MAX_MOVE_DIST):
            logger.warning(f"Distance invalide: {distance}cm")
            return False
        
        # Vérifier batterie
        if self.telemetry.battery < 15:
            logger.warning("Batterie faible, mouvement refusé")
            if self.on_low_battery:
                self.on_low_battery(self.telemetry.battery)
            return False
        
        return True
    
    def _record_movement(self, dx: float, dy: float, dz: float, dyaw: float = 0):
        """Enregistre un mouvement pour le retour"""
        self.movement_history.append({
            'dx': dx, 'dy': dy, 'dz': dz, 'dyaw': dyaw,
            'timestamp': time.time()
        })
        
        if len(self.movement_history) > self.max_history:
            self.movement_history.pop(0)
    
    def _update_position(self, dx: float, dy: float, dz: float = 0, dyaw: float = 0):
        """Met à jour la position estimée en tenant compte de l'orientation"""
        # Rotation du vecteur de déplacement selon le yaw actuel
        yaw_rad = math.radians(self.position.yaw)
        
        # Transformation du repère drone vers repère monde
        world_dx = dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad)
        world_dy = dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        
        self.position.x += world_dx
        self.position.y += world_dy
        self.position.z += dz
        self.position.yaw = (self.position.yaw + dyaw) % 360
        
        self._record_movement(world_dx, world_dy, dz, dyaw)
        
        if self.on_position_update:
            self.on_position_update(self.position)
    
    def move_forward(self, distance: int = 50) -> bool:
        """Avance le drone (direction du nez)"""
        if not self._validate_move(distance):
            return False
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.move_forward(distance)
                
                self._update_position(distance, 0)
                logger.debug(f"Avant: {distance}cm")
                return True
            except Exception as e:
                logger.error(f"Erreur move_forward: {e}")
                return False
    
    def move_back(self, distance: int = 50) -> bool:
        """Recule le drone"""
        if not self._validate_move(distance):
            return False
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.move_back(distance)
                
                self._update_position(-distance, 0)
                logger.debug(f"Arrière: {distance}cm")
                return True
            except Exception as e:
                logger.error(f"Erreur move_back: {e}")
                return False
    
    def move_left(self, distance: int = 50) -> bool:
        """Déplace à gauche"""
        if not self._validate_move(distance):
            return False
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.move_left(distance)
                
                self._update_position(0, -distance)
                logger.debug(f"Gauche: {distance}cm")
                return True
            except Exception as e:
                logger.error(f"Erreur move_left: {e}")
                return False
    
    def move_right(self, distance: int = 50) -> bool:
        """Déplace à droite"""
        if not self._validate_move(distance):
            return False
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.move_right(distance)
                
                self._update_position(0, distance)
                logger.debug(f"Droite: {distance}cm")
                return True
            except Exception as e:
                logger.error(f"Erreur move_right: {e}")
                return False
    
    def move_up(self, distance: int = 50) -> bool:
        """Monte le drone"""
        if not self._validate_move(distance):
            return False
        
        # Vérifier altitude max
        if self.position.z + distance > self.MAX_ALTITUDE:
            distance = int(self.MAX_ALTITUDE - self.position.z)
            if distance < self.MIN_MOVE_DIST:
                logger.warning("Altitude max atteinte")
                return False
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.move_up(distance)
                
                self._update_position(0, 0, distance)
                logger.debug(f"Haut: {distance}cm")
                return True
            except Exception as e:
                logger.error(f"Erreur move_up: {e}")
                return False
    
    def move_down(self, distance: int = 50) -> bool:
        """Descend le drone"""
        if not self._validate_move(distance):
            return False
        
        # Vérifier altitude min
        if self.position.z - distance < self.MIN_ALTITUDE:
            distance = int(self.position.z - self.MIN_ALTITUDE)
            if distance < self.MIN_MOVE_DIST:
                logger.warning("Altitude min atteinte")
                return False
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.move_down(distance)
                
                self._update_position(0, 0, -distance)
                logger.debug(f"Bas: {distance}cm")
                return True
            except Exception as e:
                logger.error(f"Erreur move_down: {e}")
                return False
    
    def rotate_clockwise(self, angle: int = 90) -> bool:
        """Rotation horaire"""
        if self.state != DroneState.FLYING:
            return False
        
        angle = max(1, min(360, abs(angle)))
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.rotate_clockwise(angle)
                
                self._update_position(0, 0, 0, angle)
                time.sleep(0.5)  # Stabilisation
                logger.debug(f"Rotation CW: {angle}°")
                return True
            except Exception as e:
                logger.error(f"Erreur rotation: {e}")
                return False
    
    def rotate_counter_clockwise(self, angle: int = 90) -> bool:
        """Rotation anti-horaire"""
        if self.state != DroneState.FLYING:
            return False
        
        angle = max(1, min(360, abs(angle)))
        
        with self._lock:
            try:
                if not self.simulation_mode:
                    self.drone.rotate_counter_clockwise(angle)
                
                self._update_position(0, 0, 0, -angle)
                time.sleep(0.5)
                logger.debug(f"Rotation CCW: {angle}°")
                return True
            except Exception as e:
                logger.error(f"Erreur rotation: {e}")
                return False
    
    def rotate_to_angle(self, target_angle: float) -> bool:
        """
        Rotation vers un angle absolu (0-360°)
        
        Args:
            target_angle: Angle cible en degrés (0=Nord)
        """
        # Normaliser les angles
        target_angle = target_angle % 360
        current_angle = self.position.yaw % 360
        
        # Calculer la différence
        diff = target_angle - current_angle
        
        # Prendre le chemin le plus court
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        if abs(diff) < 5:  # Tolérance de 5°
            return True
        
        if diff > 0:
            return self.rotate_clockwise(int(diff))
        else:
            return self.rotate_counter_clockwise(int(abs(diff)))
    
    def return_to_home(self) -> bool:
        """Retour au point de départ"""
        logger.info("Retour à la base...")
        
        # Calculer la direction vers home
        dx = self.home_position.x - self.position.x
        dy = self.home_position.y - self.position.y
        
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 20:
            logger.info("Déjà à la base")
            return True
        
        # Angle vers home
        angle = math.degrees(math.atan2(dy, dx))
        
        # Tourner vers home
        self.rotate_to_angle(angle)
        time.sleep(0.5)
        
        # Avancer par étapes
        while distance > 20:
            step = min(100, int(distance))
            if not self.move_forward(step):
                return False
            
            dx = self.home_position.x - self.position.x
            dy = self.home_position.y - self.position.y
            distance = math.sqrt(dx**2 + dy**2)
            
            time.sleep(0.3)
        
        logger.info("Retour à la base terminé")
        return True
    
    def get_camera_direction(self) -> Tuple[float, float, float]:
        """
        Retourne le vecteur direction de la caméra (normalisé)
        """
        yaw_rad = math.radians(self.position.yaw)
        return (
            math.cos(yaw_rad),  # X (avant)
            math.sin(yaw_rad),  # Y (côté)
            0                    # Z (horizontal)
        )


# Test unitaire
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Test TelloController ===\n")
    
    controller = TelloController(simulation_mode=True)
    
    # Test connexion
    assert controller.connect(), "Échec connexion"
    print(f"État: {controller.state}")
    
    # Test décollage
    assert controller.takeoff(), "Échec décollage"
    print(f"Position: {controller.position.to_tuple()}")
    
    # Test mouvements
    controller.move_forward(100)
    print(f"Après avance: {controller.position.to_tuple()}")
    
    controller.rotate_clockwise(90)
    print(f"Yaw: {controller.position.yaw}°")
    
    controller.move_forward(50)
    print(f"Après rotation+avance: {controller.position.to_tuple()}")
    
    # Test retour
    controller.return_to_home()
    print(f"Après retour: {controller.position.to_tuple()}")
    
    # Atterrissage
    controller.land()
    controller.disconnect()
    
    print("\n✓ Tests réussis")
