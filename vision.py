#!/usr/bin/env python3
"""
Module de vision optimisé pour Tello EDU - Exploration de bâtiments
Inclut détection d'obstacles et simulation thermique

Réglages alignés sur le notebook de terrain (drone__1_.ipynb):
  - Résolution réelle de la caméra Tello: 960x720 (largeur x hauteur)
    => frame numpy de shape (720, 960, 3)
  - Séquence flux réel: streamon() puis time.sleep(2) AVANT get_frame_read()
  - Les frames du drone sont en BGR (conversion BGR->RGB pour affichage uniquement)
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

# Délai de stabilisation après streamon avant lecture des frames (cf. notebook: "TRÈS IMPORTANT")
STREAM_WARMUP_DELAY = 2.0


class ObstacleType(Enum):
    """Types d'obstacles détectables"""
    UNKNOWN = "unknown"
    WALL = "wall"
    DEBRIS = "debris"
    PERSON = "person"
    OBJECT = "object"
    FIRE = "fire"
    SMOKE = "smoke"
    HOLE = "hole"
    CEILING = "ceiling"
    FLOOR = "floor"


class ThreatLevel(Enum):
    """Niveau de menace"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class VisualObstacle:
    """Obstacle détecté visuellement"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    area: int
    distance_estimate: float         # cm
    obstacle_type: ObstacleType = ObstacleType.UNKNOWN
    confidence: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.SAFE
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_close(self) -> bool:
        return self.distance_estimate < 80
    
    @property
    def is_dangerous(self) -> bool:
        return self.threat_level.value >= ThreatLevel.HIGH.value


@dataclass
class ThermalPoint:
    """Point thermique détecté"""
    x: float
    y: float
    temperature: float  # °C estimée
    intensity: float    # 0-1
    timestamp: float = field(default_factory=time.time)


@dataclass
class FrameFeatures:
    """Features extraites d'une frame"""
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    timestamp: float
    frame_id: int


class VideoStream:
    """
    Gestionnaire de flux vidéo optimisé
    Support caméra Tello et simulation
    """
    
    # Résolution native de la caméra Tello: largeur=960, hauteur=720
    # (cf. notebook: frame_reader.frame de shape (720, 960, 3))
    DEFAULT_FRAME_SIZE = (960, 720)  # (width, height)
    
    def __init__(self, drone=None, simulation_mode: bool = False):
        self.drone = drone
        self.simulation_mode = simulation_mode
        
        self.is_streaming = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps = 0
        # (width, height) - cohérent avec la résolution réelle du Tello
        self.frame_size = self.DEFAULT_FRAME_SIZE
        
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_capture = threading.Event()
        self._frame_lock = threading.Lock()
        
        self.frame_buffer: deque = deque(maxlen=30)
        self._sim_cap: Optional[cv2.VideoCapture] = None
        self._frame_reader = None  # BackgroundFrameRead djitellopy
        
        # Callbacks
        self.on_frame_received: Optional[Callable] = None
        
        logger.info(f"VideoStream initialisé (simulation: {simulation_mode}, "
                    f"résolution: {self.frame_size[0]}x{self.frame_size[1]})")
    
    def start(self) -> bool:
        """Démarre le flux vidéo"""
        if self.is_streaming:
            return True
        
        try:
            if self.simulation_mode:
                self._sim_cap = cv2.VideoCapture(0)
                if not self._sim_cap.isOpened():
                    logger.warning("Pas de webcam, frames synthétiques")
                    self._sim_cap = None
            else:
                if self.drone:
                    # Séquence du notebook terrain: streamon puis attente de stabilisation
                    self.drone.streamon()
                    time.sleep(STREAM_WARMUP_DELAY)  # TRÈS IMPORTANT (cf. notebook)
                    # Récupérer le frame reader une fois le flux stabilisé
                    self._frame_reader = self.drone.get_frame_read()
                    time.sleep(0.5)
            
            self._stop_capture.clear()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            
            self.is_streaming = True
            logger.info("Flux vidéo démarré")
            return True
            
        except Exception as e:
            logger.error(f"Erreur démarrage vidéo: {e}")
            return False
    
    def stop(self):
        """Arrête le flux vidéo"""
        self._stop_capture.set()
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        
        if self.simulation_mode and self._sim_cap:
            self._sim_cap.release()
        elif self.drone:
            try:
                self.drone.streamoff()
            except:
                pass
        
        self._frame_reader = None
        self.is_streaming = False
        logger.info("Flux vidéo arrêté")
    
    def _capture_loop(self):
        """Boucle de capture"""
        last_time = time.time()
        frame_times = deque(maxlen=30)
        
        while not self._stop_capture.is_set():
            try:
                frame = self._get_frame()
                
                if frame is not None:
                    with self._frame_lock:
                        self.current_frame = frame
                        self.frame_count += 1
                        self.frame_buffer.append((frame.copy(), time.time()))
                    
                    # FPS
                    current_time = time.time()
                    frame_times.append(current_time - last_time)
                    last_time = current_time
                    if len(frame_times) > 0:
                        self.fps = 1.0 / (sum(frame_times) / len(frame_times))
                    
                    if self.on_frame_received:
                        self.on_frame_received(frame)
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Erreur capture: {e}")
                time.sleep(0.1)
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Récupère une frame (BGR)"""
        if self.simulation_mode:
            if self._sim_cap and self._sim_cap.isOpened():
                ret, frame = self._sim_cap.read()
                if ret:
                    return cv2.resize(frame, self.frame_size)
            return self._generate_synthetic_frame()
        else:
            # Lecture via le frame reader obtenu après streamon + sleep(2)
            if self._frame_reader is not None:
                frame = self._frame_reader.frame
                return frame
            if self.drone:
                return self.drone.get_frame_read().frame
        return None
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Génère une frame synthétique simulant un bâtiment délabré"""
        frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Fond sombre (intérieur bâtiment)
        frame[:] = [30, 35, 40]
        
        # Sol avec texture
        for y in range(500, self.frame_size[1]):
            intensity = 50 + int((y - 500) / 3)
            frame[y, :] = [intensity, intensity-5, intensity-10]
        
        # Murs latéraux
        t = time.time()
        
        # Mur gauche
        pts_left = np.array([[0, 200], [150, 300], [150, 600], [0, 720]], np.int32)
        cv2.fillPoly(frame, [pts_left], (45, 50, 55))
        
        # Mur droit
        pts_right = np.array([[960, 200], [810, 300], [810, 600], [960, 720]], np.int32)
        cv2.fillPoly(frame, [pts_right], (45, 50, 55))
        
        # Débris au sol (obstacles)
        debris_positions = [
            (300, 550, 60, 40),
            (600, 520, 80, 50),
            (450, 580, 50, 30),
        ]
        for dx, dy, dw, dh in debris_positions:
            offset = int(10 * np.sin(t + dx))
            cv2.rectangle(frame, (dx + offset, dy), (dx + dw + offset, dy + dh), (60, 55, 50), -1)
        
        # Trou au sol (danger)
        hole_x = 200 + int(50 * np.sin(t * 0.5))
        cv2.ellipse(frame, (hole_x, 650), (60, 30), 0, 0, 360, (10, 10, 15), -1)
        
        # Zone de feu/chaleur (point chaud simulé)
        fire_x = 700
        fire_y = 400
        fire_intensity = int(100 + 50 * np.sin(t * 5))
        cv2.circle(frame, (fire_x, fire_y), 40, (30, 50 + fire_intensity//2, fire_intensity), -1)
        cv2.circle(frame, (fire_x, fire_y), 25, (50, 100 + fire_intensity//2, min(255, fire_intensity + 100)), -1)
        
        # Fumée (particules)
        for _ in range(20):
            sx = fire_x + np.random.randint(-50, 50)
            sy = fire_y - np.random.randint(50, 200)
            sr = np.random.randint(5, 15)
            alpha = np.random.randint(20, 60)
            cv2.circle(frame, (sx, sy), sr, (alpha, alpha, alpha), -1)
        
        # Personne au sol (détection victime)
        if np.sin(t * 0.3) > 0:
            person_x = 500
            cv2.ellipse(frame, (person_x, 620), (25, 40), 0, 0, 360, (80, 70, 65), -1)
            cv2.circle(frame, (person_x, 580), 15, (100, 90, 85), -1)  # Tête
        
        # Plafond endommagé
        for i in range(0, 960, 100):
            drop = int(30 * np.sin(i / 50 + t))
            cv2.line(frame, (i, 0), (i + 50, 100 + drop), (35, 40, 45), 3)
        
        # Texte informatif
        cv2.putText(frame, "SIMULATION - BATIMENT DELABRE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        
        # Ajouter du bruit réaliste
        noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        return frame
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Retourne la frame courante (BGR)"""
        with self._frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """
        Retourne la frame courante convertie en RGB (pour affichage matplotlib/notebook).
        Reproduit cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) du notebook terrain.
        """
        frame = self.get_frame()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def get_recent_frames(self, n: int = 10) -> List[Tuple[np.ndarray, float]]:
        """Retourne les n dernières frames"""
        return list(self.frame_buffer)[-n:]


class ThermalDetector:
    """
    Détecteur de zones thermiques
    Simule une caméra thermique en analysant les couleurs chaudes
    """
    
    def __init__(self):
        # Seuils de température simulée
        self.temp_thresholds = {
            'cold': 20,
            'normal': 30,
            'warm': 50,
            'hot': 80,
            'fire': 150
        }
        
        # Historique des détections
        self.thermal_history: deque = deque(maxlen=100)
        self.hotspots: List[ThermalPoint] = []
        
        logger.info("ThermalDetector initialisé")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[ThermalPoint]]:
        """
        Analyse thermique de la frame
        
        Args:
            frame: Image BGR
            
        Returns:
            (thermal_map, hotspots)
        """
        if frame is None:
            return None, []
        
        # Conversion HSV pour détecter les couleurs chaudes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Masque pour couleurs chaudes (orange/rouge/jaune)
        # Bas : 0-30 (rouge/orange), Haut: 160-180 (rouge)
        mask_low = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([30, 255, 255]))
        mask_high = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        mask_warm = cv2.bitwise_or(mask_low, mask_high)
        
        # Détection de zones blanches/jaunes (très chaud)
        mask_hot = cv2.inRange(hsv, np.array([15, 50, 200]), np.array([35, 255, 255]))
        
        # Combinaison
        thermal_mask = cv2.bitwise_or(mask_warm, mask_hot)
        
        # Création de la carte thermique
        thermal_map = self._create_thermal_map(frame, thermal_mask, mask_hot)
        
        # Détection des points chauds
        hotspots = self._find_hotspots(thermal_mask, mask_hot, frame.shape)
        
        self.hotspots = hotspots
        self.thermal_history.append((time.time(), len(hotspots)))
        
        return thermal_map, hotspots
    
    def _create_thermal_map(self, frame: np.ndarray, warm_mask: np.ndarray, 
                           hot_mask: np.ndarray) -> np.ndarray:
        """Crée une visualisation thermique colorée"""
        # Base en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Appliquer une colormap thermique
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Accentuer les zones chaudes
        thermal[warm_mask > 0] = [0, 100, 255]  # Orange
        thermal[hot_mask > 0] = [0, 0, 255]     # Rouge vif
        
        # Ajouter légende de température
        cv2.putText(thermal, "THERMAL VIEW", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Barre de température
        bar_height = 150
        bar_width = 20
        bar_x = thermal.shape[1] - 40
        for i in range(bar_height):
            temp_color = cv2.applyColorMap(
                np.array([[int(255 * i / bar_height)]], dtype=np.uint8),
                cv2.COLORMAP_JET
            )[0, 0]
            cv2.line(thermal, (bar_x, 50 + bar_height - i), 
                    (bar_x + bar_width, 50 + bar_height - i), 
                    temp_color.tolist(), 1)
        
        cv2.putText(thermal, "150C", (bar_x - 5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(thermal, "20C", (bar_x - 5, 50 + bar_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return thermal
    
    def _find_hotspots(self, warm_mask: np.ndarray, hot_mask: np.ndarray,
                       shape: Tuple) -> List[ThermalPoint]:
        """Identifie les points chauds"""
        hotspots = []
        
        # Trouver les contours des zones chaudes
        contours, _ = cv2.findContours(warm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Ignorer les petites zones
                continue
            
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Estimer la température basée sur l'intensité
            is_hot = hot_mask[cy, cx] > 0 if cy < hot_mask.shape[0] and cx < hot_mask.shape[1] else False
            
            intensity = min(1.0, area / 5000)
            temp = self.temp_thresholds['warm'] + intensity * 100
            
            if is_hot:
                temp = self.temp_thresholds['fire']
            
            hotspots.append(ThermalPoint(
                x=cx / shape[1],  # Normalisé 0-1
                y=cy / shape[0],
                temperature=temp,
                intensity=intensity
            ))
        
        return hotspots
    
    def get_max_temperature(self) -> float:
        """Retourne la température max détectée"""
        if not self.hotspots:
            return self.temp_thresholds['normal']
        return max(h.temperature for h in self.hotspots)
    
    def has_fire_detected(self) -> bool:
        """Vérifie si du feu est détecté"""
        return any(h.temperature >= self.temp_thresholds['fire'] for h in self.hotspots)


class ObstacleDetector:
    """
    Détecteur d'obstacles optimisé pour environnements dégradés
    """
    
    def __init__(self):
        # Paramètres de détection
        self.min_obstacle_area = 800
        self.max_obstacle_area = 400000
        
        # Distances de sécurité (cm)
        self.danger_distance = 50
        self.warning_distance = 100
        self.safe_distance = 150
        
        # Paramètres de calibration caméra Tello
        self.focal_length = 700
        self.known_width = 50
        
        # Historique
        self.detection_history: deque = deque(maxlen=30)
        self.last_detections: List[VisualObstacle] = []
        
        # Détection de mouvement
        self.prev_frame_gray = None
        self.motion_threshold = 25
        
        logger.info("ObstacleDetector initialisé")
    
    def detect(self, frame: np.ndarray) -> List[VisualObstacle]:
        """
        Détecte les obstacles dans la frame
        
        Args:
            frame: Image BGR
            
        Returns:
            Liste d'obstacles détectés
        """
        if frame is None:
            return []
        
        obstacles = []
        
        # 1. Détection par contours (obstacles statiques)
        obstacles.extend(self._detect_by_contours(frame))
        
        # 2. Détection par mouvement (obstacles mobiles/personnes)
        obstacles.extend(self._detect_motion(frame))
        
        # 3. Détection de zones dangereuses
        obstacles.extend(self._detect_danger_zones(frame))
        
        # 4. Détection de trous/vides
        obstacles.extend(self._detect_holes(frame))
        
        # Fusion et filtrage
        obstacles = self._merge_detections(obstacles)
        
        # Mise à jour historique
        self.last_detections = obstacles
        self.detection_history.append(obstacles)
        
        return obstacles
    
    def _detect_by_contours(self, frame: np.ndarray) -> List[VisualObstacle]:
        """Détection par analyse de contours"""
        obstacles = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Détection de bords
        edges = cv2.Canny(blurred, 30, 100)
        
        # Morphologie pour connecter les contours
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_obstacle_area < area < self.max_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                
                distance = self._estimate_distance(w, h, y, frame.shape[0])
                obs_type = self._classify_obstacle(frame, (x, y, w, h))
                threat = self._assess_threat(distance, obs_type)
                
                obstacles.append(VisualObstacle(
                    bbox=(x, y, w, h),
                    center=center,
                    area=area,
                    distance_estimate=distance,
                    obstacle_type=obs_type,
                    confidence=min(1.0, area / 8000),
                    threat_level=threat
                ))
        
        return obstacles
    
    def _detect_motion(self, frame: np.ndarray) -> List[VisualObstacle]:
        """Détection d'objets en mouvement"""
        obstacles = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return []
        
        # Différence entre frames
        frame_diff = cv2.absdiff(self.prev_frame_gray, gray)
        thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 1500:  # Mouvement significatif
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                
                distance = self._estimate_distance(w, h, y, frame.shape[0])
                
                # Mouvement = potentiellement une personne ou danger mobile
                obs_type = ObstacleType.PERSON if h > w else ObstacleType.DEBRIS
                
                obstacles.append(VisualObstacle(
                    bbox=(x, y, w, h),
                    center=center,
                    area=area,
                    distance_estimate=distance,
                    obstacle_type=obs_type,
                    confidence=0.7,
                    threat_level=ThreatLevel.HIGH if distance < self.warning_distance else ThreatLevel.MEDIUM
                ))
        
        self.prev_frame_gray = gray
        return obstacles
    
    def _detect_danger_zones(self, frame: np.ndarray) -> List[VisualObstacle]:
        """Détecte les zones de danger immédiat"""
        obstacles = []
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Zone centrale (devant le drone)
        center_region = gray[h//3:2*h//3, w//4:3*w//4]
        
        # Zones très sombres = obstacle proche
        _, dark_thresh = cv2.threshold(center_region, 40, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(dark_thresh > 0) / dark_thresh.size
        
        if dark_ratio > 0.4:
            obstacles.append(VisualObstacle(
                bbox=(w//4, h//3, w//2, h//3),
                center=(w//2, h//2),
                area=int(w//2 * h//3),
                distance_estimate=max(30, 80 * (1 - dark_ratio)),
                obstacle_type=ObstacleType.WALL,
                confidence=dark_ratio,
                threat_level=ThreatLevel.CRITICAL if dark_ratio > 0.6 else ThreatLevel.HIGH
            ))
        
        return obstacles
    
    def _detect_holes(self, frame: np.ndarray) -> List[VisualObstacle]:
        """Détecte les trous/vides au sol"""
        obstacles = []
        h, w = frame.shape[:2]
        
        # Zone du sol (partie basse de l'image)
        floor_region = frame[2*h//3:, :]
        gray_floor = cv2.cvtColor(floor_region, cv2.COLOR_BGR2GRAY)
        
        # Trous = zones très sombres au sol
        _, hole_thresh = cv2.threshold(gray_floor, 30, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(hole_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Trou significatif
                x, y, bw, bh = cv2.boundingRect(contour)
                y_global = y + 2*h//3
                
                obstacles.append(VisualObstacle(
                    bbox=(x, y_global, bw, bh),
                    center=(x + bw//2, y_global + bh//2),
                    area=area,
                    distance_estimate=100,  # Au sol
                    obstacle_type=ObstacleType.HOLE,
                    confidence=0.8,
                    threat_level=ThreatLevel.CRITICAL
                ))
        
        return obstacles
    
    def _estimate_distance(self, width: int, height: int, y_pos: int, 
                          frame_height: int) -> float:
        """Estime la distance basée sur taille et position"""
        apparent_size = max(width, height)
        
        # Distance par taille
        if apparent_size > 0:
            dist_size = (self.known_width * self.focal_length) / apparent_size
        else:
            dist_size = 500
        
        # Ajustement par position verticale (bas = plus proche)
        y_factor = 1.0 - (y_pos / frame_height) * 0.5
        
        distance = dist_size * y_factor
        return max(20, min(400, distance))
    
    def _classify_obstacle(self, frame: np.ndarray, bbox: Tuple) -> ObstacleType:
        """Classifie le type d'obstacle"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return ObstacleType.UNKNOWN
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_val = np.mean(hsv[:, :, 2])
        mean_sat = np.mean(hsv[:, :, 1])
        
        aspect_ratio = w / h if h > 0 else 1
        
        # Feu/chaleur (haute saturation, couleurs chaudes)
        if mean_sat > 100:
            mask_fire = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([30, 255, 255]))
            if np.sum(mask_fire > 0) / mask_fire.size > 0.3:
                return ObstacleType.FIRE
        
        # Classification par forme
        if aspect_ratio > 3:
            return ObstacleType.WALL
        elif aspect_ratio < 0.4:
            return ObstacleType.PERSON
        elif y < frame.shape[0] // 4:
            return ObstacleType.CEILING
        elif mean_val < 40:
            return ObstacleType.HOLE
        
        return ObstacleType.DEBRIS
    
    def _assess_threat(self, distance: float, obs_type: ObstacleType) -> ThreatLevel:
        """Évalue le niveau de menace"""
        # Menaces critiques
        if obs_type in [ObstacleType.FIRE, ObstacleType.HOLE]:
            return ThreatLevel.CRITICAL
        
        # Par distance
        if distance < self.danger_distance:
            return ThreatLevel.CRITICAL
        elif distance < self.warning_distance:
            return ThreatLevel.HIGH
        elif distance < self.safe_distance:
            return ThreatLevel.MEDIUM
        
        return ThreatLevel.LOW
    
    def _merge_detections(self, obstacles: List[VisualObstacle]) -> List[VisualObstacle]:
        """Fusionne les détections qui se chevauchent"""
        if len(obstacles) <= 1:
            return obstacles
        
        merged = []
        used = set()
        
        for i, obs1 in enumerate(obstacles):
            if i in used:
                continue
            
            overlapping = [obs1]
            
            for j, obs2 in enumerate(obstacles[i+1:], start=i+1):
                if j in used:
                    continue
                
                if self._boxes_overlap(obs1.bbox, obs2.bbox):
                    overlapping.append(obs2)
                    used.add(j)
            
            if len(overlapping) > 1:
                merged.append(self._merge_obstacles(overlapping))
            else:
                merged.append(obs1)
            
            used.add(i)
        
        return merged
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple, threshold: float = 0.3) -> bool:
        """Vérifie le chevauchement de deux boîtes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union > threshold if union > 0 else False
    
    def _merge_obstacles(self, obstacles: List[VisualObstacle]) -> VisualObstacle:
        """Fusionne plusieurs obstacles"""
        x_min = min(o.bbox[0] for o in obstacles)
        y_min = min(o.bbox[1] for o in obstacles)
        x_max = max(o.bbox[0] + o.bbox[2] for o in obstacles)
        y_max = max(o.bbox[1] + o.bbox[3] for o in obstacles)
        
        w, h = x_max - x_min, y_max - y_min
        
        total_conf = sum(o.confidence for o in obstacles)
        avg_distance = sum(o.distance_estimate * o.confidence for o in obstacles) / max(total_conf, 0.1)
        
        best_type = max(obstacles, key=lambda o: o.confidence).obstacle_type
        max_threat = max(obstacles, key=lambda o: o.threat_level.value).threat_level
        
        return VisualObstacle(
            bbox=(x_min, y_min, w, h),
            center=(x_min + w // 2, y_min + h // 2),
            area=w * h,
            distance_estimate=avg_distance,
            obstacle_type=best_type,
            confidence=min(1.0, total_conf / len(obstacles)),
            threat_level=max_threat
        )
    
    def draw_detections(self, frame: np.ndarray, obstacles: List[VisualObstacle]) -> np.ndarray:
        """Dessine les détections sur la frame"""
        result = frame.copy()
        
        for obs in obstacles:
            x, y, w, h = obs.bbox
            
            # Couleur selon menace
            colors = {
                ThreatLevel.SAFE: (0, 255, 0),
                ThreatLevel.LOW: (0, 200, 100),
                ThreatLevel.MEDIUM: (0, 165, 255),
                ThreatLevel.HIGH: (0, 100, 255),
                ThreatLevel.CRITICAL: (0, 0, 255)
            }
            color = colors.get(obs.threat_level, (255, 255, 255))
            
            # Boîte
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"{obs.obstacle_type.value}: {obs.distance_estimate:.0f}cm"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Point central
            cv2.circle(result, obs.center, 5, color, -1)
        
        # Statistiques
        cv2.putText(result, f"Obstacles: {len(obstacles)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        critical = sum(1 for o in obstacles if o.threat_level == ThreatLevel.CRITICAL)
        if critical > 0:
            cv2.putText(result, f"DANGER: {critical} critiques!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def get_closest_obstacle(self) -> Optional[VisualObstacle]:
        """Retourne l'obstacle le plus proche"""
        if not self.last_detections:
            return None
        return min(self.last_detections, key=lambda o: o.distance_estimate)
    
    def is_path_clear(self, min_distance: float = 80) -> Tuple[bool, Optional[VisualObstacle]]:
        """Vérifie si le chemin est dégagé"""
        for obs in self.last_detections:
            if obs.distance_estimate < min_distance:
                return False, obs
        return True, None


class FeatureExtractor:
    """Extracteur de features pour le SLAM"""
    
    def __init__(self, max_features: int = 1500):
        self.max_features = max_features
        
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=15
        )
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.frame_id = 0
        
        logger.info(f"FeatureExtractor initialisé (max: {max_features})")
    
    def extract(self, frame: np.ndarray) -> FrameFeatures:
        """Extrait les features d'une frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        features = FrameFeatures(
            keypoints=list(keypoints) if keypoints else [],
            descriptors=descriptors if descriptors is not None else np.array([]),
            timestamp=time.time(),
            frame_id=self.frame_id
        )
        
        self.frame_id += 1
        return features
    
    def match_features(self, f1: FrameFeatures, f2: FrameFeatures) -> List[cv2.DMatch]:
        """Match les features entre deux frames"""
        if f1.descriptors is None or f2.descriptors is None:
            return []
        if len(f1.descriptors) < 2 or len(f2.descriptors) < 2:
            return []
        
        try:
            matches = self.bf_matcher.match(f1.descriptors, f2.descriptors)
            return sorted(matches, key=lambda x: x.distance)
        except:
            return []


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Test module vision ===\n")
    
    video = VideoStream(simulation_mode=True)
    detector = ObstacleDetector()
    thermal = ThermalDetector()
    
    video.start()
    time.sleep(1)
    
    for i in range(5):
        frame = video.get_frame()
        
        if frame is not None:
            obstacles = detector.detect(frame)
            thermal_map, hotspots = thermal.detect(frame)
            
            print(f"Frame {i+1}: shape={frame.shape}")
            print(f"  - Obstacles: {len(obstacles)}")
            print(f"  - Points chauds: {len(hotspots)}")
            print(f"  - Temp max: {thermal.get_max_temperature():.1f}°C")
            
            if thermal.has_fire_detected():
                print("  - ⚠️ FEU DÉTECTÉ!")
        
        time.sleep(0.5)
    
    video.stop()
    print("\n✓ Test terminé")
