#!/usr/bin/env python3
"""
Module de vision pour drone Tello EDU
- Flux vidéo en temps réel
- Détection d'obstacles par vision
- Extraction de features pour SLAM
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


class ObstacleType(Enum):
    """Types d'obstacles détectables"""
    UNKNOWN = "unknown"
    WALL = "wall"
    PERSON = "person"
    OBJECT = "object"
    CEILING = "ceiling"
    FLOOR = "floor"


@dataclass
class VisualObstacle:
    """Obstacle détecté visuellement"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    area: int
    distance_estimate: float  # Distance estimée en cm
    obstacle_type: ObstacleType = ObstacleType.UNKNOWN
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_close(self) -> bool:
        """Vérifie si l'obstacle est proche (< 100cm)"""
        return self.distance_estimate < 100


@dataclass
class FrameFeatures:
    """Features extraites d'une frame pour le SLAM"""
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    timestamp: float
    frame_id: int


class VideoStream:
    """
    Gestionnaire du flux vidéo du drone Tello
    Gère la capture, le buffering et la distribution des frames
    """
    
    def __init__(self, drone=None, simulation_mode: bool = False):
        """
        Initialise le flux vidéo
        
        Args:
            drone: Instance du drone Tello (djitellopy)
            simulation_mode: Si True, utilise une source vidéo simulée
        """
        self.drone = drone
        self.simulation_mode = simulation_mode
        
        self.is_streaming = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps = 0
        self.frame_size = (960, 720)  # Résolution Tello
        
        # Thread de capture
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_capture = threading.Event()
        
        # Callbacks
        self.on_frame_received: Optional[Callable] = None
        
        # Buffer de frames
        self.frame_buffer: deque = deque(maxlen=30)
        
        # Source vidéo simulée
        self._sim_cap: Optional[cv2.VideoCapture] = None
        
        logger.info(f"VideoStream initialisé (simulation: {simulation_mode})")
    
    def start(self) -> bool:
        """
        Démarre le flux vidéo
        
        Returns:
            True si le flux est démarré avec succès
        """
        if self.is_streaming:
            return True
        
        try:
            if self.simulation_mode:
                # En simulation, utiliser la webcam ou un fichier vidéo
                self._sim_cap = cv2.VideoCapture(0)
                if not self._sim_cap.isOpened():
                    # Générer des frames synthétiques si pas de webcam
                    logger.warning("Pas de webcam, utilisation de frames synthétiques")
                    self._sim_cap = None
            else:
                # Démarrer le stream du drone
                if self.drone:
                    self.drone.streamon()
                    time.sleep(2)  # Attendre l'initialisation
            
            # Démarrer le thread de capture
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
        
        self.is_streaming = False
        logger.info("Flux vidéo arrêté")
    
    def _capture_loop(self):
        """Boucle de capture vidéo"""
        last_time = time.time()
        frame_times = deque(maxlen=30)
        
        while not self._stop_capture.is_set():
            try:
                frame = self._get_frame()
                
                if frame is not None:
                    self.current_frame = frame
                    self.frame_count += 1
                    self.frame_buffer.append((frame.copy(), time.time()))
                    
                    # Calcul FPS
                    current_time = time.time()
                    frame_times.append(current_time - last_time)
                    last_time = current_time
                    if len(frame_times) > 0:
                        self.fps = 1.0 / (sum(frame_times) / len(frame_times))
                    
                    # Callback
                    if self.on_frame_received:
                        self.on_frame_received(frame)
                
                time.sleep(0.01)  # ~100 Hz max
                
            except Exception as e:
                logger.error(f"Erreur capture: {e}")
                time.sleep(0.1)
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Récupère une frame depuis la source appropriée"""
        if self.simulation_mode:
            if self._sim_cap and self._sim_cap.isOpened():
                ret, frame = self._sim_cap.read()
                if ret:
                    return cv2.resize(frame, self.frame_size)
            
            # Générer une frame synthétique
            return self._generate_synthetic_frame()
        else:
            if self.drone:
                frame = self.drone.get_frame_read().frame
                return frame
        
        return None
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Génère une frame synthétique pour la simulation"""
        frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Fond avec gradient (simule le sol/ciel)
        for y in range(self.frame_size[1]):
            intensity = int(50 + (y / self.frame_size[1]) * 100)
            frame[y, :] = [intensity, intensity + 20, intensity + 40]
        
        # Ajouter du bruit
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Simuler quelques obstacles (rectangles)
        t = time.time()
        
        # Obstacle statique
        cv2.rectangle(frame, (200, 200), (350, 400), (80, 80, 80), -1)
        cv2.rectangle(frame, (200, 200), (350, 400), (60, 60, 60), 3)
        
        # Obstacle mobile (bouge avec le temps)
        x_offset = int(100 * np.sin(t))
        cv2.circle(frame, (500 + x_offset, 350), 50, (100, 70, 70), -1)
        
        # Lignes au sol (pour la détection de features)
        for i in range(5):
            y = 500 + i * 40
            cv2.line(frame, (0, y), (self.frame_size[0], y - 100), (70, 70, 70), 2)
        
        # Points de feature simulés
        for _ in range(50):
            x = np.random.randint(0, self.frame_size[0])
            y = np.random.randint(0, self.frame_size[1])
            cv2.circle(frame, (x, y), 2, (150, 150, 150), -1)
        
        # Texte informatif
        cv2.putText(frame, "SIMULATION MODE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return frame
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Retourne la frame courante"""
        return self.current_frame
    
    def get_recent_frames(self, n: int = 10) -> List[Tuple[np.ndarray, float]]:
        """Retourne les n dernières frames avec timestamps"""
        return list(self.frame_buffer)[-n:]


class ObstacleDetector:
    """
    Détecteur d'obstacles par vision
    Utilise plusieurs techniques de détection
    """
    
    def __init__(self):
        """Initialise le détecteur d'obstacles"""
        # Paramètres de détection
        self.min_obstacle_area = 1000  # Aire minimale en pixels
        self.max_obstacle_area = 500000
        self.edge_threshold = 50
        
        # Détecteur de contours
        self.blur_kernel = (5, 5)
        self.canny_low = 50
        self.canny_high = 150
        
        # Détection de profondeur par flou (focus)
        self.laplacian_threshold = 100
        
        # Historique des détections
        self.detection_history: deque = deque(maxlen=30)
        
        # Paramètres de calibration (estimation de distance)
        # Basés sur la caméra du Tello (FOV ~82.6°)
        self.focal_length = 700  # pixels (approximatif)
        self.known_width = 50  # cm (largeur de référence)
        
        logger.info("Détecteur d'obstacles initialisé")
    
    def detect(self, frame: np.ndarray) -> List[VisualObstacle]:
        """
        Détecte les obstacles dans une frame
        
        Args:
            frame: Image BGR
        
        Returns:
            Liste d'obstacles détectés
        """
        if frame is None:
            return []
        
        obstacles = []
        
        # 1. Détection par contours
        contour_obstacles = self._detect_by_contours(frame)
        obstacles.extend(contour_obstacles)
        
        # 2. Détection par différence de profondeur (flou)
        depth_obstacles = self._detect_by_depth_blur(frame)
        obstacles.extend(depth_obstacles)
        
        # 3. Détection de zones dangereuses (proches)
        danger_zones = self._detect_danger_zones(frame)
        obstacles.extend(danger_zones)
        
        # Fusion et filtrage des détections
        obstacles = self._merge_detections(obstacles)
        
        # Mise à jour de l'historique
        self.detection_history.append(obstacles)
        
        return obstacles
    
    def _detect_by_contours(self, frame: np.ndarray) -> List[VisualObstacle]:
        """Détection d'obstacles par analyse de contours"""
        obstacles = []
        
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # Détection de bords
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Dilatation pour connecter les bords
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Trouver les contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_obstacle_area < area < self.max_obstacle_area:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                
                # Estimation de la distance basée sur la taille
                distance = self._estimate_distance(w, h)
                
                # Classification basique
                obs_type = self._classify_obstacle(frame, (x, y, w, h))
                
                obstacle = VisualObstacle(
                    bbox=(x, y, w, h),
                    center=center,
                    area=area,
                    distance_estimate=distance,
                    obstacle_type=obs_type,
                    confidence=min(1.0, area / 10000)
                )
                obstacles.append(obstacle)
        
        return obstacles
    
    def _detect_by_depth_blur(self, frame: np.ndarray) -> List[VisualObstacle]:
        """
        Détection basée sur le flou (objets proches = plus nets)
        Technique de depth-from-focus simplifiée
        """
        obstacles = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Diviser l'image en grille
        grid_size = 4
        h, w = gray.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Extraire la cellule
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                
                # Calculer la variance du Laplacien (mesure de netteté)
                laplacian = cv2.Laplacian(cell, cv2.CV_64F)
                variance = laplacian.var()
                
                # Forte variance = objet proche et net
                if variance > self.laplacian_threshold:
                    # C'est peut-être un obstacle proche
                    distance = max(20, 200 - variance / 10)
                    
                    obstacle = VisualObstacle(
                        bbox=(x1, y1, cell_w, cell_h),
                        center=(x1 + cell_w // 2, y1 + cell_h // 2),
                        area=cell_w * cell_h,
                        distance_estimate=distance,
                        obstacle_type=ObstacleType.UNKNOWN,
                        confidence=min(1.0, variance / 500)
                    )
                    obstacles.append(obstacle)
        
        return obstacles
    
    def _detect_danger_zones(self, frame: np.ndarray) -> List[VisualObstacle]:
        """
        Détecte les zones de danger immédiat
        (grandes zones sombres proches, obstacles au centre)
        """
        obstacles = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Zone centrale (devant le drone)
        center_region = gray[h//3:2*h//3, w//4:3*w//4]
        
        # Seuillage pour détecter les zones sombres (obstacles proches)
        _, thresh = cv2.threshold(center_region, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Pourcentage de pixels sombres
        dark_ratio = np.sum(thresh > 0) / thresh.size
        
        if dark_ratio > 0.3:  # Plus de 30% de zone sombre
            # Obstacle proche au centre
            obstacle = VisualObstacle(
                bbox=(w//4, h//3, w//2, h//3),
                center=(w//2, h//2),
                area=int(w//2 * h//3),
                distance_estimate=max(30, 100 * (1 - dark_ratio)),
                obstacle_type=ObstacleType.WALL,
                confidence=dark_ratio
            )
            obstacles.append(obstacle)
        
        # Vérifier le sol (partie basse de l'image)
        floor_region = gray[2*h//3:, :]
        floor_variance = np.var(floor_region)
        
        if floor_variance < 500:  # Sol uniforme = proche
            obstacle = VisualObstacle(
                bbox=(0, 2*h//3, w, h//3),
                center=(w//2, 5*h//6),
                area=w * h//3,
                distance_estimate=50,
                obstacle_type=ObstacleType.FLOOR,
                confidence=0.5
            )
            obstacles.append(obstacle)
        
        return obstacles
    
    def _estimate_distance(self, width: int, height: int) -> float:
        """
        Estime la distance d'un obstacle basé sur sa taille apparente
        Utilise le modèle de caméra pinhole simplifié
        """
        # Plus l'objet est grand, plus il est proche
        apparent_size = max(width, height)
        
        # Distance approximative (calibration empirique)
        if apparent_size > 0:
            distance = (self.known_width * self.focal_length) / apparent_size
            return max(20, min(500, distance))
        
        return 500
    
    def _classify_obstacle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> ObstacleType:
        """Classifie le type d'obstacle basé sur son apparence"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return ObstacleType.UNKNOWN
        
        # Analyse de couleur
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_color = np.mean(hsv, axis=(0, 1))
        
        # Analyse de forme
        aspect_ratio = w / h if h > 0 else 1
        
        # Classification basique
        if aspect_ratio > 2:  # Large et horizontal
            return ObstacleType.WALL
        elif aspect_ratio < 0.5:  # Haut et étroit
            return ObstacleType.PERSON
        elif y < frame.shape[0] // 4:  # En haut de l'image
            return ObstacleType.CEILING
        elif y > 2 * frame.shape[0] // 3:  # En bas
            return ObstacleType.FLOOR
        
        return ObstacleType.OBJECT
    
    def _merge_detections(self, obstacles: List[VisualObstacle]) -> List[VisualObstacle]:
        """Fusionne les détections qui se chevauchent"""
        if len(obstacles) <= 1:
            return obstacles
        
        merged = []
        used = set()
        
        for i, obs1 in enumerate(obstacles):
            if i in used:
                continue
            
            # Chercher les détections qui se chevauchent
            overlapping = [obs1]
            
            for j, obs2 in enumerate(obstacles[i+1:], start=i+1):
                if j in used:
                    continue
                
                if self._boxes_overlap(obs1.bbox, obs2.bbox):
                    overlapping.append(obs2)
                    used.add(j)
            
            # Fusionner les détections
            if len(overlapping) > 1:
                merged_obs = self._merge_obstacles(overlapping)
                merged.append(merged_obs)
            else:
                merged.append(obs1)
            
            used.add(i)
        
        return merged
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple, threshold: float = 0.3) -> bool:
        """Vérifie si deux boîtes se chevauchent"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calcul de l'intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union > threshold if union > 0 else False
    
    def _merge_obstacles(self, obstacles: List[VisualObstacle]) -> VisualObstacle:
        """Fusionne plusieurs obstacles en un seul"""
        # Calculer la boîte englobante
        x_min = min(o.bbox[0] for o in obstacles)
        y_min = min(o.bbox[1] for o in obstacles)
        x_max = max(o.bbox[0] + o.bbox[2] for o in obstacles)
        y_max = max(o.bbox[1] + o.bbox[3] for o in obstacles)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # Moyenne pondérée de la distance
        total_conf = sum(o.confidence for o in obstacles)
        if total_conf > 0:
            avg_distance = sum(o.distance_estimate * o.confidence for o in obstacles) / total_conf
        else:
            avg_distance = sum(o.distance_estimate for o in obstacles) / len(obstacles)
        
        # Type le plus confiant
        best_type = max(obstacles, key=lambda o: o.confidence).obstacle_type
        
        return VisualObstacle(
            bbox=(x_min, y_min, w, h),
            center=(x_min + w // 2, y_min + h // 2),
            area=w * h,
            distance_estimate=avg_distance,
            obstacle_type=best_type,
            confidence=min(1.0, total_conf / len(obstacles))
        )
    
    def draw_detections(self, frame: np.ndarray, obstacles: List[VisualObstacle]) -> np.ndarray:
        """
        Dessine les détections sur la frame
        
        Args:
            frame: Image source
            obstacles: Liste d'obstacles détectés
        
        Returns:
            Image avec les détections dessinées
        """
        result = frame.copy()
        
        for obs in obstacles:
            x, y, w, h = obs.bbox
            
            # Couleur basée sur la distance
            if obs.distance_estimate < 50:
                color = (0, 0, 255)  # Rouge - danger
            elif obs.distance_estimate < 100:
                color = (0, 165, 255)  # Orange - attention
            else:
                color = (0, 255, 0)  # Vert - ok
            
            # Dessiner la boîte
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Informations
            label = f"{obs.obstacle_type.value}: {obs.distance_estimate:.0f}cm"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Point central
            cv2.circle(result, obs.center, 5, color, -1)
        
        # Compteur d'obstacles
        cv2.putText(result, f"Obstacles: {len(obstacles)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def get_closest_obstacle(self, obstacles: List[VisualObstacle]) -> Optional[VisualObstacle]:
        """Retourne l'obstacle le plus proche"""
        if not obstacles:
            return None
        return min(obstacles, key=lambda o: o.distance_estimate)
    
    def is_path_clear(self, obstacles: List[VisualObstacle], 
                      min_distance: float = 80) -> Tuple[bool, Optional[VisualObstacle]]:
        """
        Vérifie si le chemin devant est dégagé
        
        Returns:
            (chemin_libre, obstacle_bloquant)
        """
        for obs in obstacles:
            if obs.distance_estimate < min_distance:
                # Vérifier si l'obstacle est au centre (devant)
                frame_center_x = 480  # Moitié de 960
                if abs(obs.center[0] - frame_center_x) < 200:
                    return False, obs
        
        return True, None


class FeatureExtractor:
    """
    Extracteur de features pour le SLAM visuel
    Utilise ORB (Oriented FAST and Rotated BRIEF)
    """
    
    def __init__(self, max_features: int = 1000):
        """
        Initialise l'extracteur de features
        
        Args:
            max_features: Nombre maximum de features à extraire
        """
        self.max_features = max_features
        
        # Créer le détecteur ORB
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )
        
        # Matcher pour comparer les features
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Historique des features
        self.frame_id = 0
        self.feature_history: deque = deque(maxlen=100)
        
        logger.info(f"Extracteur de features initialisé (max: {max_features})")
    
    def extract(self, frame: np.ndarray) -> FrameFeatures:
        """
        Extrait les features d'une frame
        
        Args:
            frame: Image BGR
        
        Returns:
            FrameFeatures avec keypoints et descripteurs
        """
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste
        gray = cv2.equalizeHist(gray)
        
        # Détection et description des features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Créer l'objet FrameFeatures
        features = FrameFeatures(
            keypoints=list(keypoints) if keypoints else [],
            descriptors=descriptors if descriptors is not None else np.array([]),
            timestamp=time.time(),
            frame_id=self.frame_id
        )
        
        self.frame_id += 1
        self.feature_history.append(features)
        
        return features
    
    def match_features(self, features1: FrameFeatures, 
                       features2: FrameFeatures) -> List[cv2.DMatch]:
        """
        Match les features entre deux frames
        
        Returns:
            Liste des correspondances
        """
        if features1.descriptors is None or features2.descriptors is None:
            return []
        
        if len(features1.descriptors) == 0 or len(features2.descriptors) == 0:
            return []
        
        try:
            matches = self.bf_matcher.match(features1.descriptors, features2.descriptors)
            # Trier par distance
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        except Exception as e:
            logger.error(f"Erreur matching: {e}")
            return []
    
    def draw_features(self, frame: np.ndarray, features: FrameFeatures) -> np.ndarray:
        """Dessine les features sur la frame"""
        result = frame.copy()
        
        for kp in features.keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
        
        cv2.putText(result, f"Features: {len(features.keypoints)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result
    
    def draw_matches(self, frame1: np.ndarray, features1: FrameFeatures,
                     frame2: np.ndarray, features2: FrameFeatures,
                     matches: List[cv2.DMatch], max_matches: int = 50) -> np.ndarray:
        """Dessine les correspondances entre deux frames"""
        # Limiter le nombre de matches affichés
        matches = matches[:max_matches]
        
        result = cv2.drawMatches(
            frame1, features1.keypoints,
            frame2, features2.keypoints,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result


# Test du module
if __name__ == "__main__":
    print("=== Test du module de vision ===\n")
    
    # Créer les composants
    video_stream = VideoStream(simulation_mode=True)
    obstacle_detector = ObstacleDetector()
    feature_extractor = FeatureExtractor()
    
    # Démarrer le flux
    print("1. Démarrage du flux vidéo...")
    video_stream.start()
    time.sleep(1)
    
    # Capturer quelques frames
    print("\n2. Capture et analyse de frames...")
    prev_features = None
    
    for i in range(5):
        frame = video_stream.get_frame()
        
        if frame is not None:
            print(f"\n   Frame {i+1}:")
            print(f"   - Taille: {frame.shape}")
            
            # Détection d'obstacles
            obstacles = obstacle_detector.detect(frame)
            print(f"   - Obstacles détectés: {len(obstacles)}")
            
            for obs in obstacles[:3]:  # Afficher les 3 premiers
                print(f"     * {obs.obstacle_type.value}: {obs.distance_estimate:.0f}cm")
            
            # Extraction de features
            features = feature_extractor.extract(frame)
            print(f"   - Features: {len(features.keypoints)}")
            
            # Matching avec la frame précédente
            if prev_features is not None:
                matches = feature_extractor.match_features(prev_features, features)
                print(f"   - Matches avec frame précédente: {len(matches)}")
            
            prev_features = features
        
        time.sleep(0.5)
    
    # Arrêter le flux
    print("\n3. Arrêt du flux vidéo...")
    video_stream.stop()
    
    print("\n✓ Test terminé")
