#!/usr/bin/env python3
"""
Module SLAM visuel pour drone Tello EDU
Simultaneous Localization and Mapping basé sur les features visuelles
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
import json

from vision import VideoStream, FeatureExtractor, FrameFeatures, ObstacleDetector

logger = logging.getLogger(__name__)


@dataclass
class Landmark:
    """Point de repère 3D dans la carte"""
    id: int
    position: np.ndarray  # Position 3D (x, y, z)
    descriptor: np.ndarray  # Descripteur visuel
    observations: int = 1  # Nombre de fois observé
    last_seen: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    def update(self, new_position: np.ndarray):
        """Met à jour la position avec filtrage"""
        # Moyenne pondérée
        alpha = 0.3
        self.position = alpha * new_position + (1 - alpha) * self.position
        self.observations += 1
        self.last_seen = time.time()
        self.confidence = min(1.0, self.confidence + 0.1)


@dataclass
class KeyFrame:
    """Frame clé pour le SLAM"""
    id: int
    timestamp: float
    pose: np.ndarray  # Matrice de pose 4x4
    features: FrameFeatures
    frame: np.ndarray  # Image originale (redimensionnée)
    landmark_ids: List[int] = field(default_factory=list)


@dataclass
class CameraPose:
    """Pose de la caméra (position et orientation)"""
    translation: np.ndarray  # Vecteur translation (x, y, z)
    rotation: np.ndarray     # Matrice de rotation 3x3
    timestamp: float = field(default_factory=time.time)
    
    @property
    def position(self) -> Tuple[float, float, float]:
        """Position en coordonnées monde"""
        return tuple(self.translation.flatten())
    
    @property
    def pose_matrix(self) -> np.ndarray:
        """Matrice de pose 4x4"""
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = self.translation.flatten()
        return pose
    
    def to_dict(self) -> dict:
        return {
            'translation': self.translation.tolist(),
            'rotation': self.rotation.tolist(),
            'timestamp': self.timestamp
        }


class VisualOdometry:
    """
    Odométrie visuelle basée sur les features
    Estime le mouvement de la caméra entre les frames
    """
    
    def __init__(self, camera_matrix: np.ndarray = None):
        """
        Initialise l'odométrie visuelle
        
        Args:
            camera_matrix: Matrice intrinsèque de la caméra (3x3)
        """
        # Matrice caméra par défaut (Tello)
        if camera_matrix is None:
            # Approximation pour le Tello (960x720, FOV ~82.6°)
            fx = fy = 700  # Distance focale en pixels
            cx, cy = 480, 360  # Point principal
            self.camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            self.camera_matrix = camera_matrix
        
        # Extracteur de features
        self.feature_extractor = FeatureExtractor(max_features=2000)
        
        # Matcher FLANN pour performance
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # État
        self.prev_frame = None
        self.prev_features: Optional[FrameFeatures] = None
        self.current_pose = CameraPose(
            translation=np.zeros(3),
            rotation=np.eye(3)
        )
        
        # Historique des poses
        self.pose_history: deque = deque(maxlen=1000)
        self.trajectory: List[Tuple[float, float, float]] = []
        
        logger.info("Odométrie visuelle initialisée")
    
    def process_frame(self, frame: np.ndarray) -> Optional[CameraPose]:
        """
        Traite une frame et estime le mouvement
        
        Args:
            frame: Image BGR
        
        Returns:
            Nouvelle pose estimée
        """
        # Extraction des features
        features = self.feature_extractor.extract(frame)
        
        if self.prev_features is None or len(self.prev_features.keypoints) == 0:
            self.prev_frame = frame.copy()
            self.prev_features = features
            return self.current_pose
        
        if len(features.keypoints) < 10:
            logger.warning("Pas assez de features détectées")
            return self.current_pose
        
        # Matching des features
        matches = self._match_features(self.prev_features, features)
        
        if len(matches) < 8:
            logger.warning(f"Pas assez de matches: {len(matches)}")
            self.prev_frame = frame.copy()
            self.prev_features = features
            return self.current_pose
        
        # Extraire les points correspondants
        pts1 = np.float32([self.prev_features.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([features.keypoints[m.trainIdx].pt for m in matches])
        
        # Calculer la matrice essentielle
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            self.prev_frame = frame.copy()
            self.prev_features = features
            return self.current_pose
        
        # Récupérer la pose relative
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        
        # Mise à jour de la pose globale
        # Scale factor (approximatif, devrait être calibré)
        scale = 5.0  # cm par unité de mouvement
        
        # Composition des transformations
        self.current_pose.translation = (
            self.current_pose.translation + 
            scale * self.current_pose.rotation @ t.flatten()
        )
        self.current_pose.rotation = R @ self.current_pose.rotation
        self.current_pose.timestamp = time.time()
        
        # Sauvegarder dans l'historique
        self.pose_history.append(CameraPose(
            translation=self.current_pose.translation.copy(),
            rotation=self.current_pose.rotation.copy(),
            timestamp=self.current_pose.timestamp
        ))
        self.trajectory.append(self.current_pose.position)
        
        # Mise à jour pour la prochaine frame
        self.prev_frame = frame.copy()
        self.prev_features = features
        
        return self.current_pose
    
    def _match_features(self, features1: FrameFeatures, 
                        features2: FrameFeatures) -> List[cv2.DMatch]:
        """Match les features avec filtrage par ratio test"""
        if features1.descriptors is None or features2.descriptors is None:
            return []
        
        if len(features1.descriptors) < 2 or len(features2.descriptors) < 2:
            return []
        
        try:
            # KNN matching
            matches = self.flann.knnMatch(
                features1.descriptors, 
                features2.descriptors, 
                k=2
            )
            
            # Ratio test de Lowe
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            return good_matches
            
        except Exception as e:
            logger.error(f"Erreur matching: {e}")
            return []
    
    def reset(self):
        """Remet l'odométrie à zéro"""
        self.current_pose = CameraPose(
            translation=np.zeros(3),
            rotation=np.eye(3)
        )
        self.pose_history.clear()
        self.trajectory.clear()
        self.prev_frame = None
        self.prev_features = None


class VisualSLAM:
    """
    Système SLAM visuel complet
    Combine odométrie visuelle, gestion de landmarks et optimisation
    """
    
    def __init__(self, video_stream: VideoStream = None, simulation_mode: bool = False):
        """
        Initialise le système SLAM
        
        Args:
            video_stream: Source vidéo (optionnel)
            simulation_mode: Mode simulation
        """
        self.simulation_mode = simulation_mode
        
        # Composants
        self.video_stream = video_stream or VideoStream(simulation_mode=simulation_mode)
        self.visual_odometry = VisualOdometry()
        self.obstacle_detector = ObstacleDetector()
        
        # Carte
        self.landmarks: Dict[int, Landmark] = {}
        self.next_landmark_id = 0
        self.keyframes: List[KeyFrame] = []
        
        # Grille d'occupation 2D (vue de dessus)
        self.grid_resolution = 10  # cm par cellule
        self.grid_size = 200  # cellules
        self.occupancy_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        # -1 = inconnu, 0 = libre, 100 = occupé
        
        # État du SLAM
        self.is_running = False
        self.is_initialized = False
        
        # Thread de traitement
        self._slam_thread: Optional[threading.Thread] = None
        self._stop_slam = threading.Event()
        
        # Statistiques
        self.stats = {
            'frames_processed': 0,
            'landmarks_total': 0,
            'keyframes_total': 0,
            'trajectory_length': 0
        }
        
        logger.info("Visual SLAM initialisé")
    
    def start(self):
        """Démarre le système SLAM"""
        if self.is_running:
            return
        
        # Démarrer le flux vidéo si nécessaire
        if not self.video_stream.is_streaming:
            self.video_stream.start()
        
        # Démarrer le thread SLAM
        self._stop_slam.clear()
        self._slam_thread = threading.Thread(target=self._slam_loop, daemon=True)
        self._slam_thread.start()
        
        self.is_running = True
        logger.info("SLAM démarré")
    
    def stop(self):
        """Arrête le système SLAM"""
        self._stop_slam.set()
        
        if self._slam_thread:
            self._slam_thread.join(timeout=2.0)
        
        self.is_running = False
        logger.info("SLAM arrêté")
    
    def _slam_loop(self):
        """Boucle principale du SLAM"""
        frame_count = 0
        keyframe_interval = 10  # Créer une keyframe toutes les N frames
        
        while not self._stop_slam.is_set():
            frame = self.video_stream.get_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Traitement de la frame
            self._process_frame(frame, frame_count)
            
            # Créer une keyframe périodiquement
            if frame_count % keyframe_interval == 0:
                self._create_keyframe(frame)
            
            frame_count += 1
            self.stats['frames_processed'] = frame_count
            
            time.sleep(0.03)  # ~30 Hz
    
    def _process_frame(self, frame: np.ndarray, frame_id: int):
        """Traite une frame pour le SLAM"""
        # Odométrie visuelle
        pose = self.visual_odometry.process_frame(frame)
        
        # Détection d'obstacles
        obstacles = self.obstacle_detector.detect(frame)
        
        # Mise à jour de la grille d'occupation
        self._update_occupancy_grid(pose, obstacles)
        
        # Mise à jour des landmarks
        if len(self.visual_odometry.prev_features.keypoints) > 0:
            self._update_landmarks(pose, self.visual_odometry.prev_features)
    
    def _create_keyframe(self, frame: np.ndarray):
        """Crée une nouvelle keyframe"""
        pose = self.visual_odometry.current_pose
        features = self.visual_odometry.prev_features
        
        if features is None:
            return
        
        # Redimensionner la frame pour économiser la mémoire
        small_frame = cv2.resize(frame, (320, 240))
        
        keyframe = KeyFrame(
            id=len(self.keyframes),
            timestamp=time.time(),
            pose=pose.pose_matrix.copy(),
            features=features,
            frame=small_frame
        )
        
        self.keyframes.append(keyframe)
        self.stats['keyframes_total'] = len(self.keyframes)
        
        logger.debug(f"Keyframe {keyframe.id} créée")
    
    def _update_landmarks(self, pose: CameraPose, features: FrameFeatures):
        """Met à jour les landmarks basé sur les features observées"""
        if len(features.keypoints) == 0:
            return
        
        # Pour chaque feature, essayer de la matcher avec un landmark existant
        # ou en créer un nouveau
        
        for i, kp in enumerate(features.keypoints[:100]):  # Limiter pour performance
            # Position 2D dans l'image
            px, py = kp.pt
            
            # Estimation de la position 3D (simplifiée)
            # En réalité, il faudrait de la triangulation
            depth = 100  # Profondeur estimée en cm
            
            # Projection inverse (approximative)
            fx, fy = self.visual_odometry.camera_matrix[0, 0], self.visual_odometry.camera_matrix[1, 1]
            cx, cy = self.visual_odometry.camera_matrix[0, 2], self.visual_odometry.camera_matrix[1, 2]
            
            x = (px - cx) * depth / fx
            y = (py - cy) * depth / fy
            z = depth
            
            # Transformer en coordonnées monde
            local_pos = np.array([x, y, z])
            world_pos = pose.rotation @ local_pos + pose.translation
            
            # Chercher un landmark proche
            matched = False
            for lid, landmark in self.landmarks.items():
                dist = np.linalg.norm(world_pos - landmark.position)
                if dist < 20:  # Seuil de correspondance
                    landmark.update(world_pos)
                    matched = True
                    break
            
            # Créer un nouveau landmark si pas de correspondance
            if not matched and len(self.landmarks) < 5000:
                descriptor = features.descriptors[i] if i < len(features.descriptors) else np.array([])
                landmark = Landmark(
                    id=self.next_landmark_id,
                    position=world_pos,
                    descriptor=descriptor
                )
                self.landmarks[self.next_landmark_id] = landmark
                self.next_landmark_id += 1
        
        self.stats['landmarks_total'] = len(self.landmarks)
    
    def _update_occupancy_grid(self, pose: CameraPose, obstacles: list):
        """Met à jour la grille d'occupation basé sur la pose et les obstacles"""
        # Position du drone dans la grille
        drone_x = int(pose.translation[0] / self.grid_resolution + self.grid_size // 2)
        drone_y = int(pose.translation[1] / self.grid_resolution + self.grid_size // 2)
        
        # Marquer la position du drone comme libre
        if 0 <= drone_x < self.grid_size and 0 <= drone_y < self.grid_size:
            self.occupancy_grid[drone_y, drone_x] = 0
        
        # Marquer les obstacles
        for obs in obstacles:
            if obs.distance_estimate < 200:  # Seulement les obstacles proches
                # Direction de l'obstacle (simplifiée)
                angle = np.arctan2(
                    obs.center[0] - 480,  # Offset du centre de l'image
                    700  # Distance focale
                )
                
                # Position de l'obstacle
                obs_dist = obs.distance_estimate
                obs_x = int((pose.translation[0] + obs_dist * np.sin(angle)) / self.grid_resolution + self.grid_size // 2)
                obs_y = int((pose.translation[1] + obs_dist * np.cos(angle)) / self.grid_resolution + self.grid_size // 2)
                
                if 0 <= obs_x < self.grid_size and 0 <= obs_y < self.grid_size:
                    self.occupancy_grid[obs_y, obs_x] = 100
                    
                    # Marquer les cellules autour comme occupées aussi
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = obs_x + dx, obs_y + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                if self.occupancy_grid[ny, nx] != 100:
                                    self.occupancy_grid[ny, nx] = max(50, self.occupancy_grid[ny, nx])
    
    def get_current_pose(self) -> CameraPose:
        """Retourne la pose actuelle estimée"""
        return self.visual_odometry.current_pose
    
    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Retourne la trajectoire parcourue"""
        return self.visual_odometry.trajectory.copy()
    
    def get_map_visualization(self) -> np.ndarray:
        """
        Génère une visualisation de la carte
        
        Returns:
            Image de la carte avec landmarks et trajectoire
        """
        # Créer une image de la carte
        map_img = np.zeros((self.grid_size * 4, self.grid_size * 4, 3), dtype=np.uint8)
        
        # Dessiner la grille d'occupation
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                val = self.occupancy_grid[y, x]
                px, py = x * 4, y * 4
                
                if val == -1:  # Inconnu
                    color = (50, 50, 50)
                elif val == 0:  # Libre
                    color = (100, 100, 100)
                else:  # Occupé
                    intensity = min(255, val * 2)
                    color = (0, 0, intensity)
                
                cv2.rectangle(map_img, (px, py), (px + 3, py + 3), color, -1)
        
        # Dessiner les landmarks
        for lid, landmark in self.landmarks.items():
            lx = int(landmark.position[0] / self.grid_resolution * 4 + self.grid_size * 2)
            ly = int(landmark.position[1] / self.grid_resolution * 4 + self.grid_size * 2)
            
            if 0 <= lx < map_img.shape[1] and 0 <= ly < map_img.shape[0]:
                # Couleur basée sur la confiance
                intensity = int(landmark.confidence * 255)
                cv2.circle(map_img, (lx, ly), 2, (0, intensity, 0), -1)
        
        # Dessiner la trajectoire
        trajectory = self.get_trajectory()
        if len(trajectory) > 1:
            points = []
            for pos in trajectory:
                px = int(pos[0] / self.grid_resolution * 4 + self.grid_size * 2)
                py = int(pos[1] / self.grid_resolution * 4 + self.grid_size * 2)
                points.append((px, py))
            
            for i in range(1, len(points)):
                cv2.line(map_img, points[i-1], points[i], (255, 255, 0), 1)
        
        # Dessiner la position actuelle
        pose = self.get_current_pose()
        drone_x = int(pose.translation[0] / self.grid_resolution * 4 + self.grid_size * 2)
        drone_y = int(pose.translation[1] / self.grid_resolution * 4 + self.grid_size * 2)
        cv2.circle(map_img, (drone_x, drone_y), 8, (0, 255, 255), -1)
        cv2.circle(map_img, (drone_x, drone_y), 8, (0, 200, 200), 2)
        
        # Ajouter les informations
        cv2.putText(map_img, f"Landmarks: {len(self.landmarks)}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(map_img, f"Keyframes: {len(self.keyframes)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(map_img, f"Position: ({pose.translation[0]:.0f}, {pose.translation[1]:.0f})", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return map_img
    
    def get_occupancy_grid_ascii(self) -> str:
        """Retourne une représentation ASCII de la grille d'occupation"""
        lines = []
        lines.append("=" * 52)
        lines.append("CARTE SLAM (vue de dessus)")
        lines.append("=" * 52)
        
        # Sous-échantillonner pour l'affichage
        display_size = 50
        scale = self.grid_size // display_size
        
        pose = self.get_current_pose()
        drone_gx = int(pose.translation[0] / self.grid_resolution + self.grid_size // 2) // scale
        drone_gy = int(pose.translation[1] / self.grid_resolution + self.grid_size // 2) // scale
        
        for y in range(display_size):
            row = ""
            for x in range(display_size):
                # Position du drone
                if x == drone_gx and y == drone_gy:
                    row += "D"
                    continue
                
                # Valeur de la cellule
                cell_vals = self.occupancy_grid[y*scale:(y+1)*scale, x*scale:(x+1)*scale]
                avg_val = np.mean(cell_vals)
                
                if avg_val < 0:
                    row += "·"  # Inconnu
                elif avg_val < 20:
                    row += " "  # Libre
                elif avg_val < 60:
                    row += "░"  # Partiellement occupé
                else:
                    row += "█"  # Occupé
            
            lines.append("|" + row + "|")
        
        lines.append("=" * 52)
        lines.append(f"Position: ({pose.translation[0]:.0f}, {pose.translation[1]:.0f}, {pose.translation[2]:.0f})")
        lines.append(f"Landmarks: {len(self.landmarks)} | Keyframes: {len(self.keyframes)}")
        lines.append("Légende: · inconnu | espace libre | ░ partiel | █ occupé | D drone")
        
        return "\n".join(lines)
    
    def export_map(self, filepath: str):
        """
        Exporte la carte SLAM
        
        Args:
            filepath: Chemin de base (sans extension)
        """
        # Export JSON des données
        data = {
            'stats': self.stats,
            'landmarks': [
                {
                    'id': l.id,
                    'position': l.position.tolist(),
                    'observations': l.observations,
                    'confidence': l.confidence
                }
                for l in self.landmarks.values()
            ],
            'trajectory': self.get_trajectory(),
            'keyframes_count': len(self.keyframes),
            'grid_resolution': self.grid_resolution,
            'grid_size': self.grid_size
        }
        
        with open(f"{filepath}_slam.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        # Export de la grille d'occupation
        np.save(f"{filepath}_occupancy.npy", self.occupancy_grid)
        
        # Export de la visualisation
        map_img = self.get_map_visualization()
        cv2.imwrite(f"{filepath}_map.png", map_img)
        
        logger.info(f"Carte SLAM exportée vers {filepath}_*")
    
    def reset(self):
        """Remet le SLAM à zéro"""
        self.visual_odometry.reset()
        self.landmarks.clear()
        self.keyframes.clear()
        self.next_landmark_id = 0
        self.occupancy_grid.fill(-1)
        self.stats = {
            'frames_processed': 0,
            'landmarks_total': 0,
            'keyframes_total': 0,
            'trajectory_length': 0
        }
        self.is_initialized = False
        logger.info("SLAM réinitialisé")


# Test du module
if __name__ == "__main__":
    print("=== Test du module SLAM visuel ===\n")
    
    # Créer le système SLAM
    slam = VisualSLAM(simulation_mode=True)
    
    # Démarrer
    print("1. Démarrage du SLAM...")
    slam.start()
    
    # Laisser tourner quelques secondes
    print("2. Acquisition de données (5 secondes)...")
    time.sleep(5)
    
    # Afficher les statistiques
    print("\n3. Statistiques:")
    for key, value in slam.stats.items():
        print(f"   {key}: {value}")
    
    # Afficher la carte ASCII
    print("\n4. Carte SLAM:")
    print(slam.get_occupancy_grid_ascii())
    
    # Arrêter
    print("\n5. Arrêt du SLAM...")
    slam.stop()
    
    print("\n✓ Test terminé")
