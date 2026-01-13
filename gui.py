#!/usr/bin/env python3
"""
Interface graphique pour le système d'exploration Tello EDU
Affiche le flux vidéo, la carte SLAM, et les contrôles
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import logging
from typing import Optional, Callable
import queue

# Import des modules du projet
from tello_controller import TelloController
from exploration import ExplorationMission, MissionConfig, MissionStatus
from vision import VideoStream, ObstacleDetector, FeatureExtractor
from visual_slam import VisualSLAM, CameraPose

logger = logging.getLogger(__name__)


class VideoPanel(ttk.Frame):
    """Panneau d'affichage vidéo avec overlays"""
    
    def __init__(self, parent, width=640, height=480):
        super().__init__(parent)
        
        self.width = width
        self.height = height
        
        # Canvas pour la vidéo
        self.canvas = tk.Canvas(self, width=width, height=height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image placeholder
        self.photo = None
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        
        # Overlays
        self.show_features = tk.BooleanVar(value=True)
        self.show_obstacles = tk.BooleanVar(value=True)
        self.show_info = tk.BooleanVar(value=True)
        
        # Statistiques
        self.fps = 0
        self.frame_count = 0
        self.obstacles_count = 0
        
    def update_frame(self, frame: np.ndarray, obstacles: list = None, 
                     features=None, info: dict = None):
        """Met à jour l'affichage avec une nouvelle frame"""
        if frame is None:
            return
        
        # Copie pour modification
        display_frame = frame.copy()
        
        # Redimensionner si nécessaire
        h, w = display_frame.shape[:2]
        if w != self.width or h != self.height:
            display_frame = cv2.resize(display_frame, (self.width, self.height))
        
        # Dessiner les features
        if self.show_features.get() and features is not None:
            for kp in features.keypoints[:200]:
                x, y = int(kp.pt[0] * self.width / w), int(kp.pt[1] * self.height / h)
                cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)
        
        # Dessiner les obstacles
        if self.show_obstacles.get() and obstacles:
            self.obstacles_count = len(obstacles)
            for obs in obstacles:
                # Adapter les coordonnées
                x = int(obs.bbox[0] * self.width / w)
                y = int(obs.bbox[1] * self.height / h)
                bw = int(obs.bbox[2] * self.width / w)
                bh = int(obs.bbox[3] * self.height / h)
                
                # Couleur selon la distance
                if obs.distance_estimate < 50:
                    color = (0, 0, 255)  # Rouge
                elif obs.distance_estimate < 100:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Vert
                
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), color, 2)
                label = f"{obs.distance_estimate:.0f}cm"
                cv2.putText(display_frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Dessiner les informations
        if self.show_info.get() and info:
            y_offset = 20
            for key, value in info.items():
                text = f"{key}: {value}"
                cv2.putText(display_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        # Convertir BGR -> RGB
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Convertir en image Tkinter
        img = Image.fromarray(display_frame)
        self.photo = ImageTk.PhotoImage(image=img)
        
        # Mettre à jour le canvas
        self.canvas.itemconfig(self.image_id, image=self.photo)
        self.frame_count += 1


class MapPanel(ttk.Frame):
    """Panneau d'affichage de la carte SLAM"""
    
    def __init__(self, parent, width=400, height=400):
        super().__init__(parent)
        
        self.width = width
        self.height = height
        
        # Canvas pour la carte
        self.canvas = tk.Canvas(self, width=width, height=height, bg='#1a1a2e')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image de la carte
        self.photo = None
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        
        # Échelle et offset pour zoom/pan
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Bindings pour interaction
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        
        self.last_x = 0
        self.last_y = 0
    
    def _on_mousewheel(self, event):
        """Gestion du zoom"""
        if event.num == 4 or event.delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        self.scale = max(0.5, min(3.0, self.scale))
    
    def _on_click(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def _on_drag(self, event):
        """Gestion du déplacement"""
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.offset_x += dx
        self.offset_y += dy
        self.last_x = event.x
        self.last_y = event.y
    
    def update_map(self, slam_map: np.ndarray = None, 
                   trajectory: list = None,
                   landmarks: dict = None,
                   drone_pose: CameraPose = None):
        """Met à jour l'affichage de la carte"""
        # Créer une image de la carte
        if slam_map is not None:
            # Redimensionner
            map_img = cv2.resize(slam_map, (self.width, self.height))
        else:
            # Carte vide
            map_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            map_img[:] = (30, 30, 46)  # Fond sombre
            
            # Grille de référence
            for i in range(0, self.width, 40):
                cv2.line(map_img, (i, 0), (i, self.height), (50, 50, 70), 1)
            for i in range(0, self.height, 40):
                cv2.line(map_img, (0, i), (self.width, i), (50, 50, 70), 1)
            
            # Dessiner la trajectoire
            if trajectory and len(trajectory) > 1:
                center_x, center_y = self.width // 2, self.height // 2
                scale = 2  # pixels par cm
                
                points = []
                for pos in trajectory[-500:]:  # Limiter le nombre de points
                    px = int(center_x + pos[0] * scale * self.scale + self.offset_x)
                    py = int(center_y - pos[1] * scale * self.scale + self.offset_y)
                    points.append((px, py))
                
                for i in range(1, len(points)):
                    cv2.line(map_img, points[i-1], points[i], (255, 255, 0), 2)
            
            # Dessiner les landmarks
            if landmarks:
                center_x, center_y = self.width // 2, self.height // 2
                scale = 2
                
                for lid, lm in list(landmarks.items())[:500]:
                    lx = int(center_x + lm.position[0] * scale * self.scale + self.offset_x)
                    ly = int(center_y - lm.position[1] * scale * self.scale + self.offset_y)
                    
                    if 0 <= lx < self.width and 0 <= ly < self.height:
                        intensity = int(lm.confidence * 200) + 55
                        cv2.circle(map_img, (lx, ly), 2, (0, intensity, 0), -1)
            
            # Dessiner le drone
            if drone_pose:
                center_x, center_y = self.width // 2, self.height // 2
                scale = 2
                
                dx = int(center_x + drone_pose.translation[0] * scale * self.scale + self.offset_x)
                dy = int(center_y - drone_pose.translation[1] * scale * self.scale + self.offset_y)
                
                # Drone (cercle)
                cv2.circle(map_img, (dx, dy), 8, (0, 255, 255), -1)
                
                # Flèche d'orientation (CORRECTION ICI)
                # On récupère le yaw (en radians)
                # Si drone_pose a la propriété yaw_degrees ajoutée dans visual_slam.py :
                angle_deg = getattr(drone_pose, 'yaw_degrees', 0)
                # Sinon calcul manuel : atan2(R[1,0], R[0,0])
                if not hasattr(drone_pose, 'yaw_degrees'):
                    R = drone_pose.rotation
                    angle_rad = np.arctan2(R[1, 0], R[0, 0])
                else:
                    angle_rad = np.radians(angle_deg)

                arrow_len = 20
                # Projection: X est horizontal, Y est vertical (inversé sur image)
                # Attention aux conventions d'axe de la carte (Y vers le bas en image)
                ax = int(dx + arrow_len * np.cos(angle_rad))
                ay = int(dy + arrow_len * np.sin(angle_rad)) # +sin car Y image augmente vers le bas
                
                cv2.arrowedLine(map_img, (dx, dy), (ax, ay), (0, 0, 255), 2, tipLength=0.3)
        
        # Légende
        cv2.putText(map_img, "SLAM Map", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Convertir et afficher
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(map_img)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.image_id, image=self.photo)


class ControlPanel(ttk.Frame):
    """Panneau de contrôle du drone"""
    
    def __init__(self, parent, on_command: Callable = None):
        super().__init__(parent)
        
        self.on_command = on_command
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Style
        style = ttk.Style()
        style.configure('Control.TButton', padding=5)
        style.configure('Emergency.TButton', foreground='red')
        
        # Frame principale
        main_frame = ttk.LabelFrame(self, text="Contrôles", padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Boutons de vol
        flight_frame = ttk.LabelFrame(main_frame, text="Vol", padding=5)
        flight_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(flight_frame, text="Décollage", 
                  command=lambda: self._send("takeoff")).pack(side=tk.LEFT, padx=2)
        ttk.Button(flight_frame, text="Atterrissage",
                  command=lambda: self._send("land")).pack(side=tk.LEFT, padx=2)
        ttk.Button(flight_frame, text="URGENCE", style='Emergency.TButton',
                  command=lambda: self._send("emergency")).pack(side=tk.RIGHT, padx=2)
        
        # Contrôles directionnels
        dir_frame = ttk.LabelFrame(main_frame, text="Direction", padding=5)
        dir_frame.pack(fill=tk.X, pady=5)
        
        # Grille de boutons
        btn_grid = ttk.Frame(dir_frame)
        btn_grid.pack()
        
        ttk.Button(btn_grid, text="↑", width=5,
                  command=lambda: self._send("forward")).grid(row=0, column=1, pady=2)
        ttk.Button(btn_grid, text="←", width=5,
                  command=lambda: self._send("left")).grid(row=1, column=0, padx=2)
        ttk.Button(btn_grid, text="●", width=5,
                  command=lambda: self._send("stop")).grid(row=1, column=1)
        ttk.Button(btn_grid, text="→", width=5,
                  command=lambda: self._send("right")).grid(row=1, column=2, padx=2)
        ttk.Button(btn_grid, text="↓", width=5,
                  command=lambda: self._send("back")).grid(row=2, column=1, pady=2)
        
        # Altitude
        alt_frame = ttk.Frame(dir_frame)
        alt_frame.pack(pady=5)
        
        ttk.Button(alt_frame, text="▲ Monter", width=10,
                  command=lambda: self._send("up")).pack(side=tk.LEFT, padx=5)
        ttk.Button(alt_frame, text="▼ Descendre", width=10,
                  command=lambda: self._send("down")).pack(side=tk.LEFT, padx=5)
        
        # Rotation
        rot_frame = ttk.Frame(dir_frame)
        rot_frame.pack(pady=5)
        
        ttk.Button(rot_frame, text="↺ Rotation G", width=10,
                  command=lambda: self._send("rotate_left")).pack(side=tk.LEFT, padx=5)
        ttk.Button(rot_frame, text="↻ Rotation D", width=10,
                  command=lambda: self._send("rotate_right")).pack(side=tk.LEFT, padx=5)
        
        # Distance
        dist_frame = ttk.LabelFrame(main_frame, text="Distance (cm)", padding=5)
        dist_frame.pack(fill=tk.X, pady=5)
        
        self.distance_var = tk.IntVar(value=50)
        ttk.Scale(dist_frame, from_=20, to=200, variable=self.distance_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(dist_frame, textvariable=self.distance_var).pack()
        
        # Exploration
        exp_frame = ttk.LabelFrame(main_frame, text="Exploration", padding=5)
        exp_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(exp_frame, text="▶ Démarrer",
                  command=lambda: self._send("start_exploration")).pack(side=tk.LEFT, padx=2)
        ttk.Button(exp_frame, text="⏸ Pause",
                  command=lambda: self._send("pause_exploration")).pack(side=tk.LEFT, padx=2)
        ttk.Button(exp_frame, text="⏹ Stop",
                  command=lambda: self._send("stop_exploration")).pack(side=tk.LEFT, padx=2)
        ttk.Button(exp_frame, text="🏠 Retour",
                  command=lambda: self._send("return_home")).pack(side=tk.LEFT, padx=2)
    
    def _send(self, command: str):
        """Envoie une commande"""
        if self.on_command:
            self.on_command(command, self.distance_var.get())


class StatusPanel(ttk.Frame):
    """Panneau d'état et télémétrie"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Frame principale
        main_frame = ttk.LabelFrame(self, text="État", padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Indicateurs
        self.battery_var = tk.StringVar(value="Batterie: ---%")
        self.altitude_var = tk.StringVar(value="Altitude: --- cm")
        self.position_var = tk.StringVar(value="Position: (---, ---, ---)")
        self.state_var = tk.StringVar(value="État: Déconnecté")
        self.mission_var = tk.StringVar(value="Mission: Inactive")
        self.fps_var = tk.StringVar(value="FPS: ---")
        self.landmarks_var = tk.StringVar(value="Landmarks: ---")
        self.obstacles_var = tk.StringVar(value="Obstacles: ---")
        
        # Labels
        for var in [self.battery_var, self.altitude_var, self.position_var,
                   self.state_var, self.mission_var, self.fps_var,
                   self.landmarks_var, self.obstacles_var]:
            ttk.Label(main_frame, textvariable=var).pack(anchor=tk.W, pady=2)
        
        # Barre de progression
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        ttk.Label(main_frame, text="Progression:").pack(anchor=tk.W)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Couverture carte
        ttk.Label(main_frame, text="Couverture:").pack(anchor=tk.W)
        self.coverage_var = tk.DoubleVar(value=0)
        self.coverage_bar = ttk.Progressbar(main_frame, variable=self.coverage_var,
                                            maximum=100, mode='determinate')
        self.coverage_bar.pack(fill=tk.X, pady=5)
    
    def update_status(self, telemetry: dict = None, mission_status: str = None,
                      progress: float = None, coverage: float = None,
                      fps: float = None, landmarks: int = None, obstacles: int = None):
        """Met à jour les indicateurs d'état"""
        if telemetry:
            self.battery_var.set(f"Batterie: {telemetry.get('battery', '---')}%")
            self.altitude_var.set(f"Altitude: {telemetry.get('height', '---')} cm")
            pos = telemetry.get('position', (0, 0, 0))
            self.position_var.set(f"Position: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
        
        if mission_status:
            self.mission_var.set(f"Mission: {mission_status}")
        
        if progress is not None:
            self.progress_var.set(progress)
        
        if coverage is not None:
            self.coverage_var.set(coverage)
        
        if fps is not None:
            self.fps_var.set(f"FPS: {fps:.1f}")
        
        if landmarks is not None:
            self.landmarks_var.set(f"Landmarks: {landmarks}")
        
        if obstacles is not None:
            self.obstacles_var.set(f"Obstacles: {obstacles}")


class TelloExplorerGUI:
    """
    Interface graphique principale du système d'exploration Tello EDU
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        Initialise l'interface graphique
        
        Args:
            simulation_mode: Si True, mode simulation sans drone réel
        """
        self.simulation_mode = simulation_mode
        
        # Créer la fenêtre principale
        self.root = tk.Tk()
        self.root.title("Tello EDU Explorer - SLAM & Vision")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0f0f23')
        
        # Variables d'état
        self.is_running = False
        self.update_queue = queue.Queue()
        
        # Composants du système
        self.mission: Optional[ExplorationMission] = None
        self.slam: Optional[VisualSLAM] = None
        self.video_stream: Optional[VideoStream] = None
        self.obstacle_detector: Optional[ObstacleDetector] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        
        # Créer l'interface
        self._create_menu()
        self._create_layout()
        
        # Bindings clavier
        self._setup_keybindings()
        
        # Initialiser les composants
        self._initialize_components()
        
        # Démarrer la boucle de mise à jour
        self._start_update_loop()
        
        logger.info("Interface graphique initialisée")
    
    def _create_menu(self):
        """Crée la barre de menu"""
        menubar = tk.Menu(self.root)
        
        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exporter carte...", command=self._export_map)
        file_menu.add_command(label="Exporter rapport...", command=self._export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self._on_closing)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        
        # Menu Affichage
        view_menu = tk.Menu(menubar, tearoff=0)
        self.show_features_var = tk.BooleanVar(value=True)
        self.show_obstacles_var = tk.BooleanVar(value=True)
        self.show_trajectory_var = tk.BooleanVar(value=True)
        
        view_menu.add_checkbutton(label="Afficher features", variable=self.show_features_var)
        view_menu.add_checkbutton(label="Afficher obstacles", variable=self.show_obstacles_var)
        view_menu.add_checkbutton(label="Afficher trajectoire", variable=self.show_trajectory_var)
        menubar.add_cascade(label="Affichage", menu=view_menu)
        
        # Menu SLAM
        slam_menu = tk.Menu(menubar, tearoff=0)
        slam_menu.add_command(label="Réinitialiser SLAM", command=self._reset_slam)
        slam_menu.add_command(label="Créer keyframe", command=self._create_keyframe)
        menubar.add_cascade(label="SLAM", menu=slam_menu)
        
        # Menu Aide
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Raccourcis clavier", command=self._show_shortcuts)
        help_menu.add_command(label="À propos", command=self._show_about)
        menubar.add_cascade(label="Aide", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _create_layout(self):
        """Crée la disposition de l'interface"""
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panneau gauche (vidéo + contrôles)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Panneau vidéo
        self.video_panel = VideoPanel(left_panel, width=640, height=480)
        self.video_panel.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Panneau de contrôle
        self.control_panel = ControlPanel(left_panel, on_command=self._handle_command)
        self.control_panel.pack(fill=tk.X, pady=5)
        
        # Panneau droit (carte + état)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Panneau carte
        self.map_panel = MapPanel(right_panel, width=400, height=400)
        self.map_panel.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Panneau d'état
        self.status_panel = StatusPanel(right_panel)
        self.status_panel.pack(fill=tk.X, pady=5)
        
        # Barre de statut en bas
        self.status_bar = ttk.Label(self.root, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_keybindings(self):
        """Configure les raccourcis clavier"""
        self.root.bind('<space>', lambda e: self._handle_command('takeoff'))
        self.root.bind('<Escape>', lambda e: self._handle_command('land'))
        self.root.bind('<Return>', lambda e: self._handle_command('emergency'))
        
        self.root.bind('<Up>', lambda e: self._handle_command('forward'))
        self.root.bind('<Down>', lambda e: self._handle_command('back'))
        self.root.bind('<Left>', lambda e: self._handle_command('left'))
        self.root.bind('<Right>', lambda e: self._handle_command('right'))
        
        self.root.bind('w', lambda e: self._handle_command('forward'))
        self.root.bind('s', lambda e: self._handle_command('back'))
        self.root.bind('a', lambda e: self._handle_command('left'))
        self.root.bind('d', lambda e: self._handle_command('right'))
        self.root.bind('q', lambda e: self._handle_command('up'))
        self.root.bind('e', lambda e: self._handle_command('down'))
        self.root.bind('z', lambda e: self._handle_command('rotate_left'))
        self.root.bind('c', lambda e: self._handle_command('rotate_right'))
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _initialize_components(self):
        """Initialise les composants du système"""
        try:
            # Configuration de la mission
            config = MissionConfig(
                area_width=500,
                area_height=500,
                exploration_altitude=100,
                step_size=50
            )
            
            # Créer la mission
            self.mission = ExplorationMission(config, simulation_mode=self.simulation_mode)
            
            # Créer le flux vidéo
            self.video_stream = VideoStream(simulation_mode=self.simulation_mode)
            
            # Créer le SLAM
            self.slam = VisualSLAM(self.video_stream, simulation_mode=self.simulation_mode)
            
            # Créer les détecteurs
            self.obstacle_detector = ObstacleDetector()
            self.feature_extractor = FeatureExtractor()
            
            # Connexion au drone
            self.mission.controller.connect()
            
            # Démarrer le flux vidéo et le SLAM
            self.video_stream.start()
            self.slam.start()
            
            self.is_running = True
            self.status_bar.config(text="Système initialisé - Mode: " + 
                                  ("Simulation" if self.simulation_mode else "Réel"))
            
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            messagebox.showerror("Erreur", f"Erreur d'initialisation: {e}")
    
    def _start_update_loop(self):
        """Démarre la boucle de mise à jour de l'interface"""
        self._update_display()
    
    def _update_display(self):
        """Met à jour l'affichage"""
        if not self.is_running:
            return
        
        try:
            # Obtenir la frame courante
            frame = self.video_stream.get_frame() if self.video_stream else None
            
            if frame is not None:
                # Détecter les obstacles
                obstacles = self.obstacle_detector.detect(frame) if self.obstacle_detector else []
                
                # Extraire les features
                features = self.feature_extractor.extract(frame) if self.feature_extractor else None
                
                # Informations à afficher
                info = {
                    "Mode": "SIM" if self.simulation_mode else "REAL",
                    "FPS": f"{self.video_stream.fps:.1f}" if self.video_stream else "0"
                }
                
                # Mettre à jour le panneau vidéo
                self.video_panel.update_frame(frame, obstacles, features, info)
                self.video_panel.show_features.set(self.show_features_var.get())
                self.video_panel.show_obstacles.set(self.show_obstacles_var.get())
            
            # Mettre à jour la carte
            if self.slam:
                trajectory = self.slam.get_trajectory() if self.show_trajectory_var.get() else None
                self.map_panel.update_map(
                    trajectory=trajectory,
                    landmarks=self.slam.landmarks,
                    drone_pose=self.slam.get_current_pose()
                )
            
            # Mettre à jour l'état
            if self.mission:
                telemetry = self.mission.controller.get_telemetry()
                self.status_panel.update_status(
                    telemetry=telemetry,
                    mission_status=self.mission.status.value,
                    progress=self.mission.planner.get_progress(),
                    coverage=self.mission.altitude_map.get_exploration_coverage(),
                    fps=self.video_stream.fps if self.video_stream else 0,
                    landmarks=len(self.slam.landmarks) if self.slam else 0,
                    obstacles=len(obstacles) if obstacles else 0
                )
                
                self.status_panel.state_var.set(f"État: {self.mission.controller.state.value}")
            
        except Exception as e:
            logger.error(f"Erreur mise à jour: {e}")
        
        # Planifier la prochaine mise à jour
        self.root.after(33, self._update_display)  # ~30 FPS
    
    def _handle_command(self, command: str, distance: int = 50):
        """Gère les commandes de contrôle"""
        if not self.mission:
            return
        
        try:
            controller = self.mission.controller
            
            if command == "takeoff":
                controller.takeoff()
            elif command == "land":
                controller.land()
            elif command == "emergency":
                controller.emergency_stop()
            elif command == "forward":
                controller.move_forward(distance)
            elif command == "back":
                controller.move_back(distance)
            elif command == "left":
                controller.move_left(distance)
            elif command == "right":
                controller.move_right(distance)
            elif command == "up":
                controller.move_up(distance)
            elif command == "down":
                controller.move_down(distance)
            elif command == "rotate_left":
                controller.rotate_counter_clockwise(45)
            elif command == "rotate_right":
                controller.rotate_clockwise(45)
            elif command == "stop":
                pass  # Hover
            elif command == "start_exploration":
                if self.mission.status == MissionStatus.IDLE:
                    self.mission.prepare_mission()
                self.mission.start_exploration()
            elif command == "pause_exploration":
                self.mission.pause_exploration()
            elif command == "stop_exploration":
                self.mission.stop_exploration()
            elif command == "return_home":
                self.mission.return_to_home()
            
            self.status_bar.config(text=f"Commande: {command}")
            
        except Exception as e:
            logger.error(f"Erreur commande {command}: {e}")
            self.status_bar.config(text=f"Erreur: {e}")
    
    def _export_map(self):
        """Exporte la carte SLAM"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("Tous", "*.*")]
        )
        if filepath and self.slam:
            base_path = filepath.rsplit('.', 1)[0]
            self.slam.export_map(base_path)
            messagebox.showinfo("Export", f"Carte exportée vers {base_path}_*")
    
    def _export_report(self):
        """Exporte le rapport de mission"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Tous", "*.*")]
        )
        if filepath and self.mission:
            base_path = filepath.rsplit('.', 1)[0]
            self.mission.export_results(base_path)
            messagebox.showinfo("Export", f"Rapport exporté vers {base_path}_*")
    
    def _reset_slam(self):
        """Réinitialise le SLAM"""
        if self.slam:
            self.slam.reset()
            self.status_bar.config(text="SLAM réinitialisé")
    
    def _create_keyframe(self):
        """Force la création d'une keyframe"""
        if self.slam and self.video_stream:
            frame = self.video_stream.get_frame()
            if frame is not None:
                self.slam._create_keyframe(frame)
                self.status_bar.config(text="Keyframe créée")
    
    def _show_shortcuts(self):
        """Affiche les raccourcis clavier"""
        shortcuts = """
Raccourcis clavier:

Vol:
  Espace     - Décollage
  Échap      - Atterrissage
  Entrée     - URGENCE

Direction:
  W/↑        - Avancer
  S/↓        - Reculer
  A/←        - Gauche
  D/→        - Droite
  Q          - Monter
  E          - Descendre
  Z          - Rotation gauche
  C          - Rotation droite
"""
        messagebox.showinfo("Raccourcis clavier", shortcuts)
    
    def _show_about(self):
        """Affiche les informations sur l'application"""
        about = """
Tello EDU Explorer
Version 2.0

Système d'exploration autonome avec:
- Flux vidéo en temps réel
- Détection d'obstacles par vision
- SLAM visuel
- Cartographie d'altitude
- Évitement d'obstacles mobiles

Conçu pour l'exploration de zones NRBC
"""
        messagebox.showinfo("À propos", about)
    
    def _on_closing(self):
        """Gère la fermeture de l'application"""
        self.is_running = False
        
        # Arrêter les composants
        if self.slam:
            self.slam.stop()
        
        if self.video_stream:
            self.video_stream.stop()
        
        if self.mission:
            if self.mission.status == MissionStatus.IN_PROGRESS:
                self.mission.stop_exploration()
            self.mission.controller.disconnect()
        
        self.root.destroy()
    
    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interface graphique Tello EDU Explorer")
    parser.add_argument('--simulation', '-s', action='store_true',
                       help="Mode simulation")
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Lancer l'interface
    app = TelloExplorerGUI(simulation_mode=args.simulation)
    app.run()


if __name__ == "__main__":
    main()
