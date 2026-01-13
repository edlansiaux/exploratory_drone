# 🚁 Système d'Exploration Tello EDU pour Zones NRBC

Système complet de contrôle et d'exploration autonome pour drone DJI Tello EDU, conçu pour l'exploration de zones potentiellement dangereuses (incidents thermiques, nucléaires, radiologiques, biologiques ou chimiques) avant l'envoi de personnel humain.

## 📋 Fonctionnalités

### 🎮 Contrôle Manuel
- ✅ Décollage / Atterrissage
- ✅ Déplacement gauche/droite (50cm par défaut, configurable)
- ✅ Montée/descente (50cm par défaut, configurable)
- ✅ Avancer/reculer
- ✅ Rotation horaire/anti-horaire
- ✅ Retour automatique au point de départ
- ✅ Arrêt d'urgence

### 🤖 Exploration Autonome
- ✅ Pattern de balayage en serpent (snake)
- ✅ Pattern en spirale
- ✅ Gestion automatique des waypoints
- ✅ Limites de temps et de batterie
- ✅ Pause/reprise de mission

### 🗺️ Cartographie
- ✅ Création de carte d'altitude en temps réel
- ✅ Enregistrement de tous les points explorés
- ✅ Statistiques d'altitude (min, max, moyenne)
- ✅ Visualisation ASCII de la carte
- ✅ Export JSON et CSV

### 🚧 Évitement d'Obstacles
- ✅ Détection d'obstacles statiques
- ✅ Détection d'obstacles mobiles avec prédiction de trajectoire
- ✅ Stratégies d'évitement multiples (contournement, passage au-dessus/dessous)
- ✅ Zone de sécurité configurable
- ✅ Réaction réflexe pour dangers immédiats

### 📹 Vision par Ordinateur (NOUVEAU)
- ✅ Flux vidéo en temps réel depuis la caméra du drone
- ✅ Détection d'obstacles par analyse d'image
- ✅ Extraction de features (ORB)
- ✅ Estimation de distance des obstacles
- ✅ Classification des obstacles (mur, personne, objet, sol, plafond)

### 🧭 SLAM Visuel (NOUVEAU)
- ✅ Odométrie visuelle (estimation du mouvement)
- ✅ Cartographie 3D avec landmarks
- ✅ Grille d'occupation 2D
- ✅ Keyframes pour optimisation
- ✅ Trajectoire temps réel
- ✅ Export de carte SLAM

### 🖥️ Interface Graphique (NOUVEAU)
- ✅ Affichage du flux vidéo en temps réel
- ✅ Visualisation de la carte SLAM
- ✅ Overlay des obstacles détectés
- ✅ Affichage des features
- ✅ Contrôles interactifs
- ✅ Indicateurs de télémétrie
- ✅ Raccourcis clavier

### ☢️ Détection de Dangers (Simulation)
- ✅ Zones thermiques
- ✅ Zones radioactives
- ✅ Zones chimiques
- ✅ Alertes en temps réel

## 🛠️ Installation

### Prérequis
- Python 3.8+
- Drone DJI Tello EDU
- Connexion WiFi au drone

### Installation des dépendances

```bash
cd tello_explorer
pip install -r requirements.txt
```

## 🚀 Utilisation

### Interface Graphique (Recommandé)

```bash
# Mode simulation (sans drone)
python gui.py --simulation

# Mode réel (avec drone connecté)
python gui.py
```

### Interface en Ligne de Commande

```bash
# Mode simulation (sans drone)
python main.py --simulation

# Mode réel (avec drone)
python main.py
```

### Avec initialisation automatique

```bash
python main.py --simulation --auto-init
```

## 📖 Guide des Commandes

### Initialisation
```
init          - Initialise le système
connect       - Connecte au drone
disconnect    - Déconnecte du drone
```

### Contrôle Manuel
```
takeoff       - Décollage
land          - Atterrissage
up [dist]     - Monter (défaut: 50cm)
down [dist]   - Descendre (défaut: 50cm)
left [dist]   - Gauche (défaut: 50cm)
right [dist]  - Droite (défaut: 50cm)
forward [dist]- Avancer (défaut: 50cm)
back [dist]   - Reculer (défaut: 50cm)
rotate left|right [angle] - Rotation (défaut: 90°)
```

### Exploration Automatique
```
config [param] [value] - Voir/modifier la configuration
prepare       - Préparer la mission
start         - Démarrer l'exploration
pause         - Mettre en pause
resume        - Reprendre
stop          - Arrêter et atterrir
home          - Retour au point de départ
```

### Visualisation
```
status        - État du drone
map           - Afficher la carte
report        - Rapport complet
live [on|off] - Affichage en direct
```

### Simulation d'Obstacles
```
obstacle <x> <y> <z>              - Obstacle fixe
mobile <x> <y> <z> <vx> <vy> <vz> - Obstacle mobile
hazard <x> <y> <z> <type> <intensity> - Zone de danger
```

### Urgence
```
emergency     - ARRÊT D'URGENCE IMMÉDIAT
```

### Export
```
export [path] - Exporter les résultats
```

## ⚙️ Configuration

Paramètres modifiables via la commande `config`:

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| width | Largeur de la zone (cm) | 500 |
| height | Hauteur de la zone (cm) | 500 |
| altitude | Altitude d'exploration (cm) | 120 |
| step | Pas d'exploration (cm) | 50 |
| pattern | Pattern (snake/spiral) | snake |
| duration | Durée max (s) | 600 |
| battery | Batterie min (%) | 20 |

Exemple:
```
config width 300
config pattern spiral
config altitude 150
```

## 🖥️ Interface Graphique

L'interface graphique (`gui.py`) offre une vue complète du système:

### Panneau Vidéo (Gauche)
- Flux vidéo en temps réel de la caméra du drone
- Overlay des obstacles détectés (boîtes colorées selon la distance)
- Affichage des features ORB extraites
- Informations FPS et mode

### Panneau Carte SLAM (Droite)
- Carte 2D avec landmarks détectés
- Trajectoire du drone en temps réel
- Grille d'occupation (zones libres/occupées)
- Position et orientation du drone
- Zoom et déplacement avec la souris

### Raccourcis Clavier

| Touche | Action |
|--------|--------|
| Espace | Décollage |
| Échap | Atterrissage |
| Entrée | URGENCE |
| W / ↑ | Avancer |
| S / ↓ | Reculer |
| A / ← | Gauche |
| D / → | Droite |
| Q | Monter |
| E | Descendre |
| Z | Rotation gauche |
| C | Rotation droite |

## 📊 Structure des Fichiers

```
tello_explorer/
├── gui.py               # Interface graphique (vidéo + carte SLAM)
├── main.py              # Interface CLI interactive
├── exploration.py       # Gestionnaire de mission
├── tello_controller.py  # Contrôle bas niveau du drone
├── mapping.py           # Cartographie et planification
├── obstacle_avoidance.py# Système d'évitement
├── vision.py            # Flux vidéo et détection d'obstacles
├── visual_slam.py       # SLAM visuel et odométrie
├── examples.py          # Exemples d'utilisation
├── requirements.txt     # Dépendances Python
└── README.md            # Documentation
```

## 🔧 Architecture du Code

### TelloController
Gère les commandes bas niveau du drone:
- Connexion/déconnexion
- Mouvements de base
- Télémétrie
- Suivi de position estimée

### AltitudeMap
Gère la cartographie:
- Grille d'altitude
- Points d'exploration
- Liste des obstacles
- Export des données

### ExplorationPlanner
Planifie l'exploration:
- Génération de patterns (snake, spiral)
- Gestion des waypoints
- Calcul de la progression

### ObstacleAvoidanceSystem
Gère l'évitement d'obstacles:
- Détection et suivi
- Prédiction de trajectoire (obstacles mobiles)
- Stratégies d'évitement
- Zone de sécurité

### VideoStream (NOUVEAU)
Gère le flux vidéo:
- Capture depuis le drone ou webcam
- Génération de frames synthétiques (simulation)
- Buffer de frames pour traitement

### ObstacleDetector (NOUVEAU)
Détection d'obstacles par vision:
- Analyse de contours
- Estimation de profondeur par flou
- Classification des obstacles
- Estimation de distance

### FeatureExtractor (NOUVEAU)
Extraction de features pour SLAM:
- Détecteur ORB (Oriented FAST and Rotated BRIEF)
- Matching de features entre frames
- Suivi des points d'intérêt

### VisualOdometry (NOUVEAU)
Odométrie visuelle:
- Estimation du mouvement entre frames
- Calcul de la matrice essentielle
- Récupération de la pose (rotation + translation)

### VisualSLAM (NOUVEAU)
SLAM visuel complet:
- Gestion des landmarks 3D
- Création de keyframes
- Grille d'occupation 2D
- Fusion des données visuelles et de navigation

### ExplorationMission
Coordonne tous les composants:
- Gestion de l'état de mission
- Boucle d'exploration
- Callbacks et événements

## 📈 Exemple de Mission

```python
from exploration import ExplorationMission, MissionConfig

# Configuration
config = MissionConfig(
    area_width=400,
    area_height=400,
    exploration_altitude=100,
    step_size=50,
    pattern="snake"
)

# Création de la mission
mission = ExplorationMission(config, simulation_mode=True)

# Préparation
mission.prepare_mission()

# Ajout d'obstacles simulés
mission.add_simulated_obstacle(100, 50, 100, is_mobile=False)
mission.add_simulated_obstacle(0, 100, 100, is_mobile=True, velocity=(10, 5, 0))

# Démarrage
mission.start_exploration()

# ... attendre ...

# Arrêt et export
mission.stop_exploration()
mission.export_results("mission_results")
```

## ⚠️ Sécurité

### Recommandations
1. **Toujours** tester en mode simulation d'abord
2. Maintenir un contact visuel avec le drone
3. Voler dans un espace dégagé
4. Vérifier la batterie avant chaque vol
5. Connaître l'emplacement du bouton d'arrêt d'urgence

### Limites du Tello EDU
- Distance de contrôle: ~100m
- Altitude max: ~30m (limité par défaut)
- Autonomie: ~13 minutes
- Vent max: ~10 m/s

## 🔮 Améliorations Futures

- [x] ~~Intégration flux vidéo (OpenCV)~~
- [x] ~~SLAM visuel pour cartographie plus précise~~
- [x] ~~Détection d'obstacles par vision~~
- [x] ~~Interface graphique (GUI)~~
- [ ] Support de capteurs NRBC réels
- [ ] Communication multi-drones
- [ ] Planification de trajectoire A*
- [ ] Optimisation du graphe SLAM (bundle adjustment)
- [ ] Détection de personnes (YOLO/MobileNet)
- [ ] Reconnaissance de QR codes/ArUco markers

## 📄 Licence

Ce projet est fourni à des fins éducatives et de recherche.

## 👤 Auteur

Généré par Claude (Anthropic) pour l'exploration de zones dangereuses avec drone Tello EDU.
