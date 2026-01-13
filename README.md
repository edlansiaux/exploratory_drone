# 🚁 Tello Explorer - Système d'Exploration Autonome

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Système complet d'exploration autonome pour drone **DJI Tello EDU**, conçu pour l'inspection de bâtiments endommagés (incendies, explosions, catastrophes naturelles).

## 🎯 Fonctionnalités

- **Navigation Autonome** : Patterns d'exploration (snake, spiral, room_search)
- **Évitement d'Obstacles** : Détection et contournement temps réel (<100ms)
- **Cartographie Duale** : Cartes spatiale + thermique simultanées
- **Détection Thermique** : Identification zones chaudes et foyers d'incendie
- **Sécurité Multi-Couches** : Réflexes, évitement planifié, scans 360°
- **Mode Simulation** : Test complet sans matériel

## 📁 Structure du Projet

```
tello_explorer_optimized/
├── tello_controller.py      # Contrôle drone et navigation
├── vision.py                # Traitement vidéo et détection
├── mapping.py               # Cartographie duale (altitude + thermique)
├── obstacle_avoidance.py    # Système d'évitement d'obstacles
├── exploration.py           # Orchestration mission complète
├── demo_exploration_complete.ipynb  # Notebook de démonstration
├── main.py                  # Point d'entrée rapide
├── requirements.txt         # Dépendances Python
└── README.md               # Cette documentation
```

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- Drone DJI Tello EDU (optionnel avec mode simulation)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Vérification de l'installation

```bash
python main.py --test
```

## 💻 Utilisation Rapide

### Mode Simulation (sans drone)

```python
from exploration import ExplorationMission, MissionConfig

# Configuration mission
config = MissionConfig(
    area_width=300,           # Zone 3m x 3m
    area_height=300,
    exploration_altitude=100, # Vol à 1m
    pattern="snake"           # Pattern serpent
)

# Création mission en mode simulation
mission = ExplorationMission(config, simulation_mode=True)

# Préparation et lancement
mission.prepare_mission()
mission.start_exploration()

# Attente (la mission s'exécute en background)
import time
time.sleep(20)

# Arrêt et résultats
mission.stop_exploration()
mission.display_map()
print(mission.get_mission_report())
```

### Mode Réel (avec drone)

```python
from exploration import ExplorationMission, MissionConfig

config = MissionConfig(
    area_width=500,
    area_height=500,
    exploration_altitude=120,
    min_battery=20,           # Sécurité batterie
    max_duration=300          # 5 minutes max
)

mission = ExplorationMission(config, simulation_mode=False)
mission.prepare_mission()
mission.start_exploration()
```

### Ligne de commande

```bash
# Mode simulation avec paramètres par défaut
python main.py --simulation

# Mode réel
python main.py

# Personnalisation
python main.py --simulation --width 400 --height 400 --pattern spiral --duration 30
```

## ⚙️ Configuration

### Paramètres de Mission

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `area_width` | 500 | Largeur zone exploration (cm) |
| `area_height` | 500 | Hauteur zone exploration (cm) |
| `exploration_altitude` | 100 | Altitude de vol (cm) |
| `step_size` | 50 | Précision grille (cm) |
| `pattern` | "snake" | Pattern: snake, spiral, room_search |
| `scan_interval` | 300 | Distance entre scans 360° (cm) |
| `safety_margin` | 80 | Marge sécurité obstacles (cm) |
| `min_battery` | 15 | Batterie minimum (%) |
| `max_duration` | 600 | Durée maximum mission (s) |

### Zones de Sécurité

```python
from obstacle_avoidance import SafetyZone

zone = SafetyZone(
    front=100,      # Distance sécurité avant (cm)
    back=60,        # Distance sécurité arrière (cm)
    left=80,        # Distance sécurité gauche (cm)
    right=80,       # Distance sécurité droite (cm)
    above=50,       # Distance sécurité haut (cm)
    below=40        # Distance sécurité bas (cm)
)
```

## 📊 Cartographie

Le système génère deux cartes simultanément :

### Carte d'Altitude
- Grille avec altitude estimée du sol
- Détection de trous et dénivelés
- Résolution configurable

### Carte Thermique
- Températures par cellule (20-200°C)
- Identification zones chaudes
- Détection foyers d'incendie

### Export des Données

```python
# Export JSON + NumPy
mission.export_results("output_folder")

# Fichiers générés :
# - output_folder_map.json      : Données structurées
# - output_folder_altitude.npy  : Grille altitude
# - output_folder_thermal.npy   : Grille thermique
# - output_folder_occupancy.npy : Grille occupation
```

## 🛡️ Système de Sécurité

### 3 Niveaux de Protection

1. **Réflexes** (<100ms)
   - Arrêt immédiat si obstacle <40cm
   - Montée automatique si altitude <25cm
   - Évitement latéral si obstacle <30cm

2. **Évitement Planifié**
   - Analyse trajectoire avant mouvement
   - Calcul chemin de contournement
   - 9 stratégies adaptatives

3. **Scans Périodiques**
   - Rotation 360° tous les X mètres
   - Vérification 6 directions
   - Mise à jour carte obstacles

### Stratégies d'Évitement

| Stratégie | Condition |
|-----------|-----------|
| STOP | Obstacle proche, évaluation nécessaire |
| GO_LEFT/RIGHT | Obstacle frontal, côté libre |
| GO_UP/DOWN | Obstacle horizontal, vertical libre |
| BACKTRACK | Impasse, retour arrière |
| GO_AROUND | Contournement planifié |
| EMERGENCY_LAND | Danger critique |

## 🔥 Détection Thermique

### Classification des Zones

| Type | Température | Action |
|------|-------------|--------|
| cold | <20°C | Normal |
| normal | 20-50°C | Normal |
| warm | 50-80°C | Attention |
| hot | 80-150°C | ⚠️ Alerte |
| fire | >150°C | 🔥 Danger |

### Callbacks d'Alertes

```python
def on_thermal_alert(position, temperature, hotspots):
    print(f"⚠️ Zone chaude détectée: {temperature}°C à {position}")
    for hs in hotspots:
        print(f"  - {hs['type']}: {hs['temperature']}°C")

mission.on_thermal_alert = on_thermal_alert
```

## 📈 Rapport de Mission

```python
report = mission.get_mission_report()
```

Contenu du rapport :
- **Progression** : Waypoints complétés, pourcentage
- **Couverture** : Pourcentage zone cartographiée
- **Obstacles** : Nombre détecté, types, mobiles
- **Thermique** : Température max, zones chaudes, feu détecté
- **Sécurité** : Collisions évitées, temps réaction
- **Télémétrie** : Batterie, durée, position finale

## 🧪 Tests

### Tests Unitaires

```bash
# Test contrôleur
python tello_controller.py

# Test vision
python vision.py

# Test cartographie
python mapping.py

# Test évitement
python obstacle_avoidance.py

# Test exploration complète
python exploration.py
```

### Notebook Jupyter

```bash
jupyter notebook demo_exploration_complete.ipynb
```

## 📚 Architecture Technique

### Flux de Données

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Tello     │────▶│   Vision     │────▶│  Mapping    │
│  Controller │     │  Processing  │     │   (Dual)    │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │                    │
       │                   ▼                    │
       │           ┌──────────────┐             │
       └──────────▶│  Obstacle    │◀────────────┘
                   │  Avoidance   │
                   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ Exploration  │
                   │   Mission    │
                   └──────────────┘
```

### Threads et Timing

| Composant | Fréquence | Description |
|-----------|-----------|-------------|
| Video Stream | 30 Hz | Capture frames |
| Obstacle Monitor | 50 Hz | Surveillance continue |
| Reactive Avoidance | <100ms | Réflexes |
| Safety Scanner | Configurable | Scans 360° |

## ⚠️ Limitations

- **Caméra thermique** : Simulée via analyse colorimétrique (sans capteur IR réel)
- **SLAM** : Odométrie simple (pas de visual SLAM complet)
- **Indoor** : Conçu pour espaces intérieurs
- **Tello EDU** : Spécifique à ce modèle de drone

## 🔧 Extension

### Ajout de Capteurs

```python
# Exemple : intégration capteur externe
class CustomSensor:
    def read(self):
        return sensor_value

# Dans exploration.py
self.custom_sensor = CustomSensor()
```

### Nouveaux Patterns d'Exploration

```python
# Dans mapping.py, classe ExplorationPlanner
def custom_pattern(self):
    waypoints = []
    # Logique personnalisée
    return waypoints
```

## 📄 Licence

MIT License - Voir fichier [LICENSE](LICENSE)

## 👥 Contribution

Les contributions sont bienvenues ! Merci de :
1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📞 Support

Pour toute question ou problème :
- Ouvrir une Issue sur le dépôt
- Consulter la documentation des modules (docstrings)

---

**⚡ Développé pour sauver des vies dans les environnements dangereux**
