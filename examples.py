#!/usr/bin/env python3
"""
Exemple d'utilisation simple du système d'exploration Tello EDU
Ce script montre les fonctionnalités de base en mode simulation
"""

import time
import sys

# Ajouter le répertoire au path
sys.path.insert(0, '/home/claude/tello_explorer')

from tello_controller import TelloController
from mapping import AltitudeMap, ExplorationPlanner
from obstacle_avoidance import ObstacleAvoidanceSystem
from exploration import ExplorationMission, MissionConfig


def exemple_controle_manuel():
    """Exemple de contrôle manuel basique"""
    print("\n" + "="*60)
    print("EXEMPLE 1: CONTRÔLE MANUEL")
    print("="*60)
    
    # Créer le contrôleur en mode simulation
    controller = TelloController(simulation_mode=True)
    
    # Connexion
    print("\n1. Connexion au drone...")
    controller.connect()
    
    # Décollage
    print("\n2. Décollage...")
    controller.takeoff()
    print(f"   Position: {controller.position.to_tuple()}")
    
    # Mouvements de base
    print("\n3. Mouvements de 50cm chacun:")
    
    print("   - Droite 50cm")
    controller.move_right(50)
    print(f"     Position: {controller.position.to_tuple()}")
    
    print("   - Avant 50cm")
    controller.move_forward(50)
    print(f"     Position: {controller.position.to_tuple()}")
    
    print("   - Monter 50cm")
    controller.move_up(50)
    print(f"     Position: {controller.position.to_tuple()}")
    
    print("   - Gauche 50cm")
    controller.move_left(50)
    print(f"     Position: {controller.position.to_tuple()}")
    
    print("   - Descendre 50cm")
    controller.move_down(50)
    print(f"     Position: {controller.position.to_tuple()}")
    
    print("   - Arrière 50cm")
    controller.move_back(50)
    print(f"     Position: {controller.position.to_tuple()}")
    
    # Télémétrie
    print("\n4. Télémétrie:")
    telemetry = controller.get_telemetry()
    for key, value in telemetry.items():
        print(f"   {key}: {value}")
    
    # Atterrissage
    print("\n5. Atterrissage...")
    controller.land()
    
    # Déconnexion
    controller.disconnect()
    print("\n✓ Exemple terminé")


def exemple_cartographie():
    """Exemple de création de carte d'altitude"""
    print("\n" + "="*60)
    print("EXEMPLE 2: CARTOGRAPHIE")
    print("="*60)
    
    import numpy as np
    
    # Créer une carte
    altitude_map = AltitudeMap(resolution=50, size=(500, 500))
    
    print("\n1. Ajout de points d'exploration simulés...")
    
    # Simuler un terrain avec variation d'altitude
    for x in range(-200, 250, 50):
        for y in range(-200, 250, 50):
            # Terrain vallonné
            ground_dist = 100 + 30 * np.sin(x/100) * np.cos(y/100)
            altitude_map.add_point(x, y, 150, ground_dist)
    
    print(f"   Points enregistrés: {len(altitude_map.raw_points)}")
    
    # Ajouter des obstacles
    print("\n2. Ajout d'obstacles...")
    altitude_map.add_obstacle(100, 100, 100, radius=30, is_mobile=False)
    altitude_map.add_obstacle(-50, 50, 80, radius=25, is_mobile=True)
    print(f"   Obstacles: {len(altitude_map.obstacles)}")
    
    # Statistiques
    print("\n3. Statistiques d'altitude:")
    stats = altitude_map.get_altitude_stats()
    print(f"   Min: {stats['min']:.1f} cm")
    print(f"   Max: {stats['max']:.1f} cm")
    print(f"   Moyenne: {stats['mean']:.1f} cm")
    print(f"   Écart-type: {stats['std']:.1f} cm")
    
    print(f"\n4. Couverture: {altitude_map.get_exploration_coverage():.1f}%")
    
    # Afficher la carte ASCII
    print("\n5. Carte d'altitude:")
    print(altitude_map.to_ascii_map(drone_pos=(0, 0)))
    
    print("\n✓ Exemple terminé")


def exemple_evitement_obstacles():
    """Exemple du système d'évitement d'obstacles"""
    print("\n" + "="*60)
    print("EXEMPLE 3: ÉVITEMENT D'OBSTACLES")
    print("="*60)
    
    # Créer le système
    avoidance = ObstacleAvoidanceSystem()
    
    # Position du drone
    drone_pos = (0, 0, 100)
    target_pos = (200, 100, 100)
    
    print(f"\n1. Position drone: {drone_pos}")
    print(f"   Position cible: {target_pos}")
    
    # Ajouter un obstacle sur le chemin
    print("\n2. Ajout d'un obstacle sur le trajet...")
    avoidance.add_obstacle(100, 50, 100, 112, (0.89, 0.45, 0))
    
    # Vérifier le risque de collision
    print("\n3. Vérification du risque de collision...")
    collision, obstacle = avoidance.check_collision_risk(
        *drone_pos, *target_pos
    )
    
    print(f"   Risque de collision: {collision}")
    
    if collision and obstacle:
        print(f"   Obstacle à: ({obstacle.x:.0f}, {obstacle.y:.0f}, {obstacle.z:.0f})")
        
        # Déterminer la stratégie
        strategy = avoidance.get_avoidance_strategy(drone_pos, target_pos, obstacle)
        print(f"\n4. Stratégie choisie: {strategy.value}")
        
        # Calculer le chemin d'évitement
        waypoints = avoidance.calculate_avoidance_waypoints(
            drone_pos, target_pos, obstacle, strategy
        )
        print(f"\n5. Waypoints d'évitement:")
        for i, wp in enumerate(waypoints):
            print(f"   {i+1}. ({wp[0]:.0f}, {wp[1]:.0f}, {wp[2]:.0f})")
    
    # Test avec obstacle mobile
    print("\n6. Test avec obstacle mobile...")
    mobile_obs = avoidance.add_obstacle(0, 150, 100, 150, (0, 1, 0))
    
    # Simuler mouvement
    for i in range(5):
        time.sleep(0.1)
        avoidance.simulate_obstacle_movement(1, (20, 10, 0))
    
    print(f"   Obstacle mobile: {mobile_obs.is_mobile}")
    print(f"   Vélocité: {mobile_obs.get_velocity()}")
    print(f"   Position prédite (2s): {mobile_obs.predict_position(2.0)}")
    
    print("\n✓ Exemple terminé")


def exemple_mission_complete():
    """Exemple de mission d'exploration complète"""
    print("\n" + "="*60)
    print("EXEMPLE 4: MISSION D'EXPLORATION COMPLÈTE")
    print("="*60)
    
    # Configuration
    config = MissionConfig(
        area_width=300,
        area_height=300,
        exploration_altitude=100,
        step_size=50,
        pattern="snake",
        max_duration=60
    )
    
    print("\n1. Configuration:")
    print(f"   Zone: {config.area_width}x{config.area_height} cm")
    print(f"   Altitude: {config.exploration_altitude} cm")
    print(f"   Pattern: {config.pattern}")
    
    # Créer la mission
    print("\n2. Création de la mission (simulation)...")
    mission = ExplorationMission(config, simulation_mode=True)
    
    # Callbacks
    waypoint_count = [0]
    def on_waypoint(wp, progress):
        waypoint_count[0] += 1
        if waypoint_count[0] % 5 == 0:  # Afficher tous les 5 waypoints
            print(f"   Waypoint {waypoint_count[0]}: ({wp[0]:.0f}, {wp[1]:.0f}) - {progress:.1f}%")
    
    mission.on_waypoint_reached = on_waypoint
    
    # Préparer
    print("\n3. Préparation de la mission...")
    if mission.prepare_mission():
        print(f"   Waypoints planifiés: {mission.total_waypoints}")
    
    # Ajouter obstacles simulés
    print("\n4. Ajout d'obstacles simulés...")
    mission.add_simulated_obstacle(75, 50, 100, is_mobile=False)
    mission.add_simulated_obstacle(0, 100, 100, is_mobile=True, velocity=(10, 5, 0))
    
    # Ajouter zones de danger
    print("\n5. Ajout de zones de danger simulées...")
    mission.hazard_detector.add_simulated_hotspot(100, 100, 0, 'thermal', 80)
    mission.hazard_detector.add_simulated_hotspot(-50, 50, 0, 'radiation', 1.0)
    
    # Démarrer l'exploration
    print("\n6. Démarrage de l'exploration...")
    mission.start_exploration()
    
    # Laisser l'exploration se dérouler
    print("   Exploration en cours (5 secondes)...")
    time.sleep(5)
    
    # Arrêter
    print("\n7. Arrêt de l'exploration...")
    mission.stop_exploration()
    
    # Afficher la carte
    print("\n8. Carte d'exploration:")
    mission.display_map()
    
    # Rapport
    print("\n9. Rapport de mission:")
    report = mission.get_mission_report()
    print(f"   Statut: {report['status']}")
    print(f"   Waypoints: {report['waypoints']['completed']}/{report['waypoints']['total']}")
    print(f"   Couverture carte: {report['mapping']['coverage']:.1f}%")
    print(f"   Obstacles détectés: {report['obstacles']['total_obstacles']}")
    
    print("\n✓ Exemple terminé")


def main():
    """Exécute tous les exemples"""
    print("╔" + "═"*58 + "╗")
    print("║" + " EXEMPLES D'UTILISATION DU SYSTÈME TELLO EDU ".center(58) + "║")
    print("╚" + "═"*58 + "╝")
    
    try:
        exemple_controle_manuel()
        exemple_cartographie()
        exemple_evitement_obstacles()
        exemple_mission_complete()
        
        print("\n" + "="*60)
        print("TOUS LES EXEMPLES TERMINÉS AVEC SUCCÈS!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
