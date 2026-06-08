#!/usr/bin/env python3
"""
Tello Explorer - Point d'entrée principal
==========================================

Lancement rapide d'une mission d'exploration.

Usage:
    python main.py --simulation              # Mode simulation
    python main.py                           # Mode réel (drone connecté)
    python main.py --help                    # Aide
    python main.py --test                    # Test rapide des modules

Exemples:
    python main.py --simulation --width 400 --height 400 --pattern spiral
    python main.py --simulation --duration 30 --altitude 120
    python main.py --host 192.168.10.1       # Connexion drone par IP explicite
"""

import argparse
import sys
import time

from tello_controller import DEFAULT_TELLO_HOST


def test_modules():
    """Test rapide de tous les modules."""
    print("=" * 60)
    print("🧪 TEST DES MODULES TELLO EXPLORER")
    print("=" * 60)
    
    results = []
    
    # Test 1: Controller
    print("\n📦 Test TelloController...")
    try:
        from tello_controller import TelloController
        ctrl = TelloController(simulation_mode=True)
        ctrl.connect()
        ctrl.takeoff()
        ctrl.move_forward(50)
        ctrl.rotate_clockwise(90)
        ctrl.land()
        print("   ✅ TelloController OK")
        results.append(("TelloController", True))
    except Exception as e:
        print(f"   ❌ TelloController ERREUR: {e}")
        results.append(("TelloController", False))
    
    # Test 2: Vision
    print("\n📦 Test Vision...")
    try:
        from vision import ObstacleDetector, ThermalDetector
        import numpy as np
        
        detector = ObstacleDetector()
        thermal = ThermalDetector()
        
        # Frame test (résolution native Tello: 720x960)
        frame = np.random.randint(0, 255, (720, 960, 3), dtype=np.uint8)
        obstacles = detector.detect(frame)
        thermal_map, hotspots = thermal.detect(frame)
        
        print(f"   Obstacles détectés: {len(obstacles)}")
        print(f"   Hotspots détectés: {len(hotspots)}")
        print("   ✅ Vision OK")
        results.append(("Vision", True))
    except Exception as e:
        print(f"   ❌ Vision ERREUR: {e}")
        results.append(("Vision", False))
    
    # Test 3: Mapping
    print("\n📦 Test Mapping...")
    try:
        from mapping import DualMap, ExplorationPlanner
        
        dual_map = DualMap(resolution=50, size=(200, 200))
        dual_map.add_point(50, 50, 100, ground_distance=100)
        dual_map.add_obstacle(100, 100, 100, radius=30, obstacle_type="debris")
        dual_map.add_thermal_zone(150, 150, 50, radius=30, temperature=120.0)
        
        planner = ExplorationPlanner(dual_map, step_size=50)
        waypoints = planner.generate_snake_pattern(200, 200, 100)
        
        print(f"   Waypoints générés: {len(waypoints)}")
        print(f"   Couverture: {dual_map.get_exploration_coverage():.1f}%")
        print("   ✅ Mapping OK")
        results.append(("Mapping", True))
    except Exception as e:
        print(f"   ❌ Mapping ERREUR: {e}")
        results.append(("Mapping", False))
    
    # Test 4: Obstacle Avoidance
    print("\n📦 Test Obstacle Avoidance...")
    try:
        from obstacle_avoidance import ObstacleAvoidanceSystem, SafetyZone, AvoidanceStrategy
        
        zone = SafetyZone(front=80, back=50, left=60, right=60)
        avoidance = ObstacleAvoidanceSystem(safety_zone=zone)
        
        # Simuler obstacle avec direction
        obs = avoidance.add_obstacle(50, 0, 100, distance=50, direction=(1, 0, 0), obstacle_type="wall")
        
        # Test stratégie avec arguments
        strategy = avoidance.get_avoidance_strategy(
            drone_pos=(0, 0, 100),
            target_pos=(100, 0, 100),
            obstacle=obs
        )
        
        print(f"   Obstacle ajouté: {obs.obstacle_type}")
        print(f"   Stratégie: {strategy.name}")
        print("   ✅ Obstacle Avoidance OK")
        results.append(("ObstacleAvoidance", True))
    except Exception as e:
        print(f"   ❌ Obstacle Avoidance ERREUR: {e}")
        results.append(("ObstacleAvoidance", False))
    
    # Test 5: Exploration
    print("\n📦 Test Exploration...")
    try:
        from exploration import ExplorationMission, MissionConfig
        
        config = MissionConfig(
            area_width=200,
            area_height=200,
            step_size=50,
            pattern="snake"
        )
        mission = ExplorationMission(config, simulation_mode=True)
        mission.prepare_mission()
        
        print(f"   Waypoints: {mission.total_waypoints}")
        print(f"   Pattern: {config.pattern}")
        print(f"   Host configuré: {config.host}")
        print("   ✅ Exploration OK")
        results.append(("Exploration", True))
    except Exception as e:
        print(f"   ❌ Exploration ERREUR: {e}")
        results.append(("Exploration", False))
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   Total: {passed}/{total} modules OK")
    
    if passed == total:
        print("\n🎉 Tous les tests passés ! Le système est opérationnel.")
        return 0
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        return 1


def run_mission(args):
    """Exécute une mission d'exploration."""
    from exploration import ExplorationMission, MissionConfig
    
    print("=" * 60)
    print("🚁 TELLO EXPLORER - MISSION D'EXPLORATION")
    print("=" * 60)
    
    # Configuration
    config = MissionConfig(
        area_width=args.width,
        area_height=args.height,
        exploration_altitude=args.altitude,
        step_size=args.step,
        pattern=args.pattern,
        scan_interval=args.scan_interval,
        safety_margin=args.safety_margin,
        min_battery=args.min_battery,
        max_duration=args.max_duration,
        host=args.host
    )
    
    print(f"\n📋 Configuration:")
    print(f"   Zone: {args.width}x{args.height} cm")
    print(f"   Altitude: {args.altitude} cm")
    print(f"   Pattern: {args.pattern}")
    print(f"   Mode: {'Simulation' if args.simulation else 'Réel'}")
    if not args.simulation:
        print(f"   Host drone: {args.host}")
    print(f"   Durée max: {args.duration}s")
    
    # Création mission
    mission = ExplorationMission(config, simulation_mode=args.simulation)
    
    # Callbacks
    def on_status(old, new):
        print(f"   📡 Status: {old.value} → {new.value}")
    
    def on_waypoint(wp, progress):
        print(f"   📍 Waypoint ({wp[0]:.0f}, {wp[1]:.0f}): {progress:.1f}%")
    
    def on_obstacle(obs):
        otype = getattr(obs, 'obstacle_type', 'unknown')
        odist = getattr(obs, 'distance', '?')
        print(f"   ⚠️  Obstacle: {otype} à {odist}cm")
    
    def on_thermal(pos, temp, hotspots):
        print(f"   🔥 Alerte thermique: {temp:.1f}°C à ({pos.x:.0f}, {pos.y:.0f})")
    
    mission.on_status_change = on_status
    mission.on_waypoint_reached = on_waypoint
    mission.on_obstacle_detected = on_obstacle
    mission.on_thermal_alert = on_thermal
    
    # Préparation
    print(f"\n🔧 Préparation de la mission...")
    if not mission.prepare_mission():
        print("❌ Échec de la préparation")
        return 1
    
    print(f"   Waypoints planifiés: {mission.total_waypoints}")
    
    # Ajout obstacles simulés (mode simulation uniquement)
    if args.simulation:
        print("\n🏗️  Ajout environnement simulé...")
        mission.add_simulated_obstacle(100, 50, 100, obstacle_type="debris")
        mission.add_simulated_obstacle(0, 150, 100, obstacle_type="wall")
        mission.dual_map.add_thermal_zone(120, 80, 50, radius=40, temperature=150.0)
        mission.dual_map.add_thermal_zone(50, 180, 50, radius=30, temperature=85.0)
        print("   ✅ Environnement configuré")
    
    # Lancement
    print(f"\n🚀 Démarrage exploration ({args.duration}s)...")
    mission.start_exploration()
    
    # Attente avec affichage progression
    start_time = time.time()
    try:
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            
            # Affichage périodique
            if int(elapsed) % 5 == 0 and elapsed > 0:
                report = mission.get_mission_report()
                coverage = report['mapping']['coverage']
                battery = report['drone']['battery']
                print(f"\n   ⏱️  {int(elapsed)}s | Couverture: {coverage:.1f}% | Batterie: {battery}%")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⚡ Interruption utilisateur")
    
    # Arrêt
    print("\n🛑 Arrêt de la mission...")
    mission.stop_exploration()
    
    # Rapport final
    print("\n" + "=" * 60)
    print("📊 RAPPORT DE MISSION")
    print("=" * 60)
    
    report = mission.get_mission_report()
    
    print(f"\n📈 Progression:")
    print(f"   Waypoints: {report['waypoints']['completed']}/{report['waypoints']['total']}")
    print(f"   Couverture: {report['mapping']['coverage']:.1f}%")
    
    print(f"\n🚧 Obstacles:")
    print(f"   Surveillés: {report['avoidance']['total_obstacles']}")
    print(f"   Collisions évitées: {report['avoidance']['stats']['collisions_avoided']}")
    
    print(f"\n🌡️  Thermique:")
    print(f"   Température max: {report['thermal']['max_temperature']:.1f}°C")
    print(f"   Zones chaudes: {report['thermal']['zones']}")
    print(f"   Feu détecté: {'Oui 🔥' if report['thermal']['fire_detected'] else 'Non'}")
    
    print(f"\n🔋 Télémétrie:")
    print(f"   Batterie: {report['drone']['battery']}%")
    print(f"   Durée: {report['duration_seconds']:.1f}s")
    
    # Affichage carte
    if args.show_map:
        print("\n📍 Carte d'exploration:")
        mission.display_map(show_thermal=False)
        
        print("\n🌡️  Carte thermique:")
        mission.display_map(show_thermal=True)
    
    # Export
    if args.export:
        print(f"\n💾 Export des données vers '{args.export}'...")
        mission.export_results(args.export)
        print("   ✅ Export terminé")
    
    print("\n" + "=" * 60)
    print("✅ MISSION TERMINÉE")
    print("=" * 60)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Tello Explorer - Système d'exploration autonome",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py --simulation                    Mode simulation rapide
  python main.py --simulation --duration 60      Simulation 60 secondes
  python main.py --simulation --pattern spiral   Pattern spirale
  python main.py --host 192.168.10.1             Connexion drone par IP
  python main.py --test                          Test des modules
        """
    )
    
    # Mode
    parser.add_argument('--simulation', '-s', action='store_true',
                        help='Mode simulation (sans drone)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Test rapide des modules')
    
    # Connexion drone
    parser.add_argument('--host', type=str, default=DEFAULT_TELLO_HOST,
                        help=f'Adresse IP du drone (défaut: {DEFAULT_TELLO_HOST})')
    
    # Zone
    parser.add_argument('--width', '-W', type=int, default=300,
                        help='Largeur zone exploration en cm (défaut: 300)')
    parser.add_argument('--height', '-H', type=int, default=300,
                        help='Hauteur zone exploration en cm (défaut: 300)')
    
    # Navigation
    parser.add_argument('--altitude', '-a', type=int, default=100,
                        help='Altitude de vol en cm (défaut: 100)')
    parser.add_argument('--step', type=int, default=50,
                        help='Taille des pas en cm (défaut: 50)')
    parser.add_argument('--pattern', '-p', choices=['snake', 'spiral', 'room_search'],
                        default='snake', help='Pattern exploration (défaut: snake)')
    
    # Sécurité
    parser.add_argument('--scan-interval', type=int, default=200,
                        help='Intervalle scans 360° en cm (défaut: 200)')
    parser.add_argument('--safety-margin', type=int, default=80,
                        help='Marge sécurité en cm (défaut: 80)')
    parser.add_argument('--min-battery', type=int, default=15,
                        help='Batterie minimum en %% (défaut: 15)')
    
    # Durée
    parser.add_argument('--duration', '-d', type=int, default=20,
                        help='Durée mission en secondes (défaut: 20)')
    parser.add_argument('--max-duration', type=int, default=600,
                        help='Durée maximum en secondes (défaut: 600)')
    
    # Sortie
    parser.add_argument('--show-map', '-m', action='store_true', default=True,
                        help='Afficher les cartes (défaut: True)')
    parser.add_argument('--no-map', action='store_false', dest='show_map',
                        help='Ne pas afficher les cartes')
    parser.add_argument('--export', '-e', type=str, default=None,
                        help='Préfixe fichiers export (ex: mission_001)')
    
    args = parser.parse_args()
    
    # Mode test
    if args.test:
        return test_modules()
    
    # Mode mission
    return run_mission(args)


if __name__ == "__main__":
    sys.exit(main())
