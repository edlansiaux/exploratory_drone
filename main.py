#!/usr/bin/env python3
"""
Interface utilisateur principale pour le système d'exploration Tello EDU
Permet le contrôle manuel et automatique du drone
"""

import cmd
import sys
import time
import threading
from typing import Optional

from exploration import (
    ExplorationMission, 
    MissionConfig, 
    ManualController,
    MissionStatus,
    ExplorationMode
)


class TelloExplorerCLI(cmd.Cmd):
    """Interface en ligne de commande pour l'exploration Tello"""
    
    intro = """
╔══════════════════════════════════════════════════════════════════╗
║          SYSTÈME D'EXPLORATION TELLO EDU - v1.0                  ║
║                                                                  ║
║  Conçu pour l'exploration de zones à risque NRBC                 ║
║  (Thermique/Nucléaire/Radiologique/Biologique/Chimique)          ║
╚══════════════════════════════════════════════════════════════════╝

Tapez 'help' ou '?' pour la liste des commandes.
Tapez 'quickstart' pour un guide rapide.
"""
    prompt = "tello> "
    
    def __init__(self, simulation_mode: bool = False):
        super().__init__()
        self.simulation_mode = simulation_mode
        self.mission: Optional[ExplorationMission] = None
        self.manual: Optional[ManualController] = None
        self._live_display_thread: Optional[threading.Thread] = None
        self._stop_live_display = threading.Event()
        
        # Configuration par défaut
        self.config = MissionConfig()
        
        if simulation_mode:
            print("\n⚠️  MODE SIMULATION ACTIVÉ - Aucun drone réel nécessaire\n")
    
    # ==================== Commandes d'aide ====================
    
    def do_quickstart(self, arg):
        """Affiche un guide de démarrage rapide"""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                      GUIDE DE DÉMARRAGE RAPIDE                   ║
╚══════════════════════════════════════════════════════════════════╝

1. INITIALISATION
   > init                    # Initialise la mission
   
2. CONTRÔLE MANUEL
   > takeoff                 # Décollage
   > up / down              # Monter/descendre de 50cm
   > left / right           # Gauche/droite de 50cm
   > forward / back         # Avant/arrière de 50cm
   > land                    # Atterrissage

3. EXPLORATION AUTOMATIQUE
   > config                  # Voir/modifier la configuration
   > prepare                 # Préparer la mission
   > start                   # Démarrer l'exploration auto
   > pause / resume          # Pause/reprise
   > stop                    # Arrêter et atterrir
   
4. VISUALISATION
   > status                  # État du drone
   > map                     # Afficher la carte
   > report                  # Rapport de mission
   
5. SIMULATION D'OBSTACLES
   > obstacle 100 50 100     # Ajouter obstacle fixe
   > mobile 0 100 100 10 5 0 # Ajouter obstacle mobile

6. URGENCE
   > emergency               # ARRÊT D'URGENCE IMMÉDIAT
   > home                    # Retour au point de départ
""")
    
    # ==================== Commandes d'initialisation ====================
    
    def do_init(self, arg):
        """Initialise le système avec la configuration actuelle"""
        print("Initialisation du système...")
        self.mission = ExplorationMission(self.config, self.simulation_mode)
        self.manual = ManualController(self.mission)
        
        # Configuration des callbacks
        self.mission.on_status_change = self._on_status_change
        self.mission.on_waypoint_reached = self._on_waypoint
        self.mission.on_hazard_detected = self._on_hazard
        self.mission.on_obstacle_detected = self._on_obstacle
        
        print("✓ Système initialisé")
        print(f"  Mode: {'Simulation' if self.simulation_mode else 'Réel'}")
        print(f"  Zone: {self.config.area_width}x{self.config.area_height} cm")
    
    def do_connect(self, arg):
        """Connecte au drone"""
        if not self._check_init():
            return
        
        print("Connexion au drone...")
        if self.mission.controller.connect():
            print("✓ Connecté au drone")
            telemetry = self.mission.controller.get_telemetry()
            print(f"  Batterie: {telemetry.get('battery', 'N/A')}%")
        else:
            print("✗ Échec de la connexion")
    
    def do_disconnect(self, arg):
        """Déconnecte du drone"""
        if not self._check_init():
            return
        self.mission.controller.disconnect()
        print("✓ Déconnecté")
    
    # ==================== Commandes de configuration ====================
    
    def do_config(self, arg):
        """
        Affiche ou modifie la configuration
        Usage: config [paramètre] [valeur]
        Paramètres: width, height, altitude, step, pattern, duration, battery
        """
        args = arg.split()
        
        if not args:
            # Afficher la configuration actuelle
            print("\n=== Configuration actuelle ===")
            print(f"  area_width:    {self.config.area_width} cm")
            print(f"  area_height:   {self.config.area_height} cm")
            print(f"  altitude:      {self.config.exploration_altitude} cm")
            print(f"  step_size:     {self.config.step_size} cm")
            print(f"  pattern:       {self.config.pattern}")
            print(f"  max_duration:  {self.config.max_duration} s")
            print(f"  min_battery:   {self.config.min_battery}%")
            print(f"  mapping:       {self.config.enable_mapping}")
            print(f"  avoidance:     {self.config.enable_avoidance}")
            return
        
        if len(args) < 2:
            print("Usage: config <paramètre> <valeur>")
            return
        
        param, value = args[0], args[1]
        
        try:
            if param == "width":
                self.config.area_width = float(value)
            elif param == "height":
                self.config.area_height = float(value)
            elif param == "altitude":
                self.config.exploration_altitude = float(value)
            elif param == "step":
                self.config.step_size = float(value)
            elif param == "pattern":
                if value in ["snake", "spiral"]:
                    self.config.pattern = value
                else:
                    print("Pattern invalide (snake ou spiral)")
                    return
            elif param == "duration":
                self.config.max_duration = float(value)
            elif param == "battery":
                self.config.min_battery = int(value)
            else:
                print(f"Paramètre inconnu: {param}")
                return
            
            print(f"✓ {param} = {value}")
            
        except ValueError:
            print("Valeur invalide")
    
    # ==================== Commandes de vol manuel ====================
    
    def do_takeoff(self, arg):
        """Fait décoller le drone"""
        if not self._check_init():
            return
        
        print("Décollage...")
        if self.manual.takeoff():
            print("✓ En vol")
        else:
            print("✗ Échec du décollage")
    
    def do_land(self, arg):
        """Fait atterrir le drone"""
        if not self._check_init():
            return
        
        print("Atterrissage...")
        if self.manual.land():
            print("✓ Au sol")
        else:
            print("✗ Échec de l'atterrissage")
    
    def do_up(self, arg):
        """Monte le drone (défaut: 50cm). Usage: up [distance]"""
        if not self._check_init():
            return
        
        distance = int(arg) if arg else 50
        print(f"Montée de {distance}cm...")
        self.manual.up(distance)
    
    def do_down(self, arg):
        """Descend le drone (défaut: 50cm). Usage: down [distance]"""
        if not self._check_init():
            return
        
        distance = int(arg) if arg else 50
        print(f"Descente de {distance}cm...")
        self.manual.down(distance)
    
    def do_left(self, arg):
        """Déplace le drone à gauche (défaut: 50cm). Usage: left [distance]"""
        if not self._check_init():
            return
        
        distance = int(arg) if arg else 50
        print(f"Gauche de {distance}cm...")
        self.manual.left(distance)
    
    def do_right(self, arg):
        """Déplace le drone à droite (défaut: 50cm). Usage: right [distance]"""
        if not self._check_init():
            return
        
        distance = int(arg) if arg else 50
        print(f"Droite de {distance}cm...")
        self.manual.right(distance)
    
    def do_forward(self, arg):
        """Avance le drone (défaut: 50cm). Usage: forward [distance]"""
        if not self._check_init():
            return
        
        distance = int(arg) if arg else 50
        print(f"Avant de {distance}cm...")
        self.manual.forward(distance)
    
    def do_back(self, arg):
        """Recule le drone (défaut: 50cm). Usage: back [distance]"""
        if not self._check_init():
            return
        
        distance = int(arg) if arg else 50
        print(f"Arrière de {distance}cm...")
        self.manual.back(distance)
    
    def do_rotate(self, arg):
        """
        Fait pivoter le drone. Usage: rotate <left|right> [angle]
        """
        if not self._check_init():
            return
        
        args = arg.split()
        if not args:
            print("Usage: rotate <left|right> [angle]")
            return
        
        direction = args[0]
        angle = int(args[1]) if len(args) > 1 else 90
        
        if direction == "left":
            self.manual.rotate_left(angle)
        elif direction == "right":
            self.manual.rotate_right(angle)
        else:
            print("Direction: left ou right")
    
    # ==================== Commandes d'exploration auto ====================
    
    def do_prepare(self, arg):
        """Prépare la mission d'exploration"""
        if not self._check_init():
            return
        
        print("Préparation de la mission...")
        if self.mission.prepare_mission():
            print("✓ Mission prête")
            print(f"  Waypoints: {self.mission.total_waypoints}")
        else:
            print("✗ Échec de la préparation")
    
    def do_start(self, arg):
        """Démarre l'exploration automatique"""
        if not self._check_init():
            return
        
        print("Démarrage de l'exploration...")
        if self.mission.start_exploration():
            print("✓ Exploration en cours")
        else:
            print("✗ Échec du démarrage")
    
    def do_pause(self, arg):
        """Met en pause l'exploration"""
        if not self._check_init():
            return
        
        self.mission.pause_exploration()
        print("⏸ Exploration en pause")
    
    def do_resume(self, arg):
        """Reprend l'exploration"""
        if not self._check_init():
            return
        
        self.mission.resume_exploration()
        print("▶ Exploration reprise")
    
    def do_stop(self, arg):
        """Arrête l'exploration et atterrit"""
        if not self._check_init():
            return
        
        print("Arrêt de l'exploration...")
        self.mission.stop_exploration()
        print("✓ Exploration terminée")
    
    def do_home(self, arg):
        """Retourne au point de départ"""
        if not self._check_init():
            return
        
        print("Retour à la base...")
        self.mission.return_to_home()
        print("✓ Retour effectué")
    
    def do_emergency(self, arg):
        """ARRÊT D'URGENCE - Coupe immédiatement les moteurs"""
        if not self._check_init():
            return
        
        print("⚠️  ARRÊT D'URGENCE ⚠️")
        self.mission.emergency_stop()
    
    # ==================== Commandes de visualisation ====================
    
    def do_status(self, arg):
        """Affiche l'état actuel du drone"""
        if not self._check_init():
            return
        
        self.manual.status()
        print(f"Mission: {self.mission.status.value}")
        print(f"Progression: {self.mission.planner.get_progress():.1f}%")
    
    def do_map(self, arg):
        """Affiche la carte d'exploration"""
        if not self._check_init():
            return
        
        self.mission.display_map()
    
    def do_report(self, arg):
        """Affiche le rapport de mission complet"""
        if not self._check_init():
            return
        
        report = self.mission.get_mission_report()
        
        print("\n" + "=" * 50)
        print("RAPPORT DE MISSION")
        print("=" * 50)
        
        print(f"\nÉtat: {report['status']}")
        print(f"Mode: {report['mode']}")
        print(f"Durée: {report['duration_seconds']:.1f}s")
        
        print(f"\nProgression:")
        print(f"  Waypoints: {report['waypoints']['completed']}/{report['waypoints']['total']}")
        print(f"  Couverture: {report['waypoints']['progress']:.1f}%")
        
        print(f"\nCartographie:")
        print(f"  Points enregistrés: {report['mapping']['points_recorded']}")
        print(f"  Couverture carte: {report['mapping']['coverage']:.1f}%")
        stats = report['mapping']['altitude_stats']
        if stats['min'] is not None:
            print(f"  Altitude sol: {stats['min']:.0f} - {stats['max']:.0f} cm")
        
        print(f"\nObstacles:")
        print(f"  Total: {report['obstacles']['total_obstacles']}")
        print(f"  Mobiles: {report['obstacles']['mobile_obstacles']}")
        
        print(f"\nDrone:")
        print(f"  Position: {report['drone']['position']}")
        print(f"  État: {report['drone']['state']}")
        print(f"  Batterie: {report['drone']['battery']}%")
        
        print("=" * 50)
    
    def do_live(self, arg):
        """Active/désactive l'affichage en direct. Usage: live [on|off]"""
        if not self._check_init():
            return
        
        if arg == "off":
            self._stop_live_display.set()
            print("Affichage live désactivé")
        else:
            if self._live_display_thread and self._live_display_thread.is_alive():
                print("Affichage live déjà actif")
                return
            
            self._stop_live_display.clear()
            self._live_display_thread = threading.Thread(
                target=self._live_display_loop,
                daemon=True
            )
            self._live_display_thread.start()
            print("Affichage live activé (Ctrl+C ou 'live off' pour arrêter)")
    
    def _live_display_loop(self):
        """Boucle d'affichage en direct"""
        while not self._stop_live_display.is_set():
            try:
                pos = self.mission.controller.position
                status = self.mission.status.value
                progress = self.mission.planner.get_progress()
                
                print(f"\r[{status}] Pos: ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f}) | "
                      f"Progress: {progress:.1f}%", end="", flush=True)
                
                time.sleep(0.5)
            except:
                break
    
    # ==================== Commandes de simulation ====================
    
    def do_obstacle(self, arg):
        """
        Ajoute un obstacle simulé fixe
        Usage: obstacle <x> <y> <z>
        """
        if not self._check_init():
            return
        
        args = arg.split()
        if len(args) < 3:
            print("Usage: obstacle <x> <y> <z>")
            return
        
        try:
            x, y, z = float(args[0]), float(args[1]), float(args[2])
            self.mission.add_simulated_obstacle(x, y, z, is_mobile=False)
            print(f"✓ Obstacle ajouté à ({x}, {y}, {z})")
        except ValueError:
            print("Valeurs invalides")
    
    def do_mobile(self, arg):
        """
        Ajoute un obstacle mobile simulé
        Usage: mobile <x> <y> <z> <vx> <vy> <vz>
        """
        if not self._check_init():
            return
        
        args = arg.split()
        if len(args) < 6:
            print("Usage: mobile <x> <y> <z> <vx> <vy> <vz>")
            return
        
        try:
            x, y, z = float(args[0]), float(args[1]), float(args[2])
            vx, vy, vz = float(args[3]), float(args[4]), float(args[5])
            self.mission.add_simulated_obstacle(
                x, y, z, 
                is_mobile=True, 
                velocity=(vx, vy, vz)
            )
            print(f"✓ Obstacle mobile ajouté à ({x}, {y}, {z}) vitesse ({vx}, {vy}, {vz})")
        except ValueError:
            print("Valeurs invalides")
    
    def do_hazard(self, arg):
        """
        Ajoute une zone de danger simulée
        Usage: hazard <x> <y> <z> <type> <intensity>
        Types: thermal, radiation, chemical
        """
        if not self._check_init():
            return
        
        args = arg.split()
        if len(args) < 5:
            print("Usage: hazard <x> <y> <z> <type> <intensity>")
            print("Types: thermal, radiation, chemical")
            return
        
        try:
            x, y, z = float(args[0]), float(args[1]), float(args[2])
            hazard_type = args[3]
            intensity = float(args[4])
            
            if hazard_type not in ['thermal', 'radiation', 'chemical']:
                print("Type invalide")
                return
            
            self.mission.hazard_detector.add_simulated_hotspot(
                x, y, z, hazard_type, intensity
            )
            print(f"✓ Zone {hazard_type} ajoutée à ({x}, {y}, {z})")
        except ValueError:
            print("Valeurs invalides")
    
    # ==================== Commandes d'export ====================
    
    def do_export(self, arg):
        """
        Exporte les résultats de la mission
        Usage: export [chemin_base]
        """
        if not self._check_init():
            return
        
        base_path = arg if arg else "/tmp/tello_exploration"
        self.mission.export_results(base_path)
        print(f"✓ Résultats exportés vers {base_path}_*")
    
    # ==================== Commandes système ====================
    
    def do_quit(self, arg):
        """Quitte le programme"""
        print("Fermeture...")
        
        if self.mission:
            if self.mission.status == MissionStatus.IN_PROGRESS:
                self.mission.stop_exploration()
            self.mission.controller.disconnect()
        
        print("Au revoir!")
        return True
    
    do_exit = do_quit
    do_q = do_quit
    
    def do_clear(self, arg):
        """Efface l'écran"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
    
    # ==================== Méthodes utilitaires ====================
    
    def _check_init(self) -> bool:
        """Vérifie que le système est initialisé"""
        if self.mission is None:
            print("⚠️  Système non initialisé. Utilisez 'init' d'abord.")
            return False
        return True
    
    def _on_status_change(self, old, new):
        """Callback changement de statut"""
        print(f"\n[STATUS] {old.value} → {new.value}")
    
    def _on_waypoint(self, wp, progress):
        """Callback waypoint atteint"""
        print(f"\n[WAYPOINT] ({wp[0]:.0f}, {wp[1]:.0f}) - {progress:.1f}%")
    
    def _on_hazard(self, pos, hazards):
        """Callback danger détecté"""
        print(f"\n⚠️  [DANGER] Position: ({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f})")
        for alert in hazards['alerts']:
            print(f"    {alert}")
    
    def _on_obstacle(self, obstacle):
        """Callback obstacle détecté"""
        mobile_str = " (MOBILE)" if obstacle.is_mobile else ""
        print(f"\n[OBSTACLE{mobile_str}] ({obstacle.x:.0f}, {obstacle.y:.0f}, {obstacle.z:.0f})")
    
    def emptyline(self):
        """Ne rien faire sur ligne vide"""
        pass
    
    def default(self, line):
        """Commande inconnue"""
        print(f"Commande inconnue: {line}")
        print("Tapez 'help' pour la liste des commandes")


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Système d'exploration Tello EDU pour zones NRBC"
    )
    parser.add_argument(
        '--simulation', '-s',
        action='store_true',
        help="Mode simulation (sans drone réel)"
    )
    parser.add_argument(
        '--auto-init',
        action='store_true',
        help="Initialise automatiquement au démarrage"
    )
    
    args = parser.parse_args()
    
    cli = TelloExplorerCLI(simulation_mode=args.simulation)
    
    if args.auto_init:
        cli.do_init("")
    
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterruption - Fermeture...")
        cli.do_quit("")


if __name__ == "__main__":
    main()
