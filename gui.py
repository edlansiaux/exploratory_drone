# ... (Imports inchangés) ...

class MapPanel(ttk.Frame):
    # ... (__init__ et events inchangés) ...

    def update_map(self, slam_map: np.ndarray = None, 
                   trajectory: list = None,
                   landmarks: dict = None,
                   drone_pose: CameraPose = None):
        """Met à jour l'affichage de la carte"""
        # ... (Création map_img inchangée) ...
            
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
        
        # ... (Reste de l'affichage inchangé) ...
