#!/usr/bin/env python3
"""Module de cartographie optimisé pour Tello EDU - inchangé"""
import numpy as np
import json
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

@dataclass
class MapPoint:
    x: float; y: float; z: float; ground_distance: float
    temperature: float = 25.0
    timestamp: float = field(default_factory=time.time)
    @property
    def ground_altitude(self) -> float: return self.z - self.ground_distance
    def to_dict(self) -> dict:
        return {'x':self.x,'y':self.y,'z':self.z,'ground_distance':self.ground_distance,
                'ground_altitude':self.ground_altitude,'temperature':self.temperature,'timestamp':self.timestamp}

@dataclass
class Obstacle:
    x: float; y: float; z: float; radius: float
    is_mobile: bool = False
    velocity: Tuple[float,float,float] = (0,0,0)
    obstacle_type: str = "unknown"; threat_level: int = 0
    last_seen: float = field(default_factory=time.time); confidence: float = 1.0
    def to_dict(self) -> dict:
        return {'x':self.x,'y':self.y,'z':self.z,'radius':self.radius,'is_mobile':self.is_mobile,
                'velocity':self.velocity,'type':self.obstacle_type,'threat_level':self.threat_level,'confidence':self.confidence}
    def predict_position(self, dt: float):
        if not self.is_mobile: return (self.x,self.y,self.z)
        return (self.x+self.velocity[0]*dt, self.y+self.velocity[1]*dt, self.z+self.velocity[2]*dt)

@dataclass
class ThermalZone:
    x: float; y: float; z: float; radius: float; temperature: float
    is_active: bool = True
    timestamp: float = field(default_factory=time.time)
    def to_dict(self) -> dict:
        return {'x':self.x,'y':self.y,'z':self.z,'radius':self.radius,'temperature':self.temperature,
                'is_active':self.is_active,'timestamp':self.timestamp}

class DualMap:
    def __init__(self, resolution: float = 50.0, size: Tuple[float,float] = (1000,1000)):
        self.resolution = resolution; self.size = size
        self.grid_width = int(size[0]/resolution)+1; self.grid_height = int(size[1]/resolution)+1
        self.altitude_grid = np.full((self.grid_height,self.grid_width), np.nan)
        self.thermal_grid = np.full((self.grid_height,self.grid_width), 25.0)
        self.occupancy_grid = np.full((self.grid_height,self.grid_width), -1, dtype=np.int8)
        self.count_grid = np.zeros((self.grid_height,self.grid_width), dtype=int)
        self.raw_points: List[MapPoint] = []; self.obstacles: List[Obstacle] = []; self.thermal_zones: List[ThermalZone] = []
        self.offset_x = size[0]/2; self.offset_y = size[1]/2
        self.created_at = datetime.now(); self.last_updated = datetime.now()
        logger.info(f"DualMap initialisée: {self.grid_width}x{self.grid_height}")
    def _world_to_grid(self, x, y):
        gx=int((x+self.offset_x)/self.resolution); gy=int((y+self.offset_y)/self.resolution)
        gx=max(0,min(gx,self.grid_width-1)); gy=max(0,min(gy,self.grid_height-1)); return gx,gy
    def _grid_to_world(self, gx, gy):
        return gx*self.resolution-self.offset_x+self.resolution/2, gy*self.resolution-self.offset_y+self.resolution/2
    def add_point(self, x,y,z,ground_distance,temperature=25.0):
        point=MapPoint(x,y,z,ground_distance,temperature); self.raw_points.append(point)
        gx,gy=self._world_to_grid(x,y); ga=point.ground_altitude
        if np.isnan(self.altitude_grid[gy,gx]):
            self.altitude_grid[gy,gx]=ga; self.count_grid[gy,gx]=1
        else:
            n=self.count_grid[gy,gx]; self.altitude_grid[gy,gx]=(self.altitude_grid[gy,gx]*n+ga)/(n+1); self.count_grid[gy,gx]+=1
        a=0.3; self.thermal_grid[gy,gx]=a*temperature+(1-a)*self.thermal_grid[gy,gx]
        self.occupancy_grid[gy,gx]=0; self.last_updated=datetime.now()
    def add_obstacle(self, x,y,z,radius=50.0,is_mobile=False,obstacle_type="unknown",threat_level=0):
        for existing in self.obstacles:
            if math.sqrt((existing.x-x)**2+(existing.y-y)**2)<radius*2:
                if is_mobile:
                    dt=time.time()-existing.last_seen
                    if dt>0: existing.velocity=((x-existing.x)/dt,(y-existing.y)/dt,(z-existing.z)/dt)
                existing.x,existing.y,existing.z=x,y,z; existing.is_mobile=is_mobile; existing.last_seen=time.time()
                existing.obstacle_type=obstacle_type; existing.threat_level=threat_level
                gx,gy=self._world_to_grid(x,y); self.occupancy_grid[gy,gx]=100; return existing
        obs=Obstacle(x,y,z,radius,is_mobile,obstacle_type=obstacle_type,threat_level=threat_level); self.obstacles.append(obs)
        gx,gy=self._world_to_grid(x,y); self.occupancy_grid[gy,gx]=100
        logger.info(f"Obstacle ajouté: ({x:.0f}, {y:.0f}) - {obstacle_type}"); return obs
    def add_thermal_zone(self, x,y,z,radius,temperature,is_active=True):
        zone=ThermalZone(x,y,z,radius,temperature,is_active); self.thermal_zones.append(zone)
        gx,gy=self._world_to_grid(x,y); gr=int(radius/self.resolution)+1
        for dx in range(-gr,gr+1):
            for dy in range(-gr,gr+1):
                if dx**2+dy**2<=gr**2:
                    nx,ny=gx+dx,gy+dy
                    if 0<=nx<self.grid_width and 0<=ny<self.grid_height:
                        self.thermal_grid[ny,nx]=max(self.thermal_grid[ny,nx],temperature)
        logger.info(f"Zone thermique: ({x:.0f}, {y:.0f}) - {temperature:.0f}°C"); return zone
    def get_altitude_at(self,x,y):
        gx,gy=self._world_to_grid(x,y); alt=self.altitude_grid[gy,gx]; return None if np.isnan(alt) else alt
    def get_temperature_at(self,x,y):
        gx,gy=self._world_to_grid(x,y); return self.thermal_grid[gy,gx]
    def is_occupied(self,x,y):
        gx,gy=self._world_to_grid(x,y); return self.occupancy_grid[gy,gx]>50
    def get_exploration_coverage(self):
        return (np.sum(~np.isnan(self.altitude_grid))/(self.grid_width*self.grid_height))*100
    def get_hot_zones_count(self,threshold=50.0): return int(np.sum(self.thermal_grid>threshold))
    def get_statistics(self):
        va=self.altitude_grid[~np.isnan(self.altitude_grid)]
        return {'coverage':self.get_exploration_coverage(),'points_recorded':len(self.raw_points),
                'obstacles_count':len(self.obstacles),'mobile_obstacles':sum(1 for o in self.obstacles if o.is_mobile),
                'thermal_zones':len(self.thermal_zones),
                'altitude':{'min':float(np.min(va)) if len(va)>0 else None,'max':float(np.max(va)) if len(va)>0 else None,'mean':float(np.mean(va)) if len(va)>0 else None},
                'temperature':{'min':float(np.min(self.thermal_grid)),'max':float(np.max(self.thermal_grid)),'mean':float(np.mean(self.thermal_grid)),'hot_cells':self.get_hot_zones_count()}}
    def get_obstacles_near(self,x,y,radius=100.0):
        return [o for o in self.obstacles if math.sqrt((o.x-x)**2+(o.y-y)**2)<radius+o.radius]
    def get_mobile_obstacles(self): return [o for o in self.obstacles if o.is_mobile]
    def get_danger_zones(self):
        d=[]
        for o in self.obstacles:
            if o.threat_level>=3: d.append((o.x,o.y,f"obstacle_{o.obstacle_type}"))
        for z in self.thermal_zones:
            if z.temperature>100: d.append((z.x,z.y,"fire"))
            elif z.temperature>60: d.append((z.x,z.y,"hot_zone"))
        return d
    def to_ascii_map(self, drone_pos=None, show_thermal=False):
        dw=min(60,self.grid_width); dh=min(30,self.grid_height)
        sx=self.grid_width/dw; sy=self.grid_height/dh; lines=["="*(dw+2)]
        lines.append(("CARTE THERMIQUE" if show_thermal else "CARTE D'ALTITUDE").center(dw+2)); lines.append("="*(dw+2))
        grid=self.thermal_grid if show_thermal else self.altitude_grid
        if show_thermal: vmin,vmax=20,150
        else:
            valid=grid[~np.isnan(grid)]
            vmin,vmax=(np.min(valid),np.max(valid)) if len(valid)>0 else (0,1)
        vr=max(vmax-vmin,1)
        for dy in range(dh):
            row=""
            for dx in range(dw):
                gx=int(dx*sx); gy=int(dy*sy)
                if drone_pos:
                    dgx,dgy=self._world_to_grid(drone_pos[0],drone_pos[1])
                    if abs(gx-dgx)<sx and abs(gy-dgy)<sy: row+="D"; continue
                wx,wy=self._grid_to_world(gx,gy); is_obs=False
                for obs in self.obstacles:
                    if abs(obs.x-wx)<self.resolution and abs(obs.y-wy)<self.resolution:
                        row+=("M" if obs.is_mobile else ("F" if obs.obstacle_type=="fire" else "X")); is_obs=True; break
                if is_obs: continue
                val=grid[gy,gx]
                if np.isnan(val) and not show_thermal: row+="·"
                else:
                    lvl=max(0,min(9,int((val-vmin)/vr*9))); row+=str(lvl)
            lines.append("|"+row+"|")
        lines.append("="*(dw+2)); st=self.get_statistics()
        lines.append(f"Couverture: {st['coverage']:.1f}%")
        lines.append(f"Obstacles: {st['obstacles_count']} | Thermique: {st['thermal_zones']}")
        if show_thermal: lines.append(f"Temp: {st['temperature']['min']:.0f}-{st['temperature']['max']:.0f}°C")
        elif st['altitude']['min'] is not None: lines.append(f"Altitude: {st['altitude']['min']:.0f}-{st['altitude']['max']:.0f}cm")
        lines.append("Légende: D=drone | X=obstacle | M=mobile | F=feu | ·=inexploré")
        return "\n".join(lines)
    def export_to_json(self, filepath):
        data={'metadata':{'created_at':self.created_at.isoformat(),'last_updated':self.last_updated.isoformat(),
              'resolution':self.resolution,'size':self.size,'statistics':self.get_statistics()},
              'raw_points':[p.to_dict() for p in self.raw_points[-1000:]],'obstacles':[o.to_dict() for o in self.obstacles],
              'thermal_zones':[z.to_dict() for z in self.thermal_zones],'danger_zones':self.get_danger_zones()}
        with open(filepath,'w') as f: json.dump(data,f,indent=2,default=str)
        logger.info(f"Carte exportée: {filepath}")
    def export_grids(self, base_path):
        np.save(f"{base_path}_altitude.npy",self.altitude_grid); np.save(f"{base_path}_thermal.npy",self.thermal_grid)
        np.save(f"{base_path}_occupancy.npy",self.occupancy_grid); logger.info(f"Grilles exportées: {base_path}_*.npy")

class ExplorationPlanner:
    def __init__(self, dual_map: DualMap, step_size: float = 50.0):
        self.map=dual_map; self.step_size=step_size; self.exploration_path=[]; self.current_index=0
    def generate_snake_pattern(self, width, height, start_x=0, start_y=0):
        wps=[]; nx=int(width/self.step_size); ny=int(height/self.step_size); d=1
        for i in range(ny+1):
            y=start_y-height/2+i*self.step_size
            xr=range(nx+1) if d==1 else range(nx,-1,-1)
            for j in xr: wps.append((start_x-width/2+j*self.step_size,y))
            d*=-1
        self.exploration_path=wps; self.current_index=0; logger.info(f"Pattern snake: {len(wps)} waypoints"); return wps
    def generate_spiral_pattern(self, max_radius, center_x=0, center_y=0):
        wps=[(center_x,center_y)]; angle=0; radius=0; astep=math.pi/4
        while radius<max_radius:
            radius+=self.step_size/(2*math.pi); angle+=astep
            wps.append((center_x+radius*math.cos(angle),center_y+radius*math.sin(angle)))
        self.exploration_path=wps; self.current_index=0; logger.info(f"Pattern spirale: {len(wps)} waypoints"); return wps
    def generate_room_search_pattern(self, width, height):
        wps=[]
        for x in np.arange(-width/2,width/2,self.step_size): wps.append((x,-height/2))
        for y in np.arange(-height/2,height/2,self.step_size): wps.append((width/2,y))
        for x in np.arange(width/2,-width/2,-self.step_size): wps.append((x,height/2))
        for y in np.arange(height/2,-height/2,-self.step_size): wps.append((-width/2,y))
        wps.extend(self.generate_spiral_pattern(max(width,height)/2-self.step_size))
        self.exploration_path=wps; self.current_index=0; logger.info(f"Pattern room search: {len(wps)} waypoints"); return wps
    def get_next_waypoint(self):
        if self.current_index>=len(self.exploration_path): return None
        wp=self.exploration_path[self.current_index]; self.current_index+=1; return wp
    def get_progress(self):
        if len(self.exploration_path)==0: return 0.0
        return (self.current_index/len(self.exploration_path))*100
    def skip_to_waypoint(self, index): self.current_index=max(0,min(index,len(self.exploration_path)))
    def reset(self): self.current_index=0
    def replan_avoiding(self, ox, oy, radius=100):
        new=[wp for wp in self.exploration_path[self.current_index:] if math.sqrt((wp[0]-ox)**2+(wp[1]-oy)**2)>radius]
        self.exploration_path=self.exploration_path[:self.current_index]+new
        logger.info(f"Chemin replanifié, {len(new)} waypoints restants")
