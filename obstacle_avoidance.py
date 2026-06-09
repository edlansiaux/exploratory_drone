#!/usr/bin/env python3
"""Système d'évitement d'obstacles optimisé pour Tello EDU - inchangé"""
import numpy as np
import math
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
from collections import deque
logger = logging.getLogger(__name__)

class AvoidanceStrategy(Enum):
    NONE="none"; STOP="stop"; GO_LEFT="go_left"; GO_RIGHT="go_right"; GO_UP="go_up"; GO_DOWN="go_down"
    BACKTRACK="backtrack"; GO_AROUND_LEFT="go_around_left"; GO_AROUND_RIGHT="go_around_right"; EMERGENCY_LAND="emergency_land"

class ThreatLevel(Enum):
    NONE=0; LOW=1; MEDIUM=2; HIGH=3; CRITICAL=4

@dataclass
class DetectedObstacle:
    x: float; y: float; z: float; distance: float
    direction: Tuple[float,float,float]
    obstacle_type: str = "unknown"; confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    def __post_init__(self): self.position_history.append((self.x,self.y,self.z,self.timestamp))
    @property
    def is_mobile(self) -> bool:
        if len(self.position_history)<3: return False
        p=list(self.position_history); tot=0
        for i in range(1,len(p)):
            tot+=math.sqrt((p[i][0]-p[i-1][0])**2+(p[i][1]-p[i-1][1])**2+(p[i][2]-p[i-1][2])**2)
        return tot>15
    def get_velocity(self):
        if len(self.position_history)<2: return (0,0,0)
        p=list(self.position_history); vx=vy=vz=0; c=0
        for i in range(1,len(p)):
            dt=p[i][3]-p[i-1][3]
            if dt>0: vx+=(p[i][0]-p[i-1][0])/dt; vy+=(p[i][1]-p[i-1][1])/dt; vz+=(p[i][2]-p[i-1][2])/dt; c+=1
        return (vx/c,vy/c,vz/c) if c>0 else (0,0,0)
    def predict_position(self, dt):
        v=self.get_velocity(); return (self.x+v[0]*dt,self.y+v[1]*dt,self.z+v[2]*dt)
    def update_position(self, x,y,z,distance):
        self.x,self.y,self.z=x,y,z; self.distance=distance; self.timestamp=time.time()
        self.position_history.append((x,y,z,self.timestamp))

@dataclass
class SafetyZone:
    front: float = 100.0; back: float = 60.0; left: float = 60.0; right: float = 60.0
    above: float = 50.0; below: float = 40.0
    emergency_front: float = 50.0; emergency_sides: float = 30.0; emergency_vertical: float = 30.0

class ObstacleAvoidanceSystem:
    TARGET_REACTION_TIME = 100
    def __init__(self, safety_zone: SafetyZone = None):
        self.safety_zone=safety_zone or SafetyZone(); self.detected_obstacles=[]
        self.is_active=False; self.avoidance_in_progress=False; self.current_strategy=AvoidanceStrategy.NONE
        self.detection_range=250; self.prediction_time=1.5; self.obstacle_timeout=3.0
        self._monitor_thread=None; self._stop_monitoring=threading.Event(); self._monitor_rate=50
        self.last_decision_time=0; self.decision_cooldown=0.1
        self.on_obstacle_detected=None; self.on_collision_imminent=None; self.on_strategy_selected=None
        self.stats={'obstacles_detected':0,'collisions_avoided':0,'avg_reaction_time_ms':0,'reaction_times':deque(maxlen=100)}
        logger.info("ObstacleAvoidanceSystem initialisé")
    def start_monitoring(self):
        if self.is_active: return
        self.is_active=True; self._stop_monitoring.clear()
        self._monitor_thread=threading.Thread(target=self._monitor_loop,daemon=True); self._monitor_thread.start()
        logger.info(f"Monitoring démarré ({self._monitor_rate}Hz)")
    def stop_monitoring(self):
        self.is_active=False; self._stop_monitoring.set()
        if self._monitor_thread: self._monitor_thread.join(timeout=1.0)
        logger.info("Monitoring arrêté")
    def _monitor_loop(self):
        interval=1.0/self._monitor_rate
        while not self._stop_monitoring.is_set():
            start=time.time(); self._cleanup_old_obstacles(); self._update_mobile_predictions()
            time.sleep(max(0,interval-(time.time()-start)))
    def _cleanup_old_obstacles(self):
        c=time.time(); self.detected_obstacles=[o for o in self.detected_obstacles if c-o.timestamp<self.obstacle_timeout]
    def _update_mobile_predictions(self):
        for obs in self.detected_obstacles:
            if obs.is_mobile: pass
    def add_obstacle(self, x,y,z,distance,direction,obstacle_type="unknown"):
        start_time=time.time()
        for obs in self.detected_obstacles:
            if math.sqrt((obs.x-x)**2+(obs.y-y)**2+(obs.z-z)**2)<60:
                obs.update_position(x,y,z,distance); return obs
        new_obs=DetectedObstacle(x,y,z,distance,direction,obstacle_type); self.detected_obstacles.append(new_obs)
        self.stats['obstacles_detected']+=1
        if self.on_obstacle_detected: self.on_obstacle_detected(new_obs)
        rt=(time.time()-start_time)*1000; self.stats['reaction_times'].append(rt)
        if len(self.stats['reaction_times'])>0: self.stats['avg_reaction_time_ms']=sum(self.stats['reaction_times'])/len(self.stats['reaction_times'])
        logger.debug(f"Obstacle détecté: ({x:.0f},{y:.0f},{z:.0f}) dist={distance:.0f}cm [{obstacle_type}]"); return new_obs
    def check_collision_risk(self, dx_,dy_,dz_,tx_,ty_,tz_,drone_yaw=0):
        dx=tx_-dx_; dy=ty_-dy_; dz=tz_-dz_; tl=math.sqrt(dx**2+dy**2+dz**2)
        if tl==0: return False,None,ThreatLevel.NONE
        dx/=tl; dy/=tl; dz/=tl
        closest_obstacle=None; min_distance=float('inf'); threat_level=ThreatLevel.NONE
        for obs in self.detected_obstacles:
            if obs.is_mobile: ox,oy,oz=obs.predict_position(tl/50)
            else: ox,oy,oz=obs.x,obs.y,obs.z
            vx=ox-dx_; vy=oy-dy_; vz=oz-dz_; proj=vx*dx+vy*dy+vz*dz
            if proj<0: closest=(dx_,dy_,dz_)
            elif proj>tl: closest=(tx_,ty_,tz_)
            else: closest=(dx_+proj*dx,dy_+proj*dy,dz_+proj*dz)
            dist=math.sqrt((ox-closest[0])**2+(oy-closest[1])**2+(oz-closest[2])**2)
            safety=self._get_safety_margin(vx,vy,vz,drone_yaw)
            if dist<safety:
                ct=self._evaluate_threat(dist,obs)
                if dist<min_distance: min_distance=dist; closest_obstacle=obs; threat_level=ct
        risk=closest_obstacle is not None
        if risk and threat_level.value>=ThreatLevel.HIGH.value and self.on_collision_imminent:
            self.on_collision_imminent(closest_obstacle,threat_level)
        return risk,closest_obstacle,threat_level
    def _get_safety_margin(self, vx,vy,vz,drone_yaw):
        yr=math.radians(drone_yaw); lx=vx*math.cos(yr)+vy*math.sin(yr); ly=-vx*math.sin(yr)+vy*math.cos(yr)
        if abs(lx)>abs(ly) and abs(lx)>abs(vz): return self.safety_zone.front if lx>0 else self.safety_zone.back
        elif abs(ly)>abs(vz): return self.safety_zone.left if ly<0 else self.safety_zone.right
        else: return self.safety_zone.above if vz>0 else self.safety_zone.below
    def _evaluate_threat(self, distance, obstacle):
        if distance<self.safety_zone.emergency_front: return ThreatLevel.CRITICAL
        if obstacle.is_mobile:
            v=obstacle.get_velocity(); sp=math.sqrt(v[0]**2+v[1]**2+v[2]**2)
            if sp>30: return ThreatLevel.CRITICAL if distance<100 else ThreatLevel.HIGH
        if obstacle.obstacle_type in ["fire","hole"]: return ThreatLevel.CRITICAL
        if distance<self.safety_zone.emergency_front: return ThreatLevel.CRITICAL
        elif distance<self.safety_zone.front*0.5: return ThreatLevel.HIGH
        elif distance<self.safety_zone.front: return ThreatLevel.MEDIUM
        return ThreatLevel.LOW
    def get_avoidance_strategy(self, drone_pos, target_pos, obstacle, available_space=None):
        if time.time()-self.last_decision_time<self.decision_cooldown: return self.current_strategy
        self.last_decision_time=time.time()
        ox=obstacle.x-drone_pos[0]; oy=obstacle.y-drone_pos[1]; oz=obstacle.z-drone_pos[2]
        tx=target_pos[0]-drone_pos[0]; ty=target_pos[1]-drone_pos[1]
        if obstacle.is_mobile:
            v=obstacle.get_velocity(); sp=math.sqrt(v[0]**2+v[1]**2+v[2]**2)
            if sp>30: self.current_strategy=AvoidanceStrategy.STOP; return AvoidanceStrategy.STOP
        if obstacle.distance<self.safety_zone.emergency_front:
            self.stats['collisions_avoided']+=1; self.current_strategy=AvoidanceStrategy.BACKTRACK; return AvoidanceStrategy.BACKTRACK
        if obstacle.obstacle_type in ["fire","hole"]:
            self.stats['collisions_avoided']+=1
            s=(AvoidanceStrategy.GO_UP if oz>0 else AvoidanceStrategy.GO_DOWN) if (abs(oz)>abs(ox) and abs(oz)>abs(oy)) else AvoidanceStrategy.GO_AROUND_RIGHT
            self.current_strategy=s; return s
        cz=tx*oy-ty*ox
        if abs(oz)>abs(ox) and abs(oz)>abs(oy): s=AvoidanceStrategy.GO_DOWN if oz>0 else AvoidanceStrategy.GO_UP
        else: s=AvoidanceStrategy.GO_AROUND_RIGHT if cz>0 else AvoidanceStrategy.GO_AROUND_LEFT
        self.current_strategy=s
        if self.on_strategy_selected: self.on_strategy_selected(s,obstacle)
        return s
    def calculate_avoidance_path(self, drone_pos, target_pos, obstacle, strategy):
        wps=[]; ad=100
        if strategy==AvoidanceStrategy.STOP: return [drone_pos]
        elif strategy==AvoidanceStrategy.BACKTRACK:
            bx=drone_pos[0]-50*(obstacle.x-drone_pos[0])/max(obstacle.distance,1)
            by=drone_pos[1]-50*(obstacle.y-drone_pos[1])/max(obstacle.distance,1); wps=[(bx,by,drone_pos[2])]
        elif strategy in [AvoidanceStrategy.GO_AROUND_LEFT,AvoidanceStrategy.GO_AROUND_RIGHT]:
            mx=(drone_pos[0]+obstacle.x)/2; my=(drone_pos[1]+obstacle.y)/2
            dx=obstacle.x-drone_pos[0]; dy=obstacle.y-drone_pos[1]; L=math.sqrt(dx**2+dy**2)
            if L>0:
                if strategy==AvoidanceStrategy.GO_AROUND_LEFT: px=-dy/L*ad; py=dx/L*ad
                else: px=dy/L*ad; py=-dx/L*ad
                wps=[(mx+px,my+py,drone_pos[2]),target_pos]
            else: wps=[target_pos]
        elif strategy==AvoidanceStrategy.GO_UP:
            wps=[(drone_pos[0],drone_pos[1],drone_pos[2]+ad),(obstacle.x,obstacle.y,obstacle.z+ad+50),target_pos]
        elif strategy==AvoidanceStrategy.GO_DOWN:
            sa=max(40,drone_pos[2]-ad); wps=[(drone_pos[0],drone_pos[1],sa),(obstacle.x,obstacle.y,sa),target_pos]
        else: wps=[target_pos]
        return wps
    def check_immediate_danger(self, front_dist, height, left_clear=True,right_clear=True,up_clear=True,down_clear=True):
        if front_dist<self.safety_zone.emergency_front:
            self.stats['collisions_avoided']+=1; return AvoidanceStrategy.BACKTRACK
        if height<30: return AvoidanceStrategy.GO_UP
        if not up_clear and height>250: return AvoidanceStrategy.GO_DOWN
        return None
    def get_safe_direction(self, drone_pos):
        if not self.detected_obstacles: return None
        rep=[0.0,0.0,0.0]
        for obs in self.detected_obstacles:
            dx=drone_pos[0]-obs.x; dy=drone_pos[1]-obs.y; dz=drone_pos[2]-obs.z
            dist=math.sqrt(dx**2+dy**2+dz**2)
            if dist>0:
                f=1.0/(dist+1)**2; rep[0]+=dx*f/dist; rep[1]+=dy*f/dist; rep[2]+=dz*f/dist
        L=math.sqrt(sum(r**2 for r in rep))
        return tuple(r/L for r in rep) if L>0 else None
    def get_status_report(self):
        return {'is_active':self.is_active,'total_obstacles':len(self.detected_obstacles),
                'mobile_obstacles':sum(1 for o in self.detected_obstacles if o.is_mobile),
                'current_strategy':self.current_strategy.value,'avoidance_in_progress':self.avoidance_in_progress,
                'stats':{'obstacles_detected':self.stats['obstacles_detected'],'collisions_avoided':self.stats['collisions_avoided'],'avg_reaction_time_ms':self.stats['avg_reaction_time_ms']},
                'obstacles':[{'position':(o.x,o.y,o.z),'distance':o.distance,'is_mobile':o.is_mobile,'type':o.obstacle_type} for o in self.detected_obstacles]}

class ReactiveAvoidance:
    def __init__(self, emergency_distance: float = 40.0):
        self.emergency_distance=emergency_distance; self.last_reaction=0; self.cooldown=0.2
        self.reactions_count=0; self.last_reaction_type=None
    def check(self, front_dist, height, left_dist=1000,right_dist=1000,up_dist=1000,down_dist=1000):
        now=time.time()
        if now-self.last_reaction<self.cooldown: return None
        action=None
        if front_dist<self.emergency_distance: action=AvoidanceStrategy.BACKTRACK
        elif height<25: action=AvoidanceStrategy.GO_UP
        elif left_dist<self.emergency_distance*0.7: action=AvoidanceStrategy.GO_RIGHT
        elif right_dist<self.emergency_distance*0.7: action=AvoidanceStrategy.GO_LEFT
        if action: self.last_reaction=now; self.reactions_count+=1; self.last_reaction_type=action
        return action
