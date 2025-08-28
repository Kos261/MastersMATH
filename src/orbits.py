# from poliastro.twobody import Orbit
# from poliastro.bodies import Earth
# from astropy import units as u
# from astropy.time import Time
from dataclasses import dataclass, field
import numpy as np
# epoch = Time(2454283.0, format="jd", scale="tdb") 
R_EARTH = 6378.0 
# Low Earth Orbit
LEO = (
    6745.592,       # Semimajor Axis, Pół oś wielka ~ 367 km nad pow
    0.01,           # Eccentricity, Mimośród półoś mała / półoś wielka
    7.81,           # Inclination, Inklinacja - wychylenie od równika
    100.21,         # RAAN, Omega długość węzła wstępującego RAAN kąt między kierunkiem na punkt Barana
    152.83,         # Argument of Perigee, omega argument perycentrum
    0.0 ,           # True Anomaly, Anomalia prawdziwa 
    )

# Sun Synchronous Orbit
SSO = (
    7153.12,           
    0.00,        
    98.4469 ,       
    212.741,
    0.0,     
    0.0,        
    )

# GeoSynchronous Orbit
GEO = (
    42164.0,           
    0.001,       
    0.0,       
    0.0,
    0.0,     
    0.0,        
    )

# Molniya
HEO = (
    26164.0,           
    0.74,       
    63.4,       
    0.0,
    0.0,     
    0.0,
    )


@dataclass
class Planet:
    mass: float
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    pos_traj: np.ndarray = field(default=None, repr=False)
    vel_traj: np.ndarray = field(default=None, repr=False)

SUN = Planet(1.000005976782, 
             np.array([0.0, 0.0, 0.0]), 
             np.array([0.0, 0.0, 0.0]))
JUP = Planet(0.000954786104043, 
             np.array([-3.5023653, -3.8169847, -1.5507963]), 
             np.array([0.00565429, -0.00412490, -0.00190589]))
SAT = Planet(0.000285583733151, 
             np.array([ 9.0755314, -3.0458353, -1.6483708]), 
             np.array([0.00168318, 0.00483525, 0.00192462]))
URA = Planet(0.0000437273164546, 
             np.array([  8.3101420,-16.2901086, -7.2521278]), 
             np.array([0.00354178, 0.00137102, 0.00055029]))
NEP = Planet(0.0000517759138449, 
             np.array([ 11.4707666,-25.7294829,-10.8169456]), 
             np.array([0.00288930, 0.00114527, 0.00039677]))
PLU = Planet(1.0/(1.3e8),        
             np.array([-15.5387357,-25.2225594, -3.1902382]), 
             np.array([0.00276725,-0.00170702,-0.00136504]))