from dataclasses import dataclass, field
import numpy as np

@dataclass
class Orbit:
    a: float    # Semimajor Axis, Pół oś wielka ~ 367 km nad pow
    ecc: float  # Eccentricity, Mimośród półoś e = c/a
    inc: float  # Inclination, Inklinacja - wychylenie od równika
    raan: float # RAAN, Omega długość węzła wstępującego RAAN kąt między kierunkiem na punkt Barana
    argp: float # Argument of Perigee, omega argument perycentrum
    nu: float    # True Anomaly, Anomalia prawdziwa
    p: float = field(init=False)

    def __post_init__(self):
        self.p = self.a * (1 - self.ecc ** 2)
        self.P = self.a * (1 + self.ecc ** 2)

# Low Earth Orbit
LEO = Orbit(a=6745.592, ecc=0.001, inc=7.81, raan=100.21, argp=152.83, nu=152.83)

# Sun Synchronous Orbit#dRAAN/dt = 360deg/365d
SSO = Orbit(a=7153.12, ecc=0.001, inc=98.4469, raan=212.741,argp=0.0, nu=0.0)

# GeoSynchronous Orbit
GEO = Orbit(a=42164.0, ecc=0.001, inc=0.0, raan=0.0, argp=0.0, nu=0.0)

# GeoSynchronous transfer Orbit
GTO = Orbit(a=24461.0, ecc=0.7322, inc=19.3, raan=75.864, argp=-90.0, nu=90)

# Molniya
HEO = Orbit(a=26164.0, ecc=0.74, inc=63.4, raan=0.0, argp=0.0, nu=0.0)



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