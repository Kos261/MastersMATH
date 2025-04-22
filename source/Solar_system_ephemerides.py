import spiceypy
from spiceypy import furnsh
import os
import numpy as np

def ephemerides(planets,et):
    for planet in planets:
        [state, ltime] = spiceypy.spkezr('PHOEBE', et,'J2000','LT+S','CASSINI')


def initial_position():
    R_earth =   6378.0
    h = 500.0
    r = R_earth + h
    vel = 11.2 #km/s first cosmic velocity
    start_pos = np.array([r, 0.0, 0.0])
    start_vel = np.array([0.0,vel,0.0])
    return start_pos, start_vel

def load_kernels():
    kernel_path = "/home/konstanty/Projects/MastersMATH/kernels"

    kernels = [
        "naif0012.tls",     # leapseconds
        "de430.bsp",        # efemerydy planet
        "pck00010.tpc"      # parametry fizyczne
    ]    

    for kernel in kernels:
        print(f"Loaded {kernel}")
        furnsh(os.path.join(kernel_path, kernel))
        

if __name__ == "__main__":
    utctime = "2025 apr 11 19:32:00"
    start_et = spiceypy.str2et(utctime)
    planets = ["SUN", "EARTH", "MARS"]


    load_kernels(planets, start_et)

    (x0, y0, z0), (vx0, vy0, vz0) = initial_position()
    times = np.linspace(start_et, start_et + 86400, num=1441)  # co 60 sekund

    time_pos_vel_vec = np.array([(start_et, x0, y0, z0, vx0, vy0, vz0)])