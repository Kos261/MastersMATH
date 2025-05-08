import numpy as np

from astropy import units as u
from astropy.constants import G, M_earth
from astropy.time import Time, TimeDelta
from astropy.coordinates import solar_system_ephemeris

from poliastro.bodies import Earth, Moon
from poliastro.twobody import Orbit
from poliastro.constants import rho0_earth, H0_earth


# CONSTANTS
MU = (G * M_earth).to(u.km**3 / u.s**2).value
R_EARTH = 6378.0         # earth radius [km]
J2 = 1.08263e-3          # perturbation


# parameters of a body
C_D = 2.2  # dimentionless (any value would do)
A_over_m = ((np.pi / 4.0) * (u.m**2) / (100 * u.kg)).to_value(u.km**2 / u.kg)  # km^2/kg
B = C_D * A_over_m

# parameters of the atmosphere
rho0 = rho0_earth.to(u.kg / u.km**3).value  # kg/km^3
H0 = H0_earth.to(u.km).value

tofs = TimeDelta(np.linspace(0 * u.h, 100000 * u.s, num=2000))
solar_system_ephemeris.set("de432s")

epoch = Time(2454283.0, format="jd", scale="tdb")  



# SIMULATION PARAMS
t0 = 0                   # start time [s]
tf = 3600*24             # time finish (1h) [s]
dt = 10                  # [s]

altitude = 35786.0       # geostationary orbit [km]
r0 = (R_EARTH + altitude) * u.km   
n = np.sqrt(MU / r0**3).value  # [rad/s] 
vel0 = np.sqrt(MU / r0)   # orbital velocity [km/s]

sat_separation = 100


def formation():

    initial = Orbit.from_classical(
    Earth,
    r0,   # Pół oś wielka ~ 35 786 km nad pow
    0.0001 * u.one,   # Mimośród półoś mała / półoś wielka
    30 * u.deg,        # Inklinacja - wychylenie od równika
    0.0 * u.deg,      # Omega długość węzła wstępującego RAAN kąt między kierunkiem na punkt Barana
    0.0 * u.deg,      # omega argument perycentrum
    0.0 * u.rad,      # Anomalia prawdziwa 
    epoch=epoch,
    )


    # FORMATION
    # r_mother = np.array([r0, 0, 0])
    # v_mother = np.array([0, vel0, 0])
    r_mother, v_mother = initial.rv()
    r_mother, v_mother = r_mother.value, v_mother.value


    delta_r =[
        sat_separation * np.array([0, 0, 0]),
        sat_separation * np.array([0.5, -0.2887, 0.4082]),
        sat_separation * np.array([0.5,  0.2887, -0.4082]),
        sat_separation * np.array([0.5,  0.5774, 0])
    ]

    orbits = []
    states = []  #[x,y,z,vx,vy,vz]

    for dr in delta_r:
        r_i = r_mother + dr
        dv = np.cross([0, 0, n], dr)  # omega x r
        v_i = v_mother + dv                             
        '''TU ZROBIŁEM 5 * vi'''
        sat_i = Orbit.from_vectors(Earth, r_i * u.km, 5 * v_i * u.km / u.s, epoch=epoch)
        orbits.append(sat_i)
        states.append(np.hstack([r_i, v_i]))

    # print(states)
    return states