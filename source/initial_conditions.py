import numpy as np
from astropy import units as u
from astropy.constants import G, M_earth


# Stałe
MU = (G * M_earth).to(u.km**3 / u.s**2).value    # geocentryczny parametr grawitacyjny Ziemi
R_EARTH = 6378.0         # earth radius [km]
J2 = 1.08263e-3          # perturbation
r0 = 42157               # geostationary
n = np.sqrt(MU / r0**3)  # [rad/s]

# altitude = 35786.0    # [km]
# r = R_EARTH + altitude
# vel0 = np.sqrt(MU / r) # prędkość kołowa [km/s]

# Parametry symulacji
t0 = 0              # start time [s]
tf = 3600*24        # czas końcowy (2 godziny) [s]
dt = 10             # krok czasowy [s]

sat_separation = 100

# a = np.sqrt(3)/2 * sat_separation
# b = np.sqrt(3)/6 * sat_separation
# c = sat_separation/2

# pos0 = np.array([r, 0, 0, 0, vel0, 0])
# pos1 = np.array([pos0[0] + a, pos0[1] - b, pos0[2] + c, pos0[3], pos0[4], pos0[5]])
# vel1 = 

# pos2 = np.array([pos0[0] + a, pos0[1] + b, pos0[2] - c, pos0[3], pos0[4], pos0[5]])
# pos3 = np.array([pos0[0] + a, pos0[1] + 2*b, pos0[2], pos0[3], pos0[4], pos0[5]])

# pos = np.vstack([pos0, pos1, pos2, pos3])




r_mother = np.array([r0, 0, 0])
v_mother = np.array([0, np.sqrt(MU / r0), 0])

formation = [
    np.array([0, 0, 0]),
    np.array([0.5, -0.2887, 0.4082]),
    np.array([0.5,  0.2887, -0.4082]),
    np.array([0.5,  0.5774, 0])
]
formation = [100 * vec for vec in formation]

pos = []
velocities = []

for dr in formation:
    r_i = r_mother + dr
    dv = np.cross([0, 0, n], dr)  # omega x r
    v_i = v_mother + dv
    pos.append(np.hstack([r_i, v_i]))
    