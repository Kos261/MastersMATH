import numpy as np



# Stałe
MU = 398600.4418    # geocentryczny parametr grawitacyjny Ziemi [km^3/s^2]
R_EARTH = 6378.0    # promień Ziemi [km]
J2 = 1.08263e-3 


altitude = 500.0    # [km]
r = R_EARTH + altitude
v = np.sqrt(MU / r) # prędkość kołowa [km/s]

# Parametry symulacji
t0 = 0              # start time [s]
tf = 3600*2        # czas końcowy (2 godziny) [s]
dt = 10             # krok czasowy [s]

sat_separation = 100




pos0 = np.array([r, 0, 0, 0, v, 0])  # [x, y, z, vx, vy, vz]

pos1 = np.array([pos0[0] + sat_separation * np.sqrt(3) * 0.5,
                    pos0[1] - sat_separation * np.sqrt(3)/6,
                    pos0[2] + sat_separation * 0.5,
                    pos0[3],pos0[4],pos0[5]])

pos2 = np.array([pos0[0] + sat_separation * np.sqrt(3) * 0.5,
                    pos0[1] + sat_separation * np.sqrt(3)/6,
                    pos0[2] - sat_separation * 0.5,
                    pos0[3],pos0[4],pos0[5]])

pos3 = np.array([pos0[0] + sat_separation * np.sqrt(3) * 0.5,
                    pos0[1] + sat_separation * np.sqrt(3)/3,
                    pos0[2],pos0[3],pos0[4],pos0[5]])

pos = np.vstack([pos0, pos1, pos2, pos3])
