import numpy as np
from numpy.linalg import norm
import os

from orbits import LEO, GEO, SSO, HEO
from keplerian_params import kepler2cart, cart2kepler, LVLH, MU
from plot_tools import Plotter
from astropy.coordinates import solar_system_ephemeris
# from astropy import units as u
# from astropy.constants import G, M_earth
# from astropy.time import Time, TimeDelta
# from poliastro.bodies import Earth, Moon
# from poliastro.twobody import Orbit
# from poliastro.constants import rho0_earth, H0_earth

# from poliastro.core.elements import rv_pqw, rv2coe
# from poliastro.twobody import Orbit




class EquationsCore:
    def __init__(self) -> None:
        pass
    
    def eq_of_motion(self, t, state):
        # state: (4,6)  -> [x,y,z,vx,vy,vz] dla 4 satelitów
        r = state[:, :3]                      # (4,3)
        v = state[:, 3:6]                     # (4,3)

        r_norms = np.linalg.norm(r, axis=1)   # (4,)
        a = -MU * r / (r_norms[:, None]**3)   # (4,3)
        # print(a)
        derivatives = np.hstack([v, a])       # (4,6)
        return derivatives

    def hamiltonian(self, states):
        pass




def runge_kutta_4(state, t, dt, f):
    # print(state)
    # print(f(t, state))

    k1 = f(t, state)
    k2 = f(t + dt / 2, state + dt / 2 * k1)
    k3 = f(t + dt / 2, state + dt / 2 * k2)
    k4 = f(t + dt, state + dt * k3)
    
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def propagate_orbit_rk4(states, t0, tf, dt, f):
    times = np.arange(t0, tf + dt, dt)
    assert len(times) == states.shape[0], "Mismatch T vs states.shape[0]"
    
    for i in range(1, len(times)):
        states[i, :, :] = runge_kutta_4(states[i-1, :, :], times[i-1], dt, f)
    
    return times, states

def compute_states(state0, t0, tf, dt, eq_of_motion):
    '''Load existing orbits, if None, compute one'''
    def filename_hash(*args):
        import hashlib
        filename = "states"
        s = "".join(str(a) for a in args)
        h = hashlib.md5(s.encode()).hexdigest()
        return filename + h[0:5] + ".npz"

    filename = filename_hash(state0, t0, tf, dt)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)


    # if os.path.exists(path):
    #     print(f"[cache] loading from {path}")
    #     data = np.load(path)
    #     times = data["times"]
    #     states = data["states"]
    
    # else:
    print(f"[compute] computing orbit")
    times, states = propagate_orbit_rk4(state0, t0, tf, dt, eq_of_motion)
    np.savez(path, times=times, states=states)

    return times, states

def initial_formation(t0, tf, dt, sat_separation):
    orbit = GEO
    r_mother, v_mother = kepler2cart(*orbit)

    delta_r =(
        sat_separation * np.array([0.0,  0.0,     0.0   ]),
        sat_separation * np.array([0.5, -0.2887,  0.4082]),
        sat_separation * np.array([0.5,  0.2887, -0.4082]),
        sat_separation * np.array([0.5,  0.5774,  0.0   ])
    )

    state0 = []
    v_dir = v_mother / norm(v_mother)
    for dr in delta_r:
        r_i = r_mother + dr
        v_i =  np.sqrt(MU / norm(r_i)) * v_dir
        # v_i = v_mother.copy()  # Use same velocity as mother satellite

        state0.append(np.hstack([r_i, v_i]))

    state0 = np.asarray(state0, dtype=float)             # (4,6)

    T = int((tf - t0) / dt) + 1
    states = np.zeros((T, 4, 6), dtype=float)            # (T,4,6)
    states[0] = state0
    return states





if __name__ == "__main__":
    t0 = 0
    tf = 60*60*24
    dt = 10     
    m = 100 #kg
    sat_separation = 1000 # km
    states = initial_formation(t0, tf, dt, sat_separation)  # (T,4,6)   time, 4 satellites, x y z vx vy vz
    core = EquationsCore()

    times, states = propagate_orbit_rk4(states, t0, tf, dt, core.eq_of_motion)

    plotter = Plotter()
    
    plotter.animate_formation(states, step=50, lvlh=True)
    plotter.plot_errors(times, states)    
    # # # plotter.plot_orbit_3d_plotly(times, states)
    plotter.plot_orbit_3d(times, states)
    