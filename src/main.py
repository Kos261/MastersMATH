import numpy as np
import os

from initial_conditions import initial_formation_1, initial_formation_4, initial_formation_explosion
from orbits import GEO
from plot_tools import Plotter
from physics import f
from solvers import runge_kutta_4


def propagate_orbit_rk4(states, t0, tf, h, f):
    times = np.arange(t0, tf + h, h)
    assert len(times) == states.shape[0], "Mismatch T vs states.shape[0]"
    
    for i in range(1, len(times)):
        states[i, :, :] = runge_kutta_4(states[i - 1, :, :], times[i - 1], h, f, )
    
    return times, states

def propagate_orbits(state0, t0, tf, dt, eq_of_motion, propagator='rk4', masses=None, central_mass=None, J2_pert=False):
    '''Load existing orbits, if None, compute one'''

    filename = filename_hash(state0, t0, tf, dt)

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

def filename_hash(*args):
    import hashlib
    filename = "states"
    s = "".join(str(a) for a in args)
    h = hashlib.md5(s.encode()).hexdigest()
    return filename + h[0:5] + ".npz"

def load_cached_orbits(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    cached_orbits = np.load(path, allow_pickle=True)
    return cached_orbits

if __name__ == "__main__":
    t0 = 0
    tf = 60*60*24*10_000  #seconds
    h = 10     
    m = 100 #kg
    sat_separation = 1000 # km
    orbit = GEO

    # sol = solve_ivp(fun, [t_start, t_end], y0,
    #                 method='DOP853',  # Bardzo wysoki rząd, świetny do orbit
    #                 rtol=1e-13,  # Wysoka precyzja względna
    #                 atol=1e-13)  # Wysoka precyzja bezwzględna


    ''' 
    (T, num_sat, 6)   time, 4 satellites, x y z vx vy vz 
    '''
    # states = initial_formation_1(t0, tf, h, orbit=orbit)
    # states = initial_formation_4(t0, tf, h, sat_separation,orbit=orbit)
    states = initial_formation_explosion(t0, tf, h, orbit=orbit)
    times, states = propagate_orbit_rk4(states=states, t0=t0, tf=tf, h=h, f=f)

    plotter = Plotter()
    # plotter.animate_formation(states, step=50, lvlh=True)
    # plotter.plot_errors(times, states, sat_separation)
    # # # plotter.plot_orbit_3d_plotly(times, states)
    plotter.plot_orbit_3d(times, states, orbit='GEO')
    