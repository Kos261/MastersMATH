import numpy as np
from numpy.linalg import norm
import os

from orbits import LEO, GEO, SSO, HEO
from keplerian_params import kepler2cart, cart2kepler, LVLH, MU
from plot_tools import Plotter
from physics import f, hamiltonian


def runge_kutta_4(state, t, h,  f):
    k1 = f(t, state)
    #assert np.all(k1 == 0), "Divisor is zero"
    k2 = f(t + h / 2, state + h / 2 * k1)
    k3 = f(t + h / 2, state + h / 2 * k2)
    k4 = f(t + h, state + h * k3)
    
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def propagate_orbit_rk4(states, t0, tf, h, f):
    times = np.arange(t0, tf + h, h)
    assert len(times) == states.shape[0], "Mismatch T vs states.shape[0]"
    
    for i in range(1, len(times)):
        states[i, :, :] = runge_kutta_4(states[i-1, :, :], times[i-1], h, f,)
    
    return times, states

def compute_states(state0, t0, tf, dt, eq_of_motion, propagator='rk4'):
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

def initial_formation_4(t0, tf, dt, sat_separation, orbit=GEO):
    num_sat = 4
    r_mother, v_mother = kepler2cart(orbit)

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

    state0 = np.asarray(state0, dtype=float)             # (num_sat,6)

    T = int((tf - t0) / dt) + 1
    states = np.zeros((T, num_sat, 6), dtype=float)            # (T, num_sat, 6)
    states[0] = state0
    return states

def initial_formation_explosion(t0, tf, dt, orbit=GEO):
    def sphere_directions_normal(n, rng=None):
        """Jednostajnie po sferze: normal(0,1) i normalizacja."""
        rng = np.random.default_rng(rng)
        v = rng.normal(size=(n, 3))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)  # (n,3) – unitarne
        return v

    def sphere_directions_fibonacci(n):
        """Deterministyczna ‘równomierność’: sfera Fibonacciego (złota spirala)."""
        i = np.arange(n)
        phi = (1 + 5**0.5) / 2  # złota proporcja
        z = 1 - 2*(i + 0.5)/n
        r = np.sqrt(1 - z*z)
        theta = 2*np.pi*i/phi
        x = r*np.cos(theta); y = r*np.sin(theta)
        return np.column_stack([x, y, z])  # już unitarne


    num_sat = 10
    r_mother, v_mother = kepler2cart(orbit)   # (1,3), (1,3)
    state0 = []
    v_explode = sphere_directions_normal(num_sat)    
    for i in range(num_sat):
        r_i = r_mother                          # (3, )
        v_i =  v_mother + v_explode[i] * 0.1   # (3, )
       
        state0.append(np.hstack([r_i, v_i]))    # (6, )

    state0 = np.asarray(state0, dtype=float)             # (num_sat,6)

    T = int((tf - t0) / dt) + 1
    states = np.zeros((T, num_sat, 6), dtype=float)            # (T, num_sat, 6)
    states[0] = state0
    return states

def initial_formation_1(t0, tf, dt, orbit=GEO):
    r_sat, v_sat = kepler2cart(orbit)
    state0 = []
    state0.append(np.hstack([r_sat, v_sat]))
    state0 = np.asarray(state0, dtype=float)  # (1,6)

    T = int((tf - t0) / dt) + 1
    states = np.zeros((T, 1, 6), dtype=float)  # (T, 1, 6)
    states[0] = state0
    return states

if __name__ == "__main__":
    t0 = 0
    tf = 60*60*12
    h = 10     
    m = 100 #kg
    sat_separation = 1000 # km
    orbit = GEO
    ''' 
    (T, num_sat, 6)   time, 4 satellites, x y z vx vy vz 
    '''
    states = initial_formation_1(t0, tf, h, orbit=orbit)
    # states = initial_formation_4(t0, tf, h, sat_separation,orbit=orbit)
    # states = initial_formation_explosion(t0, tf, h, orbit=orbit)
    times, states = propagate_orbit_rk4(states=states, t0=t0, tf=tf, h=h, f=f)

    plotter = Plotter()
    # plotter.animate_formation(states, step=50, lvlh=True)
    # plotter.plot_errors(times, states, sat_separation)    
    # # # plotter.plot_orbit_3d_plotly(times, states)
    plotter.plot_orbit_3d(times, states, orbit='GEO')
    