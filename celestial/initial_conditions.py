import numpy as np
from numpy.linalg import norm

from nbody.orbits import GEO, kepler2cart, MU


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
