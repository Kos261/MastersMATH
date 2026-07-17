import numpy as np
from nbody.physics import acc
from scipy.integrate import OdeSolver

class MySolver(OdeSolver):
    def __init__(self, **kwargs):
        super(MySolver, self).__init__(**kwargs)


def runge_kutta_4(state, t, h, f, masses=None, sun_idx=None):
    k1 = f(t, state)
    k2 = f(t + h / 2, state + h / 2 * k1)
    k3 = f(t + h / 2, state + h / 2 * k2)
    k4 = f(t + h, state + h * k3)

    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def explicite_euler(state, t, h, f, masses, sun_idx=None):
    state_new = state + h * f(t, state, masses)
    return state_new


def symplectic_euler(state, t, h, f, masses=None, sun_idx=None):
    r = state[:, :3]
    v = state[:, 3:]

    a = f(t, state)[:, 3:]
    v_new = v + h * a
    r_new = r + h * v_new

    return np.hstack([r_new, v_new])


def stormer_verlet(state, t, h, f, masses=None, sun_idx=None):
    r = state[:, :3]
    v = state[:, 3:]

    a = f(t, state)[:, 3:]
    v_half = v + 0.5 * h * a
    r_new = r + h * v_half

    # Estymacja stanu w t+h do policzenia a_new
    state_mid = np.hstack([r_new, v_half])
    a_new = f(t + h, state_mid)[:, 3:]

    v_new = v_half + 0.5 * h * a_new

    return np.hstack([r_new, v_new])


def midpoint_scheme(state, t, h, f, masses, sun_idx=None):
    return state + h * f(t + h / 2, state + h/2 * f(t, state, masses), masses)