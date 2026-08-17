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


def symplectic_rk(state, t, h, f, masses, sun_idx=None):
    pass

from scipy.optimize import root
import numpy as np


def gauss_legendre_4(state, t, h, f, masses=None, sun_idx=None):
    sqrt3 = np.sqrt(3.0)

    c1 = 0.5 - sqrt3 / 6
    c2 = 0.5 + sqrt3 / 6

    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6
    a21 = 0.25 + sqrt3 / 6
    a22 = 0.25

    shape = state.shape
    size = state.size

    def equations(K):
        K1 = K[:size].reshape(shape)
        K2 = K[size:].reshape(shape)

        Y1 = state + h * (a11 * K1 + a12 * K2)
        Y2 = state + h * (a21 * K1 + a22 * K2)

        F1 = K1 - f(t + c1 * h, Y1)
        F2 = K2 - f(t + c2 * h, Y2)

        return np.concatenate([F1.ravel(), F2.ravel()])

    # initial guess
    K0 = f(t, state)
    guess = np.concatenate([K0.ravel(), K0.ravel()])

    sol = root(equations, guess)

    if not sol.success:
        raise RuntimeError(f"Gauss-Legendre solver failed: {sol.message}")

    K1 = sol.x[:size].reshape(shape)
    K2 = sol.x[size:].reshape(shape)

    return state + 0.5 * h * (K1 + K2)


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

def yoshida_4(state, t, h, f, masses=None, sun_idx=None):
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

    state = stormer_verlet(state, t, w1 * h, f)
    state = stormer_verlet(state, t + h * w1, w0 * h, f)
    state = stormer_verlet(state, t + h * (w1 + w0), w1 * h, f)
    return state

def midpoint_scheme(state, t, h, f, masses, sun_idx=None):
    return state + h * f(t + h / 2, state + h/2 * f(t, state, masses), masses)