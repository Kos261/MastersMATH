import numpy as np
from physics import acc

def runge_kutta_4(state, t, h, f):
    k1 = f(t, state)
    #assert np.all(k1 == 0), "Divisor is zero"
    k2 = f(t + h / 2, state + h / 2 * k1)
    k3 = f(t + h / 2, state + h / 2 * k2)
    k4 = f(t + h, state + h * k3)

    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def explicite_euler(state, t, masses, h, f):
    state_new = state + h * f(t, state, masses)
    return state_new


def symplectic_euler(state, t, masses, h, f):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    v_new = v + h * a
    r_new = r + h * v_new

    return np.hstack([r_new, v_new])



def stormer_verlet(state, t, masses, h, f):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    v_half = v + 0.5 * h * a
    r_new = r + h * v_half
    state_new = np.hstack([r_new, v])
    a_new = acc(state_new, masses)
    v_new = v_half + 0.5 * h * a_new

    return np.hstack([r_new, v_new])


def midpoint_scheme(state, t, masses, h, f):    # f(x,t,m)
    return state + h * f(t + h / 2, state + h/2 * f(t, state, masses), masses)
