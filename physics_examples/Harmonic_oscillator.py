import numpy as np
import matplotlib.pyplot as plt

k = 1
m = 1

def runge_kutta_4(state, t, h, f):
    k1 = f(t, state)
    # assert np.all(k1 == 0), "Divisor is zero"
    k2 = f(t + h / 2, state + h / 2 * k1)
    k3 = f(t + h / 2, state + h / 2 * k2)
    k4 = f(t + h, state + h * k3)

    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def symplectic_euler(state, t, h, f):
    r = state[0]
    v = state[1]
    a = acc(t, state)
    v_new = v + h * a
    r_new = r + h * v_new

    return np.hstack([r_new, v_new])

def stormer_verlet(state, t, h, f):
    r = state[0]
    v = state[1]
    a = acc(t, state)
    v_half = v + 0.5 * h * a
    r_new = r + h * v_half
    state_new = np.hstack([r_new, v])
    a_new = acc(t, state_new)
    v_new = v_half + 0.5 * h * a_new

    return np.hstack([r_new, v_new])

def midpoint_scheme(state, t, h, f):    # f(x,t,m)
    return state + h * f(t + h / 2, state + h/2 * f(t, state, ))

def f(t, x):
    return np.array([[0,1],[-k/m, 0]]) @ x

def acc(t, state):
    r = state[0]
    a = -k/m * r
    return a

def integrate(states, t0, tf, f, h, propagator=runge_kutta_4):
    times = np.linspace(t0, tf, len(states))

    for i in range(1, len(times)):
        states[i, :] = propagator(states[i - 1, :], times[i - 1], h, f)

    return times, states

def init_cond(r0, v0, t0, tf, h):
    T = int((tf - t0) / h) + 1
    states = np.zeros((T, 2), dtype=float)
    states[0] = [r0, v0]
    return states

def plot(t, all_states):
    real_sol = np.sin(t)
    # plt.plot(t, real_sol, "--", color='red', alpha=0.5, label="Real solution")
    names = ["RK4", "MID", "STORM", "SYM"]
    for name, states in zip(names, all_states):
        err = real_sol - states[:, 0]
        plt.plot(t, err, '-', label=name)

    plt.title("Harmonic oscillator error")
    plt.xlabel("Time")
    plt.ylabel("Position [Y]")
    plt.legend()
    plt.show()

def main():
    t0 = 0
    tf = 60*60
    h = 0.1
    states = init_cond(0,1, t0 = t0, tf = tf, h = h)
    t, states_rk4 = integrate(states=states, t0=t0, tf=tf, h=h, f=f, propagator=runge_kutta_4)
    _, states_mid = integrate(states=states, t0=t0, tf=tf, h=h, f=f, propagator=midpoint_scheme)
    _, states_stor = integrate(states=states, t0=t0, tf=tf, h=h, f=f, propagator=stormer_verlet)
    _, states_sym = integrate(states=states, t0=t0, tf=tf, h=h, f=f, propagator=symplectic_euler)
    all_states = [states_rk4, states_mid, states_stor, states_sym]

    plot(t, all_states)

if __name__ == "__main__":
    main()