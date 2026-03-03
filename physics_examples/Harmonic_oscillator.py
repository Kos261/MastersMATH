import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

k = 1   #elasticity
m = 1 #mass
omega = sqrt(k / m)
A = 1   #amplitude
phi = 0 #phase
# x(t) = A * sin(omega * t + phi)

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

def plot_err(t, all_states):
    real_x, real_v= all_states[0][0], all_states[0][1]
    names = ["REAL SOL", "RK4", "MID", "STORM", "SYM"]

    for name, states in zip(names, all_states):
        if name == "REAL SOL":
            continue
        else:
            err = real_x - states[:, 0]
            plt.plot(t, err, '-', label=name)

    plt.title("Harmonic oscillator error")
    plt.xlabel("Time")
    plt.ylabel("Position error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_states(t, all_states):
    names = ["REAL SOL", "RK4", "MID", "STORM", "SYM"]
    fig, axs = plt.subplots(2)

    for name, states in zip(names, all_states[1:]):
        if name == "REAL SOL":
            axs[0].plot(t, states[:,0], "--", color='red', alpha=0.5, label="Real solution")
            axs[1].plot(t, states[:,1], "--", color='red', alpha=0.5, label="Real solution")
        else:
            axs[0].plot(t, states[:,0])
            axs[1].plot(t, states[:,1], 'tab:orange')

        axs[0].set_title('Position')
        axs[1].set_title('Velocity')

    plt.legend()
    plt.show()

def plot_energy(t, all_states):
    real_x = all_states[0][:, 0]
    real_v = all_states[0][:, 1]
    real_E_p = 0.5 * k * real_x ** 2
    real_E_k = 0.5 * m * real_v ** 2
    real_H = real_E_k + real_E_p

    names = ["RK4", "MID", "STORM", "SYM"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(t, real_H, "--", color='black', alpha=0.8, linewidth=2, label="Analytical Energy")

    for name, states in zip(names, all_states[1:]):
        E_p = 0.5 * k * states[:, 0] ** 2
        E_k = 0.5 * m * states[:, 1] ** 2
        H = E_k + E_p

        axs[0].plot(t, H, '-', label=name)

        err = real_H - H
        axs[1].plot(t, err, '-', label=name)


    axs[0].set_title('Harmonic oscillator: Total Energy (Hamiltonian)')
    axs[0].set_ylabel('E_k + E_p')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].set_title('Energy Error (Analytical - Numerical)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Error')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    t0 = 0
    tf = 200
    h = 0.5
    x0, v0 = 0, 1
    A = v0 / omega
    states = init_cond(x0, v0, t0=t0, tf=tf, h=h)

    t, states_rk4 = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=runge_kutta_4)
    _, states_mid = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=midpoint_scheme)
    _, states_stor = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=stormer_verlet)
    _, states_sym = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=symplectic_euler)

    real_sol = np.array([A * np.sin(omega * t + phi), A * omega * np.cos(omega * t + phi)]).T

    all_states = [real_sol, states_rk4, states_mid, states_stor, states_sym]

    # plot_err(t, all_states)
    plot_states(t, all_states)
    plot_energy(t, all_states)

if __name__ == "__main__":
    main()