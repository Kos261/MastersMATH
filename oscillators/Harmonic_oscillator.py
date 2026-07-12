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

def plot_err(t, all_states, **kwargs):
    real_x = all_states[0][:, 0]
    real_v = all_states[0][:, 1]

    names = ["REAL SOL", "RK4",  "STORM", "SYM"]

    for name, states in zip(names, all_states):
        if name == "REAL SOL":
            continue
        else:
            err = abs(real_x - states[:, 0])
            plt.plot(t, err, '-', alpha=0.5, label=name)

    h = kwargs.get('h', '?')
    tf = kwargs.get('tf', '?')
    if 'h' in kwargs and 'tf' in kwargs:
        plt.title(f"Error \n(h = {h}, tf = {tf})")
    else:
        plt.title("Error")

    plt.xlabel("Time")
    plt.ylabel("Position error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_states(t, all_states, **kwargs):
    names = ["REAL SOL", "RK4", "STORM", "SYM"]
    fig, axs = plt.subplots(2)

    # for name, states in zip(names[1:], all_states[1:]):
    for name, states in zip(names, all_states):

        if name == "REAL SOL":
            axs[0].plot(t, states[:,0], "--", color='black', label="Real solution")
            axs[1].plot(t, states[:,1], "--", color='black', label="Real solution")
        else:
            axs[0].plot(t, states[:,0],alpha=0.5, label=name)
            axs[1].plot(t, states[:,1],alpha=0.5, label=name)

        axs[0].set_title('Position')
        axs[1].set_title('Velocity')

    h = kwargs.get('h', '?')
    tf = kwargs.get('tf', '?')
    if 'h' in kwargs and 'tf' in kwargs:
        axs[0].set_title(f"Positions \n(h = {h}, tf = {tf})")
        axs[1].set_title(f"Velocities \n(h = {h}, tf = {tf})")
    else:
        axs[0].set_title(f"Positions")
        axs[1].set_title(f"Velocities")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def vfield(point):
    # dx/dt = A x
    x = point[0]
    v = point[1]
    new_x = v
    new_y = -omega ** 2 * x
    return [new_x, new_y]

def plot_phase_portrait(t, all_states, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    names = ["REAL SOL", "RK4", "STORM", "SYM"]

    for name, states in zip(names, all_states):
        x = states[:, 0]
        v = states[:, 1]
        if name == "REAL SOL":
            max_val = max(np.max(np.abs(x)), np.max(np.abs(v))) * 1.5
            ax.plot(x, v, color='red', linewidth=2.5, label="Real Trajectory")
        else:
            ax.plot(x,v, alpha=0.5, label=name)

    xs = np.linspace(-max_val, max_val, 20)
    vs = np.linspace(-max_val, max_val, 20)
    X, V = np.meshgrid(xs, vs)
    dX, dV = vfield((X, V))
    ax.quiver(X, V, dX, dV, color='gray', alpha=0.5, pivot='mid')

    h = kwargs.get('h', '?')
    tf = kwargs.get('tf', '?')
    if 'h' in kwargs and 'tf' in kwargs:
        ax.set_title(f"Phase Portrait of Harmonic Oscillator\n(h = {h}, tf = {tf})")
    else:
        ax.set_title("Phase Portrait of Harmonic Oscillator")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Velocity (v)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    ax.set_aspect('equal', 'box')

    plt.show()

def plot_energy(t, all_states, **kwargs):
    real_x = all_states[0][:, 0]
    real_v = all_states[0][:, 1]
    real_E_p = 0.5 * k * real_x ** 2
    real_E_k = 0.5 * m * real_v ** 2
    real_H = real_E_k + real_E_p

    names = ["RK4", "STORM", "SYM"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(t, real_H, "--", color='black', linewidth=2, label="Analytical Energy")

    for name, states in zip(names, all_states[1:]):
        E_p = 0.5 * k * states[:, 0] ** 2
        E_k = 0.5 * m * states[:, 1] ** 2
        H = E_k + E_p

        axs[0].plot(t, H, '-', alpha=0.5, label=name)

        err = abs(real_H - H)
        axs[1].plot(t, err, '-', alpha=0.5, label=name)

    h = kwargs.get('h', '?')
    tf = kwargs.get('tf', '?')
    if 'h' in kwargs and 'tf' in kwargs:
        axs[0].set_title(f"Hamiltonian (h = {h}, tf = {tf})")
        axs[1].set_title(f'Energy Error (h = {h}, tf = {tf})')
    else:
        axs[0].set_title("Hamiltonian")
        axs[1].set_title('Energy Error')

    axs[0].set_ylabel('E_k + E_p')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].set_title('Energy Error')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Error')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    t0 = 0
    # tf = 1000
    # h = 0.5
    tf, h = 10000, 0.5

    # x0, v0 = 0, 1
    x0, v0 = 1, 0
    A = v0 / omega
    states = init_cond(x0, v0, t0=t0, tf=tf, h=h)

    t, states_rk4 = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=runge_kutta_4)
    # _, states_mid = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=midpoint_scheme)
    _, states_stor = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=stormer_verlet)
    _, states_sym = integrate(states.copy(), t0=t0, tf=tf, h=h, f=f, propagator=symplectic_euler)
    real_x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    real_v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    real_sol = np.array([real_x, real_v]).T

    all_states = [real_sol, states_rk4, states_stor, states_sym]
    # plot_err(t, all_states, h=h, tf=tf)
    # plot_states(t, all_states, h=h, tf=tf)
    plot_energy(t, all_states, h=h, tf=tf)
    plot_phase_portrait(t, all_states, h=h, tf=tf)

if __name__ == "__main__":
    main()