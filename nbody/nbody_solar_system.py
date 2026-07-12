import numpy as np
import spiceypy as sp
import matplotlib.pyplot as plt

from nbody.physics import f
from solvers import *
from load_ephemeris import *


def propagate_orbit(state0, mus, t0, tf, h, f, integrator, sun_idx=0):
    times = np.arange(t0, tf + h, h)
    T = len(times)
    N = state0.shape[0]
    states = np.zeros((T, N, 6))
    states[0] = state0

    for i in range(1, T):
        states[i] = integrator(state=states[i - 1],
                               t=times[i - 1],
                               h=h,
                               f=lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx),
                               masses=mus,
                               sun_idx=sun_idx)
        if np.any(np.isnan(states[i])):
            print(f"NaN pojawił się w kroku {i}, t = {times[i]}")
            break
    return times, states


def hamiltonian(state, mus):
    """
    state: (N,6) - JEDEN krok czasowy
    mus:   (N,)  - GM każdego ciała
    """
    r = state[:, :3]  # (N,3)
    v = state[:, 3:]  # (N,3)
    N = len(mus)

    mus = mus.astype(np.float64)

    v_squared = np.sum(v * v, axis=1)  # (N,)
    Ek = 0.5 * np.dot(mus, v_squared)

    Ep = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = r[j] - r[i]
            r_ij = np.sqrt(np.dot(rij, rij))
            Ep -= mus[i] * mus[j] / r_ij

    return Ek + Ep


def hamiltonian_series(states, mus):
    """
    states: (T,N,6) - cała trajektoria
    Zwraca: (T,) - Hamiltonian w każdym kroku czasowym
    """
    T = states.shape[0]
    H = np.zeros(T)
    for t in range(T):
        H[t] = hamiltonian(states[t], mus)
    return H


def add_asteroid_belt_torus(ax, R_au=2.7, tube_radial_au=0.6, tube_vertical_au=0.15):
    AU = 149_597_870.7  # km
    R = R_au * AU
    r_rad = tube_radial_au * AU
    r_vert = tube_vertical_au * AU

    theta = np.linspace(0, 2*np.pi, 60)   # dookoła przekroju "rury"
    phi = np.linspace(0, 2*np.pi, 60)     # dookoła głównego okręgu
    theta, phi = np.meshgrid(theta, phi)

    x = (R + r_rad * np.cos(theta)) * np.cos(phi)
    y = (R + r_rad * np.cos(theta)) * np.sin(phi)
    z = r_vert * np.sin(theta)

    ax.plot_surface(x, y, z, color='gray', alpha=0.15, linewidth=0, antialiased=True)

def plot_comparison(names, num_planets, states, truth):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, name in enumerate(names):  # tylko planety
        ax.plot(states[:, i, 0], states[:, i, 1], states[:, i, 2], label=name)
        ax.plot(truth[:, i, 0], truth[:, i, 1], truth[:, i, 2],
                linestyle='--', color='black', linewidth=0.8, alpha=0.5)

    add_asteroid_belt_torus(ax)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend(fontsize=7)
    plt.title("N-body simulation vs ephemeris (+ asteroid belt)")
    plt.show()


def plot_errors(names, states, truth, times, t0):
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, name in enumerate(names):
        err = np.linalg.norm(states[:, i, :3] - truth[:, i, :3], axis=1)
        ax.plot((times - t0) / (24 * 60 * 60), err, label=name)  # seconds to day

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Position error [km]')
    # ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.title("Position error vs ephemeris")
    plt.show()


def plot_energy(t, H, real_H, **kwargs):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(t, real_H, "--", color='black', alpha=0.5, linewidth=2, label="Analytical H")
    axs[0].plot(t, H, '-', label="Hamiltonian")

    err = abs(real_H - H)
    axs[1].plot(t, err, '-', alpha=0.5, label="ERROR")

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

    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Error')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/de442s.bsp')
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/naif0012.tls')
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/sb441-n16.bsp')
        # inspect_kernel(r'/home/konstanty/Pulpit/MastersMATH/data/sb441-n16.bsp')
        # inspect_kernel(r'/home/konstanty/Pulpit/MastersMATH/data/de442s.bsp')

        gm_by_number = parse_asteroid_gm(read_bsp_comments(
            r'/home/konstanty/Pulpit/MastersMATH/data/de442s.bsp'
        ))

        t0_str = "2000-07-01 00:00:00"
        tf_str = "2026-07-01 00:00:00"
        h = float(60 * 60 * 24)
        et0 = sp.str2et(t0_str)
        etf = sp.str2et(tf_str)

        names, num_planets, state0, mus = load_initial_states_with_asteroids(
            et0,
            r'/home/konstanty/Pulpit/MastersMATH/data/sb441-n16.bsp',
            gm_by_number
        )

        times, states = propagate_orbit(state0=state0, mus=mus, t0=et0, tf=etf,
                                          h=h, f=f, integrator=runge_kutta_4)

        truth = get_true_ephemeris(names, times)  # tylko dla planet

        plot_comparison(names, num_planets, states[:, :num_planets, :], truth)
        plot_errors(names, states[:, :num_planets, :], truth, times, et0)

    finally:
        sp.kclear()