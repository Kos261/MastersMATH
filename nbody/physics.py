import numpy as np
from multiprocessing import Pool
import copy
from numba import jit, prange
from tqdm import tqdm
from orbits import MU, G, RE,RS, J2, c
from pathlib import Path

def f(t, state, masses, J2_pert=False, relativistic=False, sun_idx=None):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(r, v, masses=masses, J2_pert=J2_pert, relativistic=relativistic, sun_idx=sun_idx)

    derivative = np.hstack([v, a])
    return derivative

def acc(r, v, masses, J2_pert=False, relativistic=False, sun_idx=None):
    a = acc_nbody(r, masses)
    if J2_pert:
        a += acc_j2(r, masses, sun_idx)
    if relativistic:
        a += acc_relativistic(r, v, masses, sun_idx)
    return a


def acc_nbody(r, mus):
    '''
    r: (N, 3) positions [km]
    mus: (N,) gravitational parameters GM [km^3/s^2]
    returns: (N,3) accelerations [km/s^2]
    '''
    N = len(mus)
    a = np.zeros((N,3))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rij = r[j] - r[i]
            a[i] += mus[j] * rij / (np.dot(rij, rij)**1.5)
    return a


def acc_central(r, central_mass):
    # r: (4,3)  -> [x,y,z] dla 4 satelitów
    rn = np.linalg.norm(r, axis=1, keepdims=True)  # (N,1)
    assert np.all(rn > 0), "zero radius encountered"
    mu = MU if central_mass is None else 6.67430e-20 * central_mass
    a = -mu * r / (rn ** 3)
    return a

def acc_relativistic(r, v, mus, sun_idx):

    N = r.shape[0]
    a_rel = np.zeros((N, 3))
    mu_sun = mus[sun_idx]
    r_sun = r[sun_idx]
    v_sun = v[sun_idx]

    for i in range(N):
        if i == sun_idx:
            continue

        dist = r[i] - r_sun
        vrel = v[i] - v_sun
        r_norm = np.linalg.norm(dist)
        v_norm = np.linalg.norm(vrel)

        bracket = (4 * mu_sun / r_norm - v_norm ** 2) * dist + 4 * np.dot(dist, vrel) * vrel
        a_rel[i] = bracket * mu_sun / (c ** 2 * r_norm ** 3)

    return a_rel


def acc_j2(r, mus, sun_idx):
    J2_sun = 2.2e-7
    R_sun = 696000.0
    mu_sun = mus[sun_idx]

    a_j2 = np.zeros_like(r)
    r_sun = r[sun_idx]

    for i in range(len(r)):
        if i == sun_idx:
            continue

        dr = r[i] - r_sun
        x, y, z = dr
        r2 = np.dot(dr, dr)
        rn = np.sqrt(r2)

        factor = 1.5 * J2_sun * mu_sun * R_sun**2 / rn**5

        a_j2[i, 0] = factor * x * (5.0 * z**2 / r2 - 1.0)
        a_j2[i, 1] = factor * y * (5.0 * z**2 / r2 - 1.0)
        a_j2[i, 2] = factor * z * (5.0 * z**2 / r2 - 3.0)

    return a_j2


def hamiltonian(state, mus, J2_pert=False, sun_idx=0):
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

    if J2_pert:
        mu_c = mus[sun_idx]
        r_c = r[sun_idx]

        for i in range(N):
            if i == sun_idx:
                continue

            rij = r[i] - r_c
            rij_norm = np.linalg.norm(rij)
            z = rij[2]

            Ep += (mus[i] * mu_c * J2 * RS**2 / (2 * rij_norm**3) * (3 * z**2 / rij_norm**2 - 1))

    return Ek + Ep


def hamiltonian_series(states, mus, J2_pert=False, Rel=False, sun_idx=0, filename=None):
    """
    states: (T,N,6) - cała trajektoria
    Zwraca: (T,) - Hamiltonian w każdym kroku czasowym
    """

    if filename is not None:
        path = Path(filename)
        if path.exists():
            data = np.load(path)
            return data["H"]

    T = states.shape[0]
    H = np.zeros(T)
    # for t in tqdm(range(T)):
    for t in range(T):
        H[t] = hamiltonian(states[t], mus, J2_pert=J2_pert, sun_idx=sun_idx)

    if filename is not None:
        path = Path(filename)
        np.savez(path, H=H)
    return H


def angular_momentum(state, mus):
    """
    state: (N,6) - JEDEN krok czasowy
    """
    r = state[:, :3]  # (N,3)
    v = state[:, 3:]  # (N,3)
    p = mus[:, None] * v
    N = len(mus)

    L = np.sum(np.cross(r, p), axis=0)

    return L


def angular_momentum_series(states, mus):
    """
    states: (T,N,6) - cała trajektoria
    Zwraca: (T,) - Hamiltonian w każdym kroku czasowym
    """
    T = states.shape[0]
    L = np.zeros((T, 3))
    for t in range(T):
        L[t] = angular_momentum(states[t], mus)
    return L


