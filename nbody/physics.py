import numpy as np
from orbits import MU, G, RE, J2, c


def f(t, state, masses, J2_pert=False, relativistic=False, sun_idx=None):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(r, v, masses=masses, J2_pert=J2_pert, relativistic=relativistic, sun_idx=sun_idx)

    derivative = np.hstack([v, a])
    return derivative

def acc(r, v, masses, J2_pert=False, relativistic=False, sun_idx=None):
    a = acc_nbody(r, masses)
    if J2_pert:
        a += acc_j2(r)
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

# def acc_nbody(r, mus):
#     """
#     r: (N,3), mus: (N,)
#     """
#     diff = r[np.newaxis, :, :] - r[:, np.newaxis, :]      # (N,N,3): r_j - r_i
#     dist3 = np.linalg.norm(diff, axis=2) ** 3             # (N,N)
#     np.fill_diagonal(dist3, np.inf)                        # unikamy 0/0 na przekątnej
#     a = np.einsum('j,ijk->ik', mus, diff / dist3[:, :, None])
#     return a

# def acc_nbody(r, mus):
#     """
#     r: tablica (N, 3), mus: tablica (N,)
#     """
#     mus_1d = np.atleast_1d(mus).flatten()
#
#     diff = r[np.newaxis, :, :] - r[:, np.newaxis, :]  # (N, N, 3): r_j - r_i
#     dist3 = np.linalg.norm(diff, axis=2) ** 3  # (N, N)
#     np.fill_diagonal(dist3, np.inf)  # unikamy 0/0 na przekątnej
#
#     # 2. Obliczamy czynnik masowy: mu_j / |r_ij|^3
#     # mus_1d[np.newaxis, :] ma kształt (1, N), dzieli się przez (N, N) dając macierz (N, N)
#     factor = mus_1d[np.newaxis, :] / dist3
#
#     # 3. Mnożenie wektorów (N, N, 3) przez skalary (N, N, 1) i suma po indeksie j (axis=1)
#     a = np.sum(diff * factor[:, :, np.newaxis], axis=1)
#
#     return a


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


def acc_j2(r):
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)      # (N,1)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]                # (N,)
    r2 = (r_norm.squeeze())**2                              # (N,)
    z2 = z**2                                           # (N,)
    factor = 1.5 * J2 * MU * (RE**2) / (r2**2.5)        # (N,)
    c = (5.0 * z2 / r2 - 1.0)                           # (N,)
    ax = factor * x * c
    ay = factor * y * c
    az = factor * z * (5.0 * z2 / r2 - 3.0)
    return np.column_stack((ax, ay, az))                # (N,3)

# def acc_j2(state):
#     r = state[:, :3]
#     v = state[:, 3:]
#     r_norm = np.linalg.norm(r)
#     v_norm = np.linalg.norm(v)
#
#     a = -r * mu / r_norm ** 3


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


