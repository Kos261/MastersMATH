import numpy as np
from orbits import MU, G, RE, J2


def f(t, state, masses=None, central_mass=None, J2_pert=False):
    r = state[:, :3]
    v = state[:, 3:]
    a = _acc_from_positions(r, masses, central_mass, J2_pert)

    derivative = np.hstack([v, a])
    return derivative

def acc(state, masses=None, central_mass=None, J2_pert=False):
    """API kompatybilne ze starymi solverami."""
    r = state[:, :3]
    return _acc_from_positions(r, masses, central_mass, J2_pert)

def _acc_from_positions(r, masses=None, central_mass=None, J2_pert=False):
    if masses is None:
        a = acc_central(r, central_mass)
        if J2_pert:
            a += acc_j2(r)
        return a
    else:
        return acc_nbody(r, masses)

def acc_nbody(r, masses):                         # (N,3)
    N = len(masses)                           #  N+1 bcs Earth 1st
    a = np.zeros((N,3))                       # (N,3)

    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            rij = r[j] - r[i]
            a[i] += G * masses[j] * rij / (np.dot(rij, rij)**1.5)
    return a

def acc_central(r, central_mass):
    # r: (4,3)  -> [x,y,z] dla 4 satelitów
    rn = np.linalg.norm(r, axis=1, keepdims=True)  # (N,1)
    assert np.all(rn > 0), "zero radius encountered"
    mu = MU if central_mass is None else 6.67430e-20 * central_mass
    a = -mu * r / (rn ** 3)
    return a

def acc_j2(r):
    rn = np.linalg.norm(r, axis=1, keepdims=True)      # (N,1)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]                # (N,)
    r2 = (rn.squeeze())**2                              # (N,)
    z2 = z**2                                           # (N,)
    factor = 1.5 * J2 * MU * (RE**2) / (r2**2.5)        # (N,)
    c = (5.0 * z2 / r2 - 1.0)                           # (N,)
    ax = factor * x * c
    ay = factor * y * c
    az = factor * z * (5.0 * z2 / r2 - 3.0)
    return np.column_stack((ax, ay, az))                # (N,3)


def hamiltonian(state, masses):
    r = state[:, :3]         # (N,3)
    v = state[:, 3:]         # (N,3)
    N = len(masses)                             
    a = np.zeros((N,3)) 
    energy = 0

    for i in range(N):
        Ek = 0.0
        Ek += 0.5 * masses[i] * float(np.dot(v[i, :], v[i, :]))

        # Potential: -G * sum_{i<j} m_i m_j / |q_i - q_j|
        Ep = 0.0
        for i in range(N):
            for j in range(N):
                if i == j: 
                    continue
                rij = r[j] - r[i]
                a[i] += G * masses[j] * rij / (np.dot(rij, rij)**1.5)
        energy = Ek + Ep

    return energy