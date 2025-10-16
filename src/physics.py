import numpy as np
from orbits import MU, G, RE, J2


def f(t, state, J2_pert=False):
    assert np.all(np.isfinite(state)), "non finite input state"
    # state: (4,6)  -> [x,y,z,vx,vy,vz] dla 4 satelitów
    r = state[:, :3]                      # (4,3)
    v = state[:, 3:6]                     # (4,3)
    # a = acc(state, masses)  # (4,3)
    rn = np.linalg.norm(r, axis=1, keepdims=True)  # (N,1)
    assert np.all(rn>0), "zero radius encountered"

    #Kepler
    a_kep = -MU * r / (rn**3)
           # (4,6)

    #J2
    a_j2 = 0
    if J2_pert:
        x,y,z = r[:,0], r[:,1], r[:,2]
        r2 = rn.squeeze()**2
        z2 = z**2
        factor = 1.5 * J2 * MU * (RE*RE)/r2**2.5
        ax = x /rn * (5 * z2 / r2 - 1)
        ay = y /rn * (5 * z2 / r2 - 1)
        az = z /rn * (5 * z2 / r2 - 3)
        a_j2 = np.column_stack((ax, ay, az))

    a = a_kep + a_j2
    derivatives = np.hstack([v, a])
    return derivatives

def acc(state, masses):
    r = state[:, :3]                          # (N,3)
    v = state[:, 3:]                          # (N,3)
    N = len(masses)                           #  N+1 bcs Earth 1st
    a = np.zeros((N,3))                       # (N,3)
    
    
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            rij = r[j] - r[i]
            a[i] += G * masses[j] * rij / (np.dot(rij, rij)**1.5)
    return a   

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