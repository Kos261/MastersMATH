import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass

G = 2.95912208286e-4

@dataclass
class Planet:
    mass: float
    pos: np.array
    vel: np.array
    traj_euler_exp: np.array
    traj_euler_imp: np.array
    traj_euler_sym: np.array
    traj_stromer: np.array

    
SUN = Planet(1.000005976782, 
             np.array([0.0, 0.0, 0.0]), 
             np.array([0.0, 0.0, 0.0]))

JUP = Planet(0.000954786104043, 
             np.array([-3.5023653, -3.8169847, -1.5507963]), 
             np.array([0.00565429, -0.00412490, -0.00190589]))

SAT = Planet(0.000285583733151, 
             np.array([ 9.0755314, -3.0458353, -1.6483708]), 
             np.array([0.00168318, 0.00483525, 0.00192462]))

URA = Planet(0.0000437273164546, 
             np.array([  8.3101420, -16.2901086, -7.2521278]), 
             np.array([0.00354178, 0.00137102, 0.00055029]))

NEP = Planet(0.0000517759138449, 
             np.array([ 11.4707666, -25.7294829, -10.8169456]), 
             np.array([0.00288930, 0.00114527, 0.00039677]))

PLU = Planet(1.0/(1.3e8), 
             np.array([-15.5387357, -25.2225594, -3.1902382]), 
             np.array([0.00276725, -0.00170702, -0.00136504]))


def explicit_euler(body, t0, tf, h):
    h=10
    i = 1
    steps = int((tf - t0) // h) + 1
    pos = np.zeros([steps, 3], dtype=float)
    vel = np.zeros([steps, 3], dtype=float)
    pos[0] = body.pos
    vel[0] = body.vel

    return

def implicit_euler():
    h=10

def symplectic_euler():
    h=100

def stormer_verlet():
    h=200

def hamiltonian(bodies):
    # Kinetic: 1/2 * sum m_i * |v_i|^2
    Ek = 0.0
    for b in bodies:
        Ek += 0.5 * b.mass * float(np.dot(b.vel, b.vel))

    # Potential: -G * sum_{i<j} m_i m_j / |q_i - q_j|
    Ep = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i):
            rij = norm(bodies[i].pos - bodies[j].pos)
            Ep -= G * bodies[i].mass * bodies[j].mass / rij

    return Ek + Ep

def plot_orbit_3d(bodies):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for body in bodies:
        ax.plot(body.pos[:, 0], body.pos[:, 1], body.pos[:, 2], linewidth=2)

if __name__ == "__main__":
    t0 = 0
    tf = 3600 * 24 * 200_000 
    # PLANETS = [SUN, JUP, SAT, URA, NEP, PLU]
    PLANETS = [SUN, JUP]
    H = hamiltonian(PLANETS)
    print("Hamiltonian: ",H)



    