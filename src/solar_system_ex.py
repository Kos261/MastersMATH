import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
G = 2.95912208286e-4  # AU^3 / (day^2 * Msun)

@dataclass
class Planet:
    mass: float
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    pos_traj: np.ndarray = field(default=None, repr=False)
    vel_traj: np.ndarray = field(default=None, repr=False)


def pack_state(bodies):
    return np.array([np.hstack([b.pos, b.vel]) for b in bodies], dtype=float)  # (N,6)

def unpack_traj(states, bodies):
    for i, b in enumerate(bodies):
        b.pos_traj = states[:, i, :3]
        b.vel_traj = states[:, i, 3:]

def acc(state, masses):
    r = state[:, :3]                          # (N,3)
    v = state[:, 3:]  
    N = len(masses)                        # (N,3)
    a = np.zeros((N,3))                       # (N,3)
    
    
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            rij = r[j] - r[i]
            a[i] += G * masses[j] * rij / (np.dot(rij, rij)**1.5)
    return a                 # (N,6)

def explicite_euler(state, masses, h):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    r_new = r + h * v
    v_new = v + h * a

    return np.hstack([r_new, v_new])


def symplectic_euler(state, masses, h):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    v_new = v + h * a
    r_new = r + h * v_new
    
    return np.hstack([r_new, v_new])

def propagate_orbit(bodies, t0, tf, h, integrator):
    T = int((tf - t0)//h) + 1
    N = len(bodies)
    masses = np.array([b.mass for b in bodies], dtype=float)
    states = np.zeros((T, N, 6), dtype=float)
    states[0] = pack_state(bodies)
    # t = t0
    
    for k in range(1, T):
        states[k] = integrator(states[k-1], masses, h)
        # t += h

    unpack_traj(states, bodies)

    return states

def plot_orbit_3d(states):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        step=10
        for i in range(6):
            ax.plot(states[::step, i, 0], states[::step, i, 1], states[::step, i, 2], linewidth=1.5)

        ax.set_xlabel('X [km]', fontsize=10)
        ax.set_ylabel('Y [km]', fontsize=10)
        ax.set_zlabel('Z [km]', fontsize=10)
        ax.set_title('Solar system', fontweight='bold')

        # ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.show()


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
             np.array([  8.3101420,-16.2901086, -7.2521278]), 
             np.array([0.00354178, 0.00137102, 0.00055029]))
NEP = Planet(0.0000517759138449, 
             np.array([ 11.4707666,-25.7294829,-10.8169456]), 
             np.array([0.00288930, 0.00114527, 0.00039677]))
PLU = Planet(1.0/(1.3e8),        
             np.array([-15.5387357,-25.2225594, -3.1902382]), 
             np.array([0.00276725,-0.00170702,-0.00136504]))

if __name__ == "__main__":
    bodies = [SUN, JUP, SAT, URA, NEP, PLU]
    t0 = 0.0
    tf = 3600*24*1.0
    h = 10
    states_exp = propagate_orbit(bodies, t0=t0, tf=tf, h=h, integrator=explicite_euler)
    states_sym = propagate_orbit(bodies, t0=t0, tf=tf, h=h, integrator=symplectic_euler)

    plot_orbit_3d(states_exp)
    plot_orbit_3d(states_sym)