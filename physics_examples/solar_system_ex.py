import numpy as np
from numpy.linalg import norm
from orbits import SUN, JUP, SAT, URA, PLU, NEP
import matplotlib.pyplot as plt
G = 2.95912208286e-4  # AU^3 / (day^2 * Msun)

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

def f(t, state, masses):
    # state: (4,6)  -> [x,y,z,vx,vy,vz] dla 4 satelitów
    r = state[:, :3]                      # (4,3)
    v = state[:, 3:6]                     # (4,3)
    a = acc(state, masses)  # (4,3)

    derivatives = np.hstack([v, a])       # (4,6)
    return derivatives

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
    return a        

def pack_state(bodies):
    return np.array([np.hstack([b.pos, b.vel]) for b in bodies], dtype=float)  # (N,6)

def unpack_traj(states, bodies):
    for i, b in enumerate(bodies):
        b.pos_traj = states[:, i, :3]
        b.vel_traj = states[:, i, 3:]

         # (N,6)

def explicite_euler(state, t, masses, h, f):
    # r = state[:, :3]
    # v = state[:, 3:]
    # a = acc(state, masses)
    # r_new = r + h * v
    # v_new = v + h * a

    # return np.hstack([r_new, v_new])
    state_new = state + h * f(t, state, masses)

    return state_new

def symplectic_euler(state, t, masses, h, f):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    v_new = v + h * a
    r_new = r + h * v_new
    
    return np.hstack([r_new, v_new])

def stormer_verlet(state, t, masses, h, f):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    v_half = v + 0.5 * h * a
    r_new = r + h * v_half
    state_new = np.hstack([r_new, v])
    a_new = acc(state_new, masses)
    v_new = v_half + 0.5 * h * a_new

    return np.hstack([r_new, v_new])

def midpoint_scheme(state, t, masses, h, f):    # f(x,t,m)
    return state + h * f(t + h / 2, state + h/2 * f(t, state, masses), masses)

def runge_kutta_4(state, t, masses, h, f):
    k1 = f(t, state, masses)
    k2 = f(t + h / 2, state + h / 2 * k1, masses)
    k3 = f(t + h / 2, state + h / 2 * k2, masses)
    k4 = f(t + h, state + h * k3, masses)
    
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)



def propagate_orbit(bodies, t0, tf, h, integrator):
    T = int((tf - t0)//h) + 1
    N = len(bodies)
    masses = np.array([b.mass for b in bodies], dtype=float)
    states = np.zeros((T, N, 6), dtype=float)
    states[0] = pack_state(bodies)
    t = t0
    
    for k in range(1, T):
        states[k] = integrator(states[k-1], t, masses, h, f)
        t += h

    unpack_traj(states, bodies)

    return states

def plot_orbit_3d(states, label):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        step=10
        for i in range(6):
            ax.plot(states[::step, i, 0], states[::step, i, 1], states[::step, i, 2], linewidth=1.5)

        ax.set_xlabel('X [AU]', fontsize=10)
        ax.set_ylabel('Y [AU]', fontsize=10)
        ax.set_zlabel('Z [AU]', fontsize=10)
        ax.set_title('Solar system', fontweight='bold')

        # ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.title(label)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    bodies = [SUN, JUP, SAT, URA, NEP, PLU]
    t0 = 0.0
    tf = 1000000
    h = 10

    # states_exp = propagate_orbit(bodies, t0=t0, tf=tf, h=h, integrator=explicite_euler)
    # states_mid = propagate_orbit(bodies, t0=t0, tf=tf, h=h, integrator=midpoint_scheme)
    states_rk4 = propagate_orbit(bodies, t0=t0, tf=tf, h=h, integrator=runge_kutta_4)
    
    # states_sym = propagate_orbit(bodies, t0=t0, tf=tf, h=h, integrator=symplectic_euler)
    states_str = propagate_orbit(bodies, t0=t0, tf=tf, h=h, integrator=stormer_verlet)

    # plot_orbit_3d(states_exp, "Explicite euler h=10")
    # plot_orbit_3d(states_mid, "Midpoint Scheme h=10")
    plot_orbit_3d(states_rk4, "Runge Kutta-4 h=10")

    # plot_orbit_3d(states_sym, "Symplectic Euler h=100")
    plot_orbit_3d(states_str, "Stromer-Verlet h=10")


    # masses = np.array([b.mass for b in bodies], dtype=float)
    # # energies_sym = np.zeros(len(states_sym))
    # # energies_exp = np.zeros(len(states_exp))
    # energies_str = np.zeros(len(states_str))
    # energies_rk4 = np.zeros(len(states_rk4))
    # for i in range(len(states_rk4)):
    #     # energies_exp[i] = hamiltonian(states_exp[i], masses)
    #     # energies_sym[i] = hamiltonian(states_sym[i], masses)
    #     energies_str[i] = hamiltonian(states_str[i], masses)
    #     energies_rk4[i] = hamiltonian(states_rk4[i], masses)
    # # print("ENERGIA DLA SYM. EULER:", energies_sym)
    # # plt.plot(energies_exp)
    # # plt.plot(energies_mid)
    # plt.plot(energies_str)
    # plt.plot(energies_rk4)
    # plt.title("ENERGIA UKLADU")
    # plt.legend(["Stormer Verlet", "RK4"])
    plt.show()