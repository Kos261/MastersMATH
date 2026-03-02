import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.initial_conditions import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def equations_of_motion(t, states):
    """Równania ruchu z uwzględnieniem J2"""
    velocities = states[:, 3:6]  # shape (N, 3)
    accelerations = lagrange(states)  # shape (N, 3)
    derivatives = np.hstack([velocities, accelerations])  # shape (N, 6)
    return derivatives

def runge_kutta_4(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt/2 * k1)
    k3 = f(t + dt/2, state + dt/2 * k2)
    k4 = f(t + dt, state + dt * k3)
    
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def propagate_orbit_rk4(state0, t0, tf, dt):
    """Propagacja orbity z użyciem RK4"""
    times = np.arange(t0, tf, dt)
    states = np.zeros((len(times), 4, 6))
    states[0] = state0
    # print(states[0])
    
    for i in range(1, len(times)):
        states[i] = runge_kutta_4(equations_of_motion, times[i-1], states[i-1], dt)
    
    return times, states

def plot_orbit_3d(times, states):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    u = np.linspace(0, 2*np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.5)
    
    colors = ['red', 'green', 'blue', 'yellow']
    labels = ['Mothership', 'Sat1', 'Sat2', 'Sat3']
    step = 50
    for i in range(4):
        ax.plot(states[::step, i, 0], 
                states[::step, i, 1], 
                states[::step, i, 2], 
                color=colors[i], 
                label=labels[i], 
                linewidth=1.5)
    
    # Punkty startowe
    ax.scatter(states[0, :, 0], states[0, :, 1], states[0, :, 2], 
               c='black', s=50, marker='o', depthshade=False, label='Start')
    
    # Konfiguracja wykresu
    ax.set_xlabel('X [km]', fontsize=10)
    ax.set_ylabel('Y [km]', fontsize=10)
    ax.set_zlabel('Z [km]', fontsize=10)
    ax.set_title('Symulacja formacji satelitów (RK4)', fontweight='bold')
    
    # Ustawienie jednakowej skali osi
    max_range = np.max(np.abs(states[:, :, :3])) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Legenda i siatka
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    line_step = 1000
    for t_idx in range(0, len(times), line_step):
        positions = states[t_idx]  # shape (4, 3)
        for i, j in combinations(range(4), 2):
            x = [positions[i, 0], positions[j, 0]]
            y = [positions[i, 1], positions[j, 1]]
            z = [positions[i, 2], positions[j, 2]]
            ax.plot(x, y, z, color='black', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_errors(times, states, sat_separation):
    positions = states[:, :, :3]  # shape (n_steps, 4, 3)
    n_steps = len(times)
    
    dist_errors = np.zeros((n_steps, 4))  # 4 satellite pairs
    
    dist_errors[:, 0] = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1) - sat_separation
    dist_errors[:, 1] = np.linalg.norm(positions[:, 1] - positions[:, 2], axis=1) - sat_separation
    dist_errors[:, 2] = np.linalg.norm(positions[:, 2] - positions[:, 3], axis=1) - sat_separation
    dist_errors[:, 3] = np.linalg.norm(positions[:, 3] - positions[:, 0], axis=1) - sat_separation
    
    # Plot settings
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Satellite Formation Distance Errors (Target: {sat_separation} km)", y=1.02)
    
    pair_names = ['Sat1-Sat2', 'Sat2-Sat3', 'Sat3-Sat4', 'Sat4-Sat1']
    colors = ['b', 'g', 'r', 'm']
    
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(times, dist_errors[:, i], color=colors[i], label=pair_names[i])
        plt.ylabel('Error (m)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Add horizontal line at zero for reference
        plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
        
        if i == 3:
            plt.xlabel('Time (s)')
        else:
            plt.tick_params(labelbottom=False)
    
    plt.tight_layout()
    plt.show()

def lagrange(states):
    positions = states[:, :3]  # shape (N, 3)
    velocities = states[:, 3:6]  # shape (N, 3)
    r_norms = np.linalg.norm(positions, axis=1)  # shape (N,)
    
    # Two-body acceleration
    accel_2body = -MU * positions / r_norms[:, np.newaxis]**3  # shape (N, 3)
    
    # J2 perturbation
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    z_over_r_squared = (z/r_norms)**2  # shape (N,)
    
    common_factor = (3/2) * J2 * MU * R_EARTH**2 / r_norms**5  # shape (N,)
    
    accel_J2 = np.empty_like(positions)  # shape (N, 3)
    accel_J2[:, 0] = common_factor * x * (5*z_over_r_squared - 1)
    accel_J2[:, 1] = common_factor * y * (5*z_over_r_squared - 1)
    accel_J2[:, 2] = common_factor * z * (5*z_over_r_squared - 3)
    
    return accel_2body + accel_J2  # shape (N, 3)


def animate_formation(states, step=1000, save=False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # konfiguracja osi
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("Ewolucja czworościanu w układzie statku-matki")

    # początkowe linie między satelitami
    lines = []
    for _ in combinations(range(4), 2):
        line, = ax.plot([], [], [], color='black', linewidth=1)
        lines.append(line)

    def update(frame):
        t_idx = frame * step
        if t_idx >= len(states):
            return lines

        pos = states[t_idx]
        ref = pos[0]  # pozycja statku-matki
        rel_pos = pos - ref  # przesunięcie względem statku-matki

        for k, (i, j) in enumerate(combinations(range(4), 2)):
            x = [rel_pos[i, 0], rel_pos[j, 0]]
            y = [rel_pos[i, 1], rel_pos[j, 1]]
            z = [rel_pos[i, 2], rel_pos[j, 2]]
            lines[k].set_data(x, y)
            lines[k].set_3d_properties(z)
        return lines

    frames = len(states) // step
    anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=100)

    if save:
        anim.save("formation_evolution.mp4", fps=5)
    else:
        plt.show()

def animate_formation_lvlh(rel_pos, step=1000, save=False):
    """
    Animuje ewolucję formacji 3 satelitów w układzie LVLH (statku-matki jako centrum).
    rel_pos: ndarray shape (T, 3, 3) — pozycje 3 satelitów w czasie w układzie LVLH
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # konfiguracja osi
    ax.set_xlim([-400, 400])
    ax.set_ylim([-400, 400])
    ax.set_zlim([-400, 400])
    ax.set_xlabel("X [km] (prostopadle do orbity)")
    ax.set_ylabel("Y [km] (wzdłuż orbity)")
    ax.set_zlabel("Z [km] (do Ziemi)")
    ax.set_title("Formacja satelitów w układzie LVLH")

    # inicjalizacja linii między satelitami (3 satelity = 3 krawędzie)
    lines = []
    for _ in combinations(range(3), 2):
        line, = ax.plot([], [], [], color='black', linewidth=1.2)
        lines.append(line)

    def update(frame):
        t_idx = frame * step
        if t_idx >= len(rel_pos):
            return lines

        pos = rel_pos[t_idx]  # shape (3, 3)

        for k, (i, j) in enumerate(combinations(range(3), 2)):
            x = [pos[i, 0], pos[j, 0]]
            y = [pos[i, 1], pos[j, 1]]
            z = [pos[i, 2], pos[j, 2]]
            lines[k].set_data(x, y)
            lines[k].set_3d_properties(z)
        return lines

    frames = len(rel_pos) // step
    anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=100)

    if save:
        anim.save("formation_LVLH.mp4", fps=5)
    else:
        plt.tight_layout()
        plt.show()

def to_mothership_ref_frame(states):
    """
    Transformuje pozycje satelitów do układu odniesienia statku matki (LVLH).
    Zakłada, że:
        - states ma shape (T, 4, 6), gdzie 4 to liczba satelitów
        - statek-matka to satelita o indeksie 0
    Zwraca pozycje pozostałych satelitów (1–3) w układzie LVLH w czasie: shape (T, 3, 3)
    """
    T = states.shape[0]
    rel_pos_lvlh = np.zeros((T, 3, 3))  # (czas, satelita, xyz)

    for t in range(T):
        r_ref = states[t, 0, 0:3]
        v_ref = states[t, 0, 3:6]

        # LVLH axes at time t
        z_l = -r_ref / np.linalg.norm(r_ref)
        y_temp = np.cross(r_ref, v_ref)
        y_l = y_temp / np.linalg.norm(y_temp)
        x_l = np.cross(y_l, z_l)

        # Rotation matrix: LVLH to inertial
        R = np.vstack([x_l, y_l, z_l]).T  # shape (3, 3)

        # pozycje względne 3 pozostałych satelitów

        rel_global = states[t, 1:4, 0:3] - r_ref  # shape (3, 3)
        rel_local = rel_global @ R  # shape (3, 3)

        rel_pos_lvlh[t] = rel_local

    return  rel_pos_lvlh





if __name__ == "__main__":
    '''
    Orbit propagation
    States (len(time), satelitte num, x y z vx vy vz)
    Shape  (len(time),       4,             6)
    '''
    times, states = propagate_orbit_rk4(pos, t0, tf, dt)
    lvlh_states = to_mothership_ref_frame(states)

    # animate_formation_lvlh(lvlh_states, step=500, save=False)
    animate_formation(states, step=500, save=False)
    # plot_errors(times, states, sat_separation)    
    # plot_orbit_3d(times, states)