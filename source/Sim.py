import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from initial_conditions import *

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

def plot_orbit_3d(states):
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
    step = 10
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
    
    # Włącz pełną interaktywność
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


if __name__ == "__main__":
    # Propagacja orbity
    # States (len(time), satelitte num, x y z vx vy vz)
    # Shape  (len(time), 4, 6)
    times, states = propagate_orbit_rk4(pos, t0, tf, dt)

    plot_errors(times, states, sat_separation)    
    # plot_orbit_3d(states)