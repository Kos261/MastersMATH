import numpy as np
from numpy.linalg import norm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.orbits import SUN, JUP, SAT, URA, PLU, NEP
import matplotlib.pyplot as plt
from multiprocessing import Pool
import copy
from numba import jit, prange
import os
from tqdm import tqdm

G = 2.95912208286e-4  # AU^3 / (day^2 * Msun)

def hamiltonian(state, masses):
    """Compute total energy (kinetic + potential) - vectorized for speed"""
    r = state[:, :3]  # (N,3)
    v = state[:, 3:]  # (N,3)
    N = len(masses)
    
    # Ensure consistent dtype
    masses = masses.astype(np.float32)
    
    # Kinetic energy: sum(0.5 * m * v^2)
    v_squared = np.sum(v * v, axis=1)  # (N,)
    Ek = 0.5 * np.dot(masses, v_squared)
    
    # Potential energy: -G * sum_{i<j} m_i m_j / r_ij
    Ep = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = r[j] - r[i]
            r_ij = np.sqrt(np.dot(rij, rij))
            Ep -= G * masses[i] * masses[j] / r_ij
    
    return Ek + Ep

def f(t, state, masses):
    """Compute state derivatives: [v, a]"""
    # state: (N,6)  -> [x,y,z,vx,vy,vz]
    v = state[:, 3:6]                     # (N,3)
    a = acc(state, masses)                # (N,3)
    return np.hstack([v, a])              # (N,6)

@jit(nopython=True, parallel=True, fastmath=True)
def acc_jit(r, masses, G):
    """Compute accelerations using Numba JIT with parallelization and fastmath"""
    N = len(masses)
    a = np.zeros((N, 3))
    
    for i in prange(N):
        for j in range(N):
            if i == j:
                continue
            rij = r[j] - r[i]
            r_sq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
            inv_r3 = 1.0 / (r_sq * np.sqrt(r_sq))
            factor = G * masses[j] * inv_r3
            a[i, 0] += factor * rij[0]
            a[i, 1] += factor * rij[1]
            a[i, 2] += factor * rij[2]
    return a

def acc(state, masses):
    """Extract positions and compute accelerations"""
    return acc_jit(state[:, :3], masses, G)        

def pack_state(bodies):
    return np.array([np.hstack([b.pos, b.vel]) for b in bodies], dtype=float)  # (N,6)

def unpack_traj(states, bodies):
    for i, b in enumerate(bodies):
        b.pos_traj = states[:, i, :3]
        b.vel_traj = states[:, i, 3:]

         # (N,6)

def explicite_euler(state, t, masses, h, f):
    """Direct Euler implementation without f function overhead"""
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    r_new = r + h * v
    v_new = v + h * a
    return np.column_stack([r_new, v_new])

def symplectic_euler(state, t, masses, h, f):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    v_new = v + h * a
    r_new = r + h * v_new
    
    return np.column_stack([r_new, v_new])

def stormer_verlet(state, t, masses, h, f):
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    v_half = v + 0.5 * h * a
    r_new = r + h * v_half
    state_new = np.column_stack([r_new, v])
    a_new = acc(state_new, masses)
    v_new = v_half + 0.5 * h * a_new

    return np.column_stack([r_new, v_new])

def midpoint_scheme(state, t, masses, h, f):
    """Optimized midpoint scheme with single acc call"""
    r = state[:, :3]
    v = state[:, 3:]
    a = acc(state, masses)
    # Midpoint velocity and position
    v_mid = v + 0.5 * h * a
    r_mid = r + 0.5 * h * v
    state_mid = np.column_stack([r_mid, v_mid])
    a_mid = acc(state_mid, masses)
    # Update using midpoint acceleration
    r_new = r + h * v_mid
    v_new = v + h * a_mid
    return np.column_stack([r_new, v_new])

def runge_kutta_4(state, t, masses, h, f):
    """RK4 integrator - optimized"""
    h_2 = h * 0.5
    h_6 = h / 6.0
    
    k1 = f(t, state, masses)
    k2 = f(t + h_2, state + h_2 * k1, masses)
    k3 = f(t + h_2, state + h_2 * k2, masses)
    k4 = f(t + h, state + h * k3, masses)
    
    return state + h_6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)



def propagate_orbit(bodies, t0, tf, h, integrator, integrator_name="Integration", position=0):
    T = int((tf - t0)//h) + 1
    N = len(bodies)
    masses = np.array([b.mass for b in bodies], dtype=np.float32)
    states = np.zeros((T, N, 6), dtype=np.float32)
    states[0] = pack_state(bodies)
    t = t0
    
    # Progress bar for this integrator with fixed position
    with tqdm(total=T-1, desc=f"{integrator_name}", unit=" steps", leave=True, position=position, ncols=80) as pbar:
        for k in range(1, T):
            # Compute new state
            state_new_np = integrator(states[k-1], t, masses, h, f)
            states[k] = state_new_np
            t += h
            pbar.update(1)

    unpack_traj(states, bodies)
    return states

def run_integration(args):
    """Worker function for parallel integration"""
    bodies, t0, tf, h, integrator_func, integrator_name, position = args
    bodies_copy = copy.deepcopy(bodies)
    states = propagate_orbit(bodies_copy, t0, tf, h, integrator_func, integrator_name, position)
    return (integrator_name, states)

def plot_orbit_3d(states, label, save_path=None):
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
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()


if __name__ == "__main__":
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    bodies = [SUN, JUP, SAT, URA, NEP, PLU]
    t0 = 0.0
    #tf = 365250  # 100000 years in days
    tf = 36525 * 100
    h = 10.0  # 1 day step

    # Prepare tasks for parallel execution with position indices
    integrators = [
        (copy.deepcopy(bodies), t0, tf, h, explicite_euler, "Explicit Euler", 0),
        (copy.deepcopy(bodies), t0, tf, h, midpoint_scheme, "Midpoint Scheme", 1),
        (copy.deepcopy(bodies), t0, tf, h, runge_kutta_4, "Runge Kutta-4", 2),
        (copy.deepcopy(bodies), t0, tf, h, symplectic_euler, "Symplectic Euler", 3),
        (copy.deepcopy(bodies), t0, tf, h, stormer_verlet, "Stormer-Verlet", 4),
    ]

    # Run integrations in parallel using multiprocessing
    print(f"\nRunning {len(integrators)} integrations in parallel:\n")
    with Pool() as pool:
        results = list(pool.imap_unordered(run_integration, integrators))

    # Unpack results
    results_dict = {name: states for name, states in results}
    states_exp = results_dict["Explicit Euler"]
    states_mid = results_dict["Midpoint Scheme"]
    states_rk4 = results_dict["Runge Kutta-4"]
    states_sym = results_dict["Symplectic Euler"]
    states_str = results_dict["Stormer-Verlet"]

    # Save trajectory data as numpy files for later 3D visualization
    print("\nSaving trajectory data...")
    with tqdm(total=5, desc="Saving .npy files", unit=" files") as pbar:
        np.save(results_dir / "states_explicit_euler.npy", states_exp)
        pbar.update(1)
        np.save(results_dir / "states_midpoint_scheme.npy", states_mid)
        pbar.update(1)
        np.save(results_dir / "states_rk4.npy", states_rk4)
        pbar.update(1)
        np.save(results_dir / "states_symplectic_euler.npy", states_sym)
        pbar.update(1)
        np.save(results_dir / "states_stormer_verlet.npy", states_str)
        pbar.update(1)

    # Plot and save 3D orbit visualizations
    print("\nGenerating 3D orbit visualizations...")
    plot_orbit_3d(states_exp, "Explicit Euler", results_dir / "orbit_explicit_euler.png")

    plot_orbit_3d(states_mid, "Midpoint Scheme", results_dir / "orbit_midpoint_scheme.png")
    plot_orbit_3d(states_rk4, "Runge Kutta-4", results_dir / "orbit_rk4.png")
    plot_orbit_3d(states_sym, "Symplectic Euler", results_dir / "orbit_symplectic_euler.png")
    plot_orbit_3d(states_str, "Stormer-Verlet", results_dir / "orbit_stormer_verlet.png")

    masses = np.array([b.mass for b in bodies], dtype=np.float32)
    
    # Vectorized energy computation with progress bar
    print("\nComputing energy for all trajectories...")
    energies_exp = np.array([hamiltonian(states_exp[i], masses) for i in tqdm(range(len(states_exp)), desc="Explicit Euler energy")])
    energies_sym = np.array([hamiltonian(states_sym[i], masses) for i in tqdm(range(len(states_sym)), desc="Symplectic Euler energy")])
    energies_str = np.array([hamiltonian(states_str[i], masses) for i in tqdm(range(len(states_str)), desc="Stormer-Verlet energy")])
    energies_rk4 = np.array([hamiltonian(states_rk4[i], masses) for i in tqdm(range(len(states_rk4)), desc="RK4 energy")])
    
    # Save energy data
    print("\nSaving energy data...")
    with tqdm(total=4, desc="Energy files", unit=" files") as pbar:
        np.save(results_dir / "energies_explicit_euler.npy", energies_exp)
        pbar.update(1)
        np.save(results_dir / "energies_symplectic_euler.npy", energies_sym)
        pbar.update(1)
        np.save(results_dir / "energies_stormer_verlet.npy", energies_str)
        pbar.update(1)
        np.save(results_dir / "energies_rk4.npy", energies_rk4)
        pbar.update(1)
    
    # Plot and save energy comparison
    print("\nGenerating energy comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(energies_exp, label="Explicit Euler", alpha=0.7)

    ax.plot(energies_str, label="Stormer Verlet", alpha=0.7)
    ax.plot(energies_rk4, label="RK4", alpha=0.7)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Total Energy", fontsize=12)
    ax.set_title("Energy Conservation Comparison", fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "energy_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {results_dir / 'energy_comparison.png'}")
    plt.show()
    
    print(f"\nAll results saved to: {results_dir}")