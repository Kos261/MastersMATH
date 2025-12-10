import numpy as np
from numpy.linalg import norm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.orbits import SUN, JUP, SAT, URA, PLU, NEP
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import copy
from numba import jit, prange
import os
import torch

G = 2.95912208286e-4  # AU^3 / (day^2 * Msun)

# Detect GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def hamiltonian(state, masses):
    """Compute total energy (kinetic + potential) - GPU accelerated"""
    # Convert to torch tensors if needed
    if isinstance(state, np.ndarray):
        state_t = torch.from_numpy(state).to(device)
        masses_t = torch.from_numpy(masses).to(device)
    else:
        state_t = state.to(device)
        masses_t = masses.to(device)
    
    r = state_t[:, :3]  # (N,3)
    v = state_t[:, 3:]  # (N,3)
    N = len(masses_t)
    
    # Kinetic energy: sum(0.5 * m * v^2)
    v_squared = torch.sum(v * v, dim=1)  # (N,)
    Ek = 0.5 * torch.dot(masses_t, v_squared)
    
    # Potential energy: -G * sum_{i<j} m_i m_j / r_ij
    Ep = torch.tensor(0.0, device=device)
    for i in range(N):
        for j in range(i + 1, N):
            rij = r[j] - r[i]
            r_ij = torch.sqrt(torch.dot(rij, rij))
            Ep -= G * masses_t[i] * masses_t[j] / r_ij
    
    energy = (Ek + Ep).item() if hasattr(Ek + Ep, 'item') else float(Ek + Ep)
    return energy

def f(t, state, masses):
    """Compute state derivatives: [v, a]"""
    # state: (N,6)  -> [x,y,z,vx,vy,vz]
    v = state[:, 3:6]                     # (N,3)
    a = acc(state, masses)                # (N,3)
    return np.hstack([v, a])              # (N,6)

def acc_gpu(r_tensor, masses_tensor, G_val, device):
    """Compute accelerations using PyTorch GPU acceleration"""
    N = len(masses_tensor)
    a = torch.zeros((N, 3), device=device, dtype=r_tensor.dtype)
    
    # Vectorized computation using broadcasting
    r_expanded_i = r_tensor.unsqueeze(1)  # (N, 1, 3)
    r_expanded_j = r_tensor.unsqueeze(0)  # (1, N, 3)
    
    rij = r_expanded_j - r_expanded_i  # (N, N, 3)
    r_sq = torch.sum(rij * rij, dim=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    r_sq = torch.where(r_sq < 1e-10, torch.ones_like(r_sq), r_sq)
    
    # Compute 1/r^3
    inv_r3 = torch.where(r_sq < 1e-10, torch.zeros_like(r_sq), 1.0 / (r_sq * torch.sqrt(r_sq)))
    
    # Broadcast masses
    masses_factor = G_val * masses_tensor.unsqueeze(0)  # (1, N)
    
    # Compute acceleration contributions
    factor = masses_factor * inv_r3  # (N, N)
    
    # Sum contributions from all j to each i
    a = torch.sum(factor.unsqueeze(2) * rij, dim=1)  # (N, 3)
    
    return a

def acc(state, masses):
    """Extract positions and compute accelerations using GPU"""
    if isinstance(state, np.ndarray):
        r_tensor = torch.from_numpy(state[:, :3]).to(device).float()
        masses_tensor = torch.from_numpy(masses).to(device).float()
    else:
        r_tensor = state[:, :3].to(device)
        masses_tensor = masses.to(device)
    
    a_tensor = acc_gpu(r_tensor, masses_tensor, G, device)
    return a_tensor.cpu().numpy() if isinstance(state, np.ndarray) else a_tensor        

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



def propagate_orbit(bodies, t0, tf, h, integrator):
    T = int((tf - t0)//h) + 1
    N = len(bodies)
    masses = np.array([b.mass for b in bodies], dtype=np.float32)
    states = np.zeros((T, N, 6), dtype=np.float32)
    states[0] = pack_state(bodies)
    t = t0
    
    # Move initial state to GPU for faster computation
    state_gpu = torch.from_numpy(states[0]).to(device).float()
    masses_gpu = torch.from_numpy(masses).to(device).float()
    
    print(f"Starting propagation with {T} steps on {device}...")
    for k in range(1, T):
        # Convert GPU tensor back to numpy for integrator
        state_np = state_gpu.cpu().numpy()
        
        # Compute new state
        state_new_np = integrator(state_np, t, masses, h, f)
        
        # Move back to GPU
        state_gpu = torch.from_numpy(state_new_np).to(device).float()
        states[k] = state_new_np
        t += h
        
        if (k + 1) % max(1, T // 10) == 0:
            print(f"  Progress: {k+1}/{T} steps completed")

    unpack_traj(states, bodies)
    return states

def run_integration(args):
    """Worker function for parallel integration"""
    bodies, t0, tf, h, integrator_func, integrator_name = args
    print(f"Starting {integrator_name}...")
    bodies_copy = copy.deepcopy(bodies)
    states = propagate_orbit(bodies_copy, t0, tf, h, integrator_func)
    print(f"Finished {integrator_name}")
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
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        multiprocessing_context = torch.multiprocessing.get_context('spawn')
    except:
        from multiprocessing import get_context
        multiprocessing_context = get_context('spawn')
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    bodies = [SUN, JUP, SAT, URA, NEP, PLU]
    t0 = 0.0
    tf = 365250 * 100  # 100000 years in days
    h = 1.0  # 1 day step

    # Prepare tasks for parallel execution
    integrators = [
        (copy.deepcopy(bodies), t0, tf, h, explicite_euler, "Explicit Euler"),
        (copy.deepcopy(bodies), t0, tf, h, midpoint_scheme, "Midpoint Scheme"),
        (copy.deepcopy(bodies), t0, tf, h, runge_kutta_4, "Runge Kutta-4"),
        (copy.deepcopy(bodies), t0, tf, h, symplectic_euler, "Symplectic Euler"),
        (copy.deepcopy(bodies), t0, tf, h, stormer_verlet, "Stormer-Verlet"),
    ]

    # Run integrations in parallel with spawn context for CUDA
    with multiprocessing_context.Pool() as pool:
        results = pool.map(run_integration, integrators)

    # Unpack results
    results_dict = {name: states for name, states in results}
    states_exp = results_dict["Explicit Euler"]
    states_mid = results_dict["Midpoint Scheme"]
    states_rk4 = results_dict["Runge Kutta-4"]
    states_sym = results_dict["Symplectic Euler"]
    states_str = results_dict["Stormer-Verlet"]

    # Save trajectory data as numpy files for later 3D visualization
    np.save(results_dir / "states_explicit_euler.npy", states_exp)
    np.save(results_dir / "states_midpoint_scheme.npy", states_mid)
    np.save(results_dir / "states_rk4.npy", states_rk4)
    np.save(results_dir / "states_symplectic_euler.npy", states_sym)
    np.save(results_dir / "states_stormer_verlet.npy", states_str)
    print("Trajectory data saved as .npy files")

    # Plot and save 3D orbit visualizations
    plot_orbit_3d(states_exp, "Explicit Euler", results_dir / "orbit_explicit_euler.png")
    plot_orbit_3d(states_mid, "Midpoint Scheme", results_dir / "orbit_midpoint_scheme.png")
    plot_orbit_3d(states_rk4, "Runge Kutta-4", results_dir / "orbit_rk4.png")
    plot_orbit_3d(states_sym, "Symplectic Euler", results_dir / "orbit_symplectic_euler.png")
    plot_orbit_3d(states_str, "Stormer-Verlet", results_dir / "orbit_stormer_verlet.png")

    masses = np.array([b.mass for b in bodies], dtype=float)
    
    # Vectorized energy computation
    energies_exp = np.array([hamiltonian(states_exp[i], masses) for i in range(len(states_exp))])
    energies_sym = np.array([hamiltonian(states_sym[i], masses) for i in range(len(states_sym))])
    energies_str = np.array([hamiltonian(states_str[i], masses) for i in range(len(states_str))])
    energies_rk4 = np.array([hamiltonian(states_rk4[i], masses) for i in range(len(states_rk4))])
    
    # Save energy data
    np.save(results_dir / "energies_explicit_euler.npy", energies_exp)
    np.save(results_dir / "energies_symplectic_euler.npy", energies_sym)
    np.save(results_dir / "energies_stormer_verlet.npy", energies_str)
    np.save(results_dir / "energies_rk4.npy", energies_rk4)
    
    # Plot and save energy comparison
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