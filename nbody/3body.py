from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from nbody.physics import *
from nbody.solvers import *
from nbody.load_ephemeris import *

BODIES = {
    'SUN':                {'id': 10, 'mu': 132712440041.253311},
    'JUPITER BARYCENTER':  {'id': 5,  'mu': 126712764.100000},
    'SATURN BARYCENTER':   {'id': 6,  'mu': 37940584.841800},
}

def propagate_orbit(state0, t0, tf, h, f, integrator):
    times = np.arange(t0, tf + h, h)
    T = len(times)
    N = state0.shape[0]
    states = np.zeros((T, N, 6))
    states[0] = state0

    for i in tqdm(range(1, T)):
        states[i] = integrator(state=states[i - 1], t=times[i - 1], h=h, f=f)
        if np.any(np.isnan(states[i])):
            print(f"NaN pojawił się w kroku {i}, t = {times[i]}")
            break
    return times, states


def plot_comparison(names, states_by_method):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for method_name, states in states_by_method.items():
        for i, body_name in enumerate(names):
            ax.plot(states[:, i, 0], states[:, i, 1], states[:, i, 2], label=f"{method_name} - {body_name}")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("Comparison of numerical integrators")
    ax.legend(fontsize=7)
    plt.show()


# def plot_errors(names, states_by_method, times, t0, plot_name="ERROR"):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     time_days = (times - t0) / (24 * 60 * 60)
#
#     for method_name, states in states_by_method.items():
#         for i, body_name in enumerate(names):
#             err = np.linalg.norm(states[:, i, :3] - truth[:, i, :3], axis=1)
#             ax.plot(time_days, err, label=f"{method_name} - {body_name}")
#
#     ax.set_xlabel("Time [days]")
#     ax.set_ylabel("Position error [km]")
#     ax.set_title("Position error vs ephemeris")
#     ax.legend(fontsize=7)
#     ax.grid(True, alpha=0.3)
#
#     output_dir = Path("plots/nbody")
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     fig.tight_layout()
#     fig.savefig(output_dir / f"{plot_name}.png")
#     plt.show()


def plot_energy(times, H_by_method, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / (24 * 60 * 60)

    for method_name, H in H_by_method.items():
        ax.plot(time_days, H, label=method_name)

    if "h" in kwargs and "tf" in kwargs:
        ax.set_title(f"Hamiltonian comparison (h = {kwargs['h']}, tf = {kwargs['tf']})")
    else:
        ax.set_title("Hamiltonian comparison")

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("E_k + E_p")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_momentum(times, L_by_method, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / (24 * 60 * 60)

    for method_name, L in L_by_method.items():
        ax.plot(time_days, L, label=method_name)

    if "h" in kwargs and "tf" in kwargs:
        ax.set_title(f"Angular momentum comparison (h = {kwargs['h']}, tf = {kwargs['tf']})")
    else:
        ax.set_title("Angular momentum comparison")

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("$\sum$ p x r")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Running 3 body simulation")
    try:
        PROJECT_DIR = Path(__file__).resolve().parent.parent
        DATA_DIR = PROJECT_DIR / "data"

        sp.furnsh(str(DATA_DIR / "de442s.bsp"))
        sp.furnsh(str(DATA_DIR / "naif0012.tls"))
        sp.furnsh(str(DATA_DIR / "sb441-n16.bsp"))

        t0_str = "1900-07-01 00:00:00"
        tf_str = "2050-07-01 00:00:00"
        h = float(60 * 60 * 6) # 6h
        et0 = sp.str2et(t0_str)
        etf = sp.str2et(tf_str)

        names, state0, mus = load_initial_states(et0,bodies=BODIES)
        sun_idx_val = 0

        model_func = lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx_val, J2_pert=False, relativistic=False)
        integrators = {"RK4" : runge_kutta_4,
                       "Stromer-Verlet": stormer_verlet,
                       "Yoshida 4": yoshida_4,
                        # "Symplectic euler": symplectic_euler,
                       }

        states_by_method = {}

        for method_name, integrator in integrators.items():
            times, states_by_method[method_name] = propagate_orbit(
                state0=state0,
                t0=et0,
                tf=etf,
                h=h,
                f=model_func,
                integrator=integrator,
            )

        # truth = get_true_ephemeris(names, times)

        H_by_method = {method_name: hamiltonian_series(states, mus) for method_name, states in states_by_method.items()}
        L_by_method = {method_name: np.linalg.norm(angular_momentum_series(states, mus), axis=1) for method_name, states in states_by_method.items()}

        plot_comparison(names, states_by_method)
        # plot_errors(names, states_by_method, truth, times, et0, "integrators_comparison")
        plot_energy(times, H_by_method, h=h, tf=tf_str)
        plot_momentum(times, L_by_method, h=h, tf=tf_str)

    finally:
        sp.kclear()