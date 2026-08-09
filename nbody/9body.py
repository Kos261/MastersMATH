from pathlib import Path
import matplotlib.pyplot as plt
import spiceypy as sp

from nbody.physics import f, hamiltonian_series, angular_momentum_series
from nbody.solvers import runge_kutta_4, symplectic_euler, stormer_verlet, yoshida_4
from nbody.load_ephemeris import load_initial_states, get_true_ephemeris
from nbody.simulation import run_models, run_integrators, run_steps
from nbody.plotting import plot_ephemeris_errors, plot_model_differences, plot_different_steps, plot_energy, plot_momentum

BODIES = {
    'SUN': {'id': 10, 'mu': 132712440041.253311},
    'MERCURY': {'id': 1, 'mu': 22031.868551},
    'VENUS': {'id': 2, 'mu': 324858.592000},
    'EARTH BARYCENTER': {'id': 3, 'mu': 398600.435544},
    'MARS BARYCENTER': {'id': 4, 'mu': 42828.375816},
    'JUPITER BARYCENTER': {'id': 5, 'mu': 126712764.100000},
    'SATURN BARYCENTER': {'id': 6, 'mu': 37940584.841800},
    'URANUS BARYCENTER': {'id': 7, 'mu': 5794556.400000},
    'NEPTUNE BARYCENTER': {'id': 8, 'mu': 6836527.100580},
    'PLUTO BARYCENTER': {'id': 9, 'mu': 975.500000},
}

if __name__ == "__main__":
    try:
        project_dir = Path(__file__).resolve().parent.parent
        data_dir = project_dir / "data"

        sp.furnsh(str(data_dir / "de442s.bsp"))
        sp.furnsh(str(data_dir / "naif0012.tls"))
        sp.furnsh(str(data_dir / "sb441-n16.bsp"))

        t0_str = "2000-07-01 00:00:00"
        tf_str = "2010-07-01 00:00:00"

        et0 = sp.str2et(t0_str)
        etf = sp.str2et(tf_str)

        names, state0, mus = load_initial_states(et0, bodies=BODIES)
        sun_idx = 0
        EXPERIMENT = "integrators"  # steps,integrators,models

        if EXPERIMENT == "models":
            h = 6 * 3600

            models = {
                "Newton": lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=False, relativistic=False),
                "J2": lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=True, relativistic=False),
                "REL": lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=False, relativistic=True),
                "J2+REL": lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=True, relativistic=True),
            }

            times, states = run_models(state0, et0, etf, h, models, runge_kutta_4)
            truth = get_true_ephemeris(names, times)

            plot_model_differences(states, times)
            plot_ephemeris_errors(states, truth, times)

        elif EXPERIMENT == "integrators":
            h = 6 * 3600

            model = lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=False, relativistic=False)

            integrators = {
                "RK4": runge_kutta_4,
                "Symplectic Euler": symplectic_euler,
                "Störmer-Verlet": stormer_verlet,
                "Yoshida 4": yoshida_4,
            }

            times, states = run_integrators(state0, et0, etf, h, model, integrators)
            truth = get_true_ephemeris(names, times)

            plot_ephemeris_errors(states, truth, times)

            H = {name: hamiltonian_series(s, mus, J2_pert=False, Rel=False) for name, s in states.items()}
            plot_energy(times, H)

        elif EXPERIMENT == "steps":
            model = lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=True, relativistic=True)

            steps = [
                3 * 3600,
                6 * 3600,
                12 * 3600,
                # 24 * 3600,
            ]

            states = run_steps(state0, et0, etf, steps, model, runge_kutta_4)
            plot_different_steps(states, names)

        plt.show()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sp.kclear()