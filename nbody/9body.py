from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as sp

from nbody.physics import *
from nbody.solvers import *
from nbody.load_ephemeris import *
from nbody.simulation import *
from nbody.plotting import *

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
    print("Running 9 body simulation")
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
        EXPERIMENT = "energy_drift"  # steps,integrators,models

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

            model = lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=True, relativistic=False)

            integrators = {
                "RK4": runge_kutta_4,
                # "Symplectic Euler": symplectic_euler,
                # "Störmer-Verlet": stormer_verlet,
                "Yoshida 4": yoshida_4,
                "Gauss-Legendre 4": gauss_legendre_4
            }

            times, states = run_integrators(state0, et0, etf, h, model, integrators, filename="gauss_j2")

            # true ephemeris
            truth = get_true_ephemeris(names, times)

            # very good approximation
            times_true, true = get_reference_solution(
                "references/reference_sol_j2_h3600_10y.npz",
                lambda: propagate_orbit(
                    state0, et0, etf,
                    h=60 * 60,
                    f=model,
                    integrator=runge_kutta_4
                )
            )

            plot_ephemeris_errors(states, truth, times)
            plot_error(states, true, times, times_true)

            H0 = hamiltonian(state0, mus)
            H = {name: hamiltonian_series(s, mus, J2_pert=True, Rel=False) for name, s in states.items()}
            plot_energy(times, H, H0)

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

        elif EXPERIMENT == "energy_drift":

            model_name = "Newton"
            model = lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=False, relativistic=False)
            integrators = { "RK4": runge_kutta_4, "Verlet": stormer_verlet, "Yoshida4": yoshida_4}
            steps = [48 * 3600 ,24 * 3600,12 * 3600,6 * 3600,3 * 3600]

            results = {}

            for method_name, integrator in integrators.items():

                results[method_name] = {}

                for h in steps:
                    cache_name = make_cache_name(
                    integrator_name=method_name,
                    model_name=model_name,
                    bodies=BODIES,
                    t0_str=t0_str,
                    tf_str=tf_str,
                    h=h)

                    times, states = propagate_cached(
                    state0=state0,
                    t0=et0,
                    tf=etf,
                    h=h,
                    f=model,
                    integrator=integrator,
                    cache_dir="results/states",
                    cache_name=cache_name,
                    )

                    H = hamiltonian_series(states, mus, J2_pert=False)
                    metrics = energy_metrics(times, H)
                    results[method_name][h] = metrics

                    print(f"{method_name:10s} h={h / 3600:5.1f} h | "
                        f"drift={metrics['drift']:.3e} | "
                        f"osc={metrics['oscillation']:.3e} | "
                        f"ratio={metrics['ratio']:.3e} | "
                        f"T*={metrics['crossover_days'] / 365.25:.2f} y")

            plot_energy_drift_ratio(results)
            plot_crossover_time(results)

        plt.show()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sp.kclear()