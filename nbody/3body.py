from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as sp

from nbody.physics import f, hamiltonian_series, angular_momentum_series
from nbody.solvers import runge_kutta_4, symplectic_euler, stormer_verlet, yoshida_4
from nbody.load_ephemeris import load_initial_states, get_true_ephemeris, get_reference_solution
from nbody.simulation import run_models, run_integrators, run_steps, propagate_orbit
from nbody.plotting import plot_ephemeris_errors, plot_model_differences, plot_different_steps, plot_energy, plot_momentum, plot_error
from solvers import gauss_legendre_4

BODIES = {
    'SUN':                {'id': 10, 'mu': 132712440041.253311},
    'JUPITER BARYCENTER':  {'id': 5,  'mu': 126712764.100000},
    'SATURN BARYCENTER':   {'id': 6,  'mu': 37940584.841800},
}




if __name__ == '__main__':
    print("Running 3 body simulation")
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


        if EXPERIMENT == "integrators":
            h = 6 * 3600

            model = lambda t, s: f(t, s, masses=mus, sun_idx=sun_idx, J2_pert=False, relativistic=False)

            integrators = {
                "RK4": runge_kutta_4,
                # "Symplectic Euler": symplectic_euler,
                # "Störmer-Verlet": stormer_verlet,
                "Yoshida 4": yoshida_4,
                # "Gauss-Legendre 4": gauss_legendre_4
            }

            times, states = run_integrators(state0, et0, etf, h, model, integrators)

            # very good approximation
            truth_times, truth = get_reference_solution(
                "references/reference_sol_h60_10y.npz",
                lambda: propagate_orbit(
                    state0, et0, etf,
                    h=60,
                    f=model,
                    integrator=runge_kutta_4
                )
            )


            plot_error(states, truth, times, truth_times)

            H = {name: hamiltonian_series(s, mus, J2_pert=False, Rel=False) for name, s in states.items()}
            H_true = hamiltonian_series(truth, mus, J2_pert=False, Rel=False)
            plot_energy(times, H, H_true)
            L = {name: angular_momentum_series(s, mus) for name, s in states.items()}
            plot_momentum(times, L)

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
        print("Clearing kernels")
        sp.kclear()