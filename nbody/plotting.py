import numpy as np
import matplotlib.pyplot as plt

from nbody.load_ephemeris import get_true_ephemeris


def plot_ephemeris_errors(states_by_method, truth, times):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / 86400

    truth_mercury = truth[:, 1, :3] - truth[:, 0, :3]

    for method_name, states in states_by_method.items():
        sim_mercury = states[:, 1, :3] - states[:, 0, :3]
        err = np.linalg.norm(sim_mercury - truth_mercury, axis=1)
        ax.plot(time_days, err, label=method_name)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Heliocentric position error [km]")
    ax.set_title("Mercury vs JPL ephemeris")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_error(states_by_method, truth, times, truth_times):
    fig, ax = plt.subplots(figsize=(10, 10))

    mask = times <= truth_times[-1]
    times_plot = times[mask]

    idx = np.searchsorted(truth_times, times_plot)

    truth_sampled = truth[idx]
    truth_helio = truth_sampled[:, 1, :3] - truth_sampled[:, 0, :3]

    for name, states in states_by_method.items():
        states = states[mask]

        sim_helio = states[:, 1, :3] - states[:, 0, :3]
        err = np.linalg.norm(sim_helio - truth_helio, axis=1)

        ax.plot(
            (times_plot - times_plot[0]) / 86400,
            err,
            label=name
        )

    # ax.set_yscale("log")
    ax.set_title("Error")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Position error [km]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_model_differences(states_by_model, times):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / 86400

    def mercury_helio(states):
        return states[:, 1, :3] - states[:, 0, :3]

    r_newton = mercury_helio(states_by_model["Newton"])

    for model_name in ["J2", "REL", "J2+REL"]:
        r_model = mercury_helio(states_by_model[model_name])
        diff = np.linalg.norm(r_model - r_newton, axis=1)
        ax.plot(time_days, diff, label=model_name)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Difference from Newtonian orbit [km]")
    ax.set_title("Influence of perturbations on Mercury")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_different_steps(states_by_step, names):
    fig, ax = plt.subplots(figsize=(10, 10))

    for step_name, (times, states) in states_by_step.items():
        truth = get_true_ephemeris(names, times)
        time_days = (times - times[0]) / 86400

        truth_mercury = truth[:, 1, :3] - truth[:, 0, :3]
        sim_mercury = states[:, 1, :3] - states[:, 0, :3]

        err = np.linalg.norm(sim_mercury - truth_mercury, axis=1)
        ax.plot(time_days, err, label=step_name)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Heliocentric position error [km]")
    ax.set_title("Influence of integration step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_energy_drift_ratio(results):
    fig, ax = plt.subplots(figsize=(10, 7))

    for method_name, method_results in results.items():
        hs = sorted(method_results.keys())
        h_hours = np.array(hs) / 3600
        ratios = [method_results[h]["ratio"] for h in hs]

        ax.plot(h_hours, ratios, marker="o", label=method_name)

    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step size h [hours]")
    ax.set_ylabel(r"$\rho_H = D_H / E_H^{osc}$")
    ax.set_title("Energy drift relative to oscillatory energy error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax

def plot_crossover_time(results):
    fig, ax = plt.subplots(figsize=(10, 7))

    for method_name, method_results in results.items():
        hs = sorted(method_results.keys())
        h_hours = np.array(hs) / 3600.0
        T_star_years = [method_results[h]["crossover_days"] / 365.25 for h in hs]

        ax.plot(h_hours, T_star_years, marker="o", label=method_name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step size h [hours]")
    ax.set_ylabel(r"$T_*$ [years]")
    ax.set_title(r"Crossover time $T_* = E_H^{osc}/|b|$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_orbits(names, states_by_method):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for method_name, states in states_by_method.items():
        for i, body_name in enumerate(names):
            ax.plot(states[:, i, 0], states[:, i, 1], states[:, i, 2], label=f"{method_name} - {body_name}")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("Comparison of numerical integrators")
    # ax.legend(fontsize=7)
    plt.show()


def plot_energy(times, H_by_method, H0):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / 86400

    for name, H in H_by_method.items():
        # err = (H - H0) / abs(H0)
        # ax.plot(time_days, err, label=name)
        ax.plot(time_days, (H - H0), label=name)

    # ax.axhline(H0, linestyle="--", alpha=0.5, label="Exact")
    ax.axhline(0, linestyle="--", alpha=0.5)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel(r"$H(t)-H_0$")
    ax.set_title("Hamiltonian")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_momentum(times, L_by_method):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / 86400

    for name, L in L_by_method.items():
        L_norm = np.linalg.norm(L, axis=1)
        ax.plot(time_days, (L_norm - L_norm[0]) / L_norm[0], label=name)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel(r"$|\mathbf{L}|$")
    ax.set_title("Angular momentum comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax