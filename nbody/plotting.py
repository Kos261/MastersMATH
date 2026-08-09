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


def plot_energy(times, H_by_method):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / 86400

    for method_name, H in H_by_method.items():
        ax.plot(time_days, H, label=method_name)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Hamiltonian")
    ax.set_title("Hamiltonian comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_momentum(times, L_by_method):
    fig, ax = plt.subplots(figsize=(10, 10))
    time_days = (times - times[0]) / 86400

    for method_name, L in L_by_method.items():
        ax.plot(time_days, L, label=method_name)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Angular momentum")
    ax.set_title("Angular momentum comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax