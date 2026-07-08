
import os
import spiceypy as sp
import numpy as np
# from spiceypy.utils.support_types import SPICEIntCell
import matplotlib.pyplot as plt

from initial_conditions import initial_formation_1, initial_formation_4, initial_formation_explosion
from orbits import GEO
from plot_tools import Plotter
from physics import f
from solvers import runge_kutta_4
from data_loader import inspect_kernel, inspect_kernels, plot_planets


def plot_selected(ephemerides, to_plot=None):
    """
    Rysuje wybrane trajektorie z ephemerides.

    ephemerides: dict {nazwa: array (N,6)}
    to_plot: lista kluczy do narysowania, np. ["EARTH", "MARS BARYCENTER", "SPACESHIP"]
             jeśli None, rysuje wszystko
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0], [0], [0], color='orange', label='SUN', s=100)

    keys = to_plot if to_plot is not None else list(ephemerides.keys())

    for key in keys:
        if key not in ephemerides:
            print(f"[warn] '{key}' nie znaleziono w ephemerides, pomijam")
            continue

        positions = ephemerides[key]
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        if key == "SPACESHIP":
            # Wyróżnienie statku: gruba, czarna, przerywana linia + większy marker startowy
            ax.plot(x, y, z, label=key, color='black', linewidth=2.5, linestyle='--')
            ax.scatter(x[0], y[0], z[0], marker=11, color='black', s=150)
        else:
            ax.plot(x, y, z, label=key, linewidth=1.2)
            ax.scatter(x[0], y[0], z[0], marker='o', s=30)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()
    plt.title("Selected trajectories")
    plt.show()



def load_mars_2020():
    positions = {}
    # start   = sp.str2et("2020-10-02 00:00:00") #FREEFLIGHT
    # end     = sp.str2et("2020-12-15 00:00:00")
    start   = sp.str2et("2020-08-01 00:00:00")
    end     = sp.str2et("2021-12-17 00:00:00")

    steps = 1000
    times = np.linspace(start, end, steps)
    planets = ["MERCURY", "VENUS", "EARTH", "MOON", "MARS BARYCENTER",
               "JUPITER BARYCENTER", "SATURN BARYCENTER", "URANUS BARYCENTER",
               "NEPTUNE BARYCENTER", "PLUTO BARYCENTER"]

    for planet in planets:
        states = [sp.spkezr(planet, t, 'J2000', 'NONE', 'SUN')[0] for t in times]
        positions[planet] = np.array(states)  # Wynik: tablica (1000, 6)

    spaceship_states = [sp.spkezr("M2020", t, 'J2000', 'NONE', 'SUN')[0] for t in times]
    positions["SPACESHIP"] = np.array(spaceship_states)

    return positions

if __name__ == "__main__":
    try:
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/de442s.bsp')  # Planety
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/mar097.bsp')  # Układ Marsa
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/m2020_cruise_od138_v1.bsp')  # Rzeczywiste położenia statku
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/naif0012.tls')  # kernel czasu
        inspect_kernel(r'/home/konstanty/Pulpit/MastersMATH/data/m2020_cruise_od138_v1.bsp')
        positions = load_mars_2020()

        plot_selected(positions, to_plot=["EARTH", "MARS BARYCENTER", "SPACESHIP"])
    finally:
        sp.kclear()
