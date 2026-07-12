import numpy as np
import matplotlib.pyplot as plt
from Spice_Lessons.data_loader import load_planets

MU_SUN = 1.32712440018e11


def main():
    t = np.linspace(0, 10, 100)

    positions = load_planets()
    x1 = positions["EARTH"][0, 0]
    y1 = positions["EARTH"][0, 1]
    r1 = np.sqrt(x1**2 + y1**2)
    x_earth = r1 * np.cos(t)
    y_earth = r1 * np.sin(t)

    x2 = positions["MARS BARYCENTER"][0, 0]
    y2 = positions["MARS BARYCENTER"][0, 1]
    r2 = np.sqrt(x2**2 + y2**2)
    x_mars = r2 * np.cos(t)
    y_mars = r2 * np.sin(t)

    #Transfer
    v1_orbit = np.sqrt(MU_SUN / r1)
    v2_orbit = np.sqrt(MU_SUN / r2)
    a = (r1 + r2) / 2           #semi_major_axis
    e = (r2 - r1) / (r2 + r1)   #eccentricity
    v1_transfer = np.sqrt(MU_SUN * (2 / r1 - 1 / a))
    v2_transfer = np.sqrt(MU_SUN * (2 / r2 - 1 / a))
    t_transfer = np.pi * np.sqrt(a ** 3 / MU_SUN)
    theta = np.linspace(0, np.pi, 100)

    r_transfer = (a * (1 - e ** 2) / (1 + e * np.cos(theta)))
    x_sat = r_transfer * np.cos(theta)
    y_sat = r_transfer * np.sin(theta)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_earth, y_earth,0, label="EARTH", color='blue')
    ax.plot(x_mars, y_mars,0, label="MARS", color='orange')
    ax.plot(x_sat, y_sat, 0, label="SATELLITE", color='red',linestyle='dashed' )

    ax.scatter(x_earth[0], y_earth[0],0, color='blue', marker='o')
    ax.scatter(-x_mars[0], y_mars[0],0, color='orange', marker='o')
    ax.scatter(x_sat[0], y_sat[0], 0, color='red', marker='o')

    ax.scatter([0], [0], [0], color='orange', label='SUN', s=100)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()
    plt.title(f"Trajectories of EARTH and MERCURY relative to SUN")

    plt.show()

if __name__ == "__main__":
    main()


