import spiceypy as sp
# from spiceypy.utils.support_types import SPICEIntCell
import numpy as np
import matplotlib.pyplot as plt



def load_test():
    state, light_time = sp.spkezr('SATURN BARYCENTER', et, 'J2000',
                                  'NONE', 'EARTH')
    print(f"Pozycja X: {state[0]} km")
    print(f"Pozycja Y: {state[1]} km")
    print(f"Pozycja Z: {state[2]} km")


def load_planets():
    sp.furnsh(r"/home/konstanty/Pulpit/MastersMATH/data/de442s.bsp")
    sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/naif0012.tls')
    et = sp.str2et('2025-12-04 12:00:00')

    positions = {}
    t_start = sp.str2et('2025-01-01 00:00:00')
    t_end   = sp.str2et('2026-01-01 00:00:00')
    steps   = 1000
    times = np.linspace(t_start, t_end, steps)
    planets = ["MERCURY", "VENUS", "EARTH", "MOON", "MARS BARYCENTER",
               "JUPITER BARYCENTER", "SATURN BARYCENTER", "URANUS BARYCENTER",
               "NEPTUNE BARYCENTER", "PLUTO BARYCENTER"]

    for planet in planets:
        states = [sp.spkezr(planet, t, 'J2000', 'NONE', 'SUN')[0] for t in times]
        positions[planet] = np.array(states) # Wynik: tablica (1000, 6)

    print("EARTHs coordinates w/r to SUN at time ?:")
    print(f"x = {positions["EARTH"][0][0]}\ny = {positions["EARTH"][0][1]}\nz = {positions["EARTH"][0][2]}")
    print("EARTHs velocity:")
    print(f"vx = {positions["EARTH"][0][3]}\nvy = {positions["EARTH"][0][4]}\nvz = {positions["EARTH"][0][5]}")

    return positions

def plot(positions, body1="EARTH", body2="MERCURY"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x1 = positions[body1][:, 0]
    y1 = positions[body1][:, 1]
    z1 = positions[body1][:, 2]

    # To samo dla drugiego ciała
    x2 = positions[body2][:,0]
    y2 = positions[body2][:,1]
    z2 = positions[body2][:,2]

    # Rysujemy trajektorie
    ax.plot(x1, y1, z1, label=body1, color='blue')
    ax.plot(x2, y2, z2, label=body2, color='red')

    ax.scatter(x1[0], y1[0], z1[0], color='blue', marker='o')
    ax.scatter(x2[0], y2[0], z2[0], color='red', marker='o')

    ax.scatter([0], [0], [0], color='orange', label='SUN', s=100)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()
    plt.title(f"Trajectories of EARTH and MERCURY relative to SUN")

    plt.show()


def load_mars_2020():
    sp.furnsh(r'..\data\de438s.bsp')  # Planety
    sp.furnsh(r'..\data\mar097.bsp')  # Układ Marsa
    sp.furnsh(r'..\data\m2020_cruise_od138_v1.bsp')

    start   = sp.str2et("2020-10-02 00:00:00")
    end     = sp.str2et("2020-12-15 00:00:00")


def inspect_kernels():
    count = sp.ktotal('SPK')

    if count == 0:
        print("No kernels found")
        return

    print(f"Found {count} kernels:\n")
    for which in range(count):
        # kdata returns a tuple: (file, file type, source, handle)
        file, file_type, source, handle = sp.kdata(which, "spk")
        print(file)


if __name__ == "__main__":
    try:
        # load_test()
        positions = load_planets()
        # print(positions)
        # states = load_mars_2020()
        # sp.furnsh(r'/home/konstanty/Pulpit/UW/MastersMATH/data/kernels.tm')  # Planety

        inspect_kernels()

        plot(positions)
    finally:
        sp.kclear()
