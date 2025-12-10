import spiceypy as sp
# from spiceypy.utils.support_types import SPICEIntCell
import numpy as np




def load_test():
    state, light_time = sp.spkezr('SATURN BARYCENTER', et, 'J2000',
                                  'NONE', 'EARTH')
    print(f"Pozycja X: {state[0]} km")
    print(f"Pozycja Y: {state[1]} km")
    print(f"Pozycja Z: {state[2]} km")


def load_planets():
    sp.furnsh(r"..\data\de442s.bsp")
    sp.furnsh(r'..\data\naif0012.tls')
    et = sp.str2et('2025-12-04 12:00:00')

    positions = {}
    t_start = sp.str2et('2025-01-01 00:00:00')
    t_end   = sp.str2et('2025-01-02 00:00:00')
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


def load_mars_2020():
    sp.furnsh('..\data\de438s.bsp')  # Planety
    sp.furnsh('..\data\mar097.bsp')  # Układ Marsa
    sp.furnsh('..\data\m2020_cruise_od138_v1.bsp')

    start   = sp.str2et("2020-10-02 00:00:00")
    end     = sp.str2et("2020-12-15 00:00:00")


def inspect_kernels():
    count = sp.ktotal('SPK')

    if count == 0:
        print("No kernels found")
        return

    print(f"Found {count} kernels:\n")
    for i in range(count):
        file_path,_,_,_ = sp.kdata(i, 'SPK')
        print("ID: ", sp.spkobj(file_path))


if __name__ == "__main__":
    try:
        # load_test()
        # positions = load_planets()
        # states = load_mars_2020()
        sp.furnsh('..\data\de438s.bsp')  # Planety
        sp.furnsh('..\data\mar097.bsp')  # Układ Marsa
        sp.furnsh('..\data\m2020_cruise_od138_v1.bsp')
        inspect_kernels()


    finally:
        sp.kclear()
