from load_ephemeris import *
from solvers import *
from nbody_solar_system import *

if __name__ == '__main__':
    try:
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/de442s.bsp')
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/naif0012.tls')
        sp.furnsh(r'/home/konstanty/Pulpit/MastersMATH/data/sb441-n16.bsp')
        # Test 2-ciałowy: Słońce + Merkury, mały krok
        mini_bodies = {
            'SUN': BODIES['SUN'],
            'MERCURY': BODIES['MERCURY'],
            'JUPITER BARYCENTER': BODIES['JUPITER BARYCENTER'],  # dominujący wpływ na Słońce
        }
        t0_str = "2000-07-01 00:00:00"
        tf_str = "2005-07-01 00:00:00"

        et0 = sp.str2et(t0_str)
        etf = sp.str2et(tf_str)

        names2, state0_2, mus2 = load_initial_states(et0, mini_bodies)

        h_small = 3600.0  # 1h zamiast 24h -> ~2000 kroków/okrążenie zamiast 88

        times2, states2 = propagate_orbit(state0_2, mus2, et0, etf, h_small, f, runge_kutta_4)
        truth2 = get_true_ephemeris(names2, times2)
        plot_errors(names2, states2, truth2, times2, et0)

    finally:
        sp.kclear()