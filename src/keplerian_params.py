import numpy as np
from numpy.linalg import norm
from numpy import sin, cos
from orbits import Orbit

MU = 398600.4418
G = 6.67430151515e-11
RE = 6378.137         # [km]
J2 = 1.08262668e-3

def cart2kepler(r,v):
    r_norm = norm(r)
    v_norm = norm(v)
    h = np.cross(r, v)
    h_norm = norm(h)
    n = np.cross(np.array([0, 0, 1]), h)
    n_norm = norm(n)

    e = ((v_norm**2 - MU / r_norm) * r - np.dot(r, v) * v) / MU
    e_norm = norm(e)
    Energy = (v_norm * v_norm) / 2 - MU / r_norm

    if abs(e_norm - 1.0) > 1e-8:
        a = -MU / (2 * Energy)
        p = a * (1 - e_norm * e_norm)
    else:
        p = a * (1 - e_norm * e_norm)
        a = np.inf

    i = np.acos(h[2]/h_norm)
    raan = np.degrees(np.arccos(n[0] / n_norm))

    if n[1] < 0:
        raan = 360 - raan
    
    argp = np.degrees(np.acos(np.dot(n, e) / (n_norm * e)))

    if e[2] < 0:
        argp = 360 - argp

    nu = np.degrees(np.acos(np.dot(e, r) / (e_norm * r_norm)))

    if np.dot(r, v) < 0:
        nu = 360 - nu

    return a, e, i, raan, argp, nu


def kepler2cart(o: Orbit):
    #OE to Perifocal plane
    a, ecc, inc, raan, argp, nu = o.a, o.ecc, o.inc, o.raan, o.argp, o.nu
    i, raan, argp, nu = np.radians(inc), np.radians(raan), np.radians(argp), np.radians(nu)

    if hasattr(ecc, '__len__'):
        e_norm = norm(ecc)
    else:
        e_norm = ecc

    p = a * (1 - e_norm * e_norm)
    r = p / (1 + e_norm * np.cos(nu))
    eci = np.array([r * np.cos(nu), r * np.sin(nu), 0])

    c = np.sqrt(MU/p)
    eci_v = np.array([-c * np.sin(nu), c * (ecc + np.cos(nu)), 0])

    R = np.array([
[cos(raan)*cos(argp)-sin(raan)*sin(argp)*cos(inc),-cos(raan)*sin(argp)-sin(raan)*cos(argp)*cos(inc),sin(raan)*sin(inc)],
[sin(raan)*cos(argp) + cos(raan)*sin(argp)*cos(inc), -sin(raan)*sin(argp) + cos(raan)*cos(argp)*cos(inc), -cos(raan) * sin(inc)],
[sin(argp)*sin(inc), cos(argp)*sin(inc), cos(inc)]
])
    
    pos = R @ eci
    vel = R @ eci_v

    return pos, vel


def orbital_vel(r):
    return np.sqrt(MU / r)


def LVLH(states):
        """
        Transformuje pozycje satelitów do układu odniesienia statku matki (LVLH).
        Zakłada, że:
            - states ma shape (T, 4, 6), gdzie 4 to liczba satelitów
            - statek-matka to satelita o indeksie 0
        Zwraca pozycje WSZYSTKICH satelitów (w tym statku-matki) w układzie LVLH: shape (T, 4, 3)
        """

        T = states.shape[0]
        rel_pos_lvlh = np.zeros((T, 4, 3))  # (czas, satelita, xyz)

        for t in range(T):
            r_ref = states[t, 0, 0:3]
            v_ref = states[t, 0, 3:6]

            # LVLH axes at time t
            z_l = -r_ref / np.linalg.norm(r_ref)
            y_temp = np.cross(r_ref, v_ref)
            y_l = y_temp / np.linalg.norm(y_temp)
            x_l = np.cross(y_l, z_l)

            # Rotation matrix: inertial -> LVLH
            R = np.vstack([x_l, y_l, z_l]).T  # shape (3, 3)

            # Mothership -> (0,0,0)
            rel_pos_lvlh[t, 0] = np.zeros(3)

            # pozostałe 3 satelity
            rel_global = states[t, 1:4, 0:3] - r_ref  # shape (3, 3)
            rel_local = rel_global @ R               # shape (3, 3)
            rel_pos_lvlh[t, 1:4] = rel_local

        return rel_pos_lvlh