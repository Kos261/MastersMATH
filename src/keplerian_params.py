import numpy as np
from numpy.linalg import norm
from numpy import sin, cos
MU = 398600.4418


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
    Omega = np.degrees(np.arccos(n[0] / n_norm))

    if n[1] < 0:
        Omega = 360 - Omega
    
    argp = np.degrees(np.acos(np.dot(n, e) / (n_norm * e)))

    if e[2] < 0:
        argp = 360 - argp

    nu = np.degrees(np.acos(np.dot(e, r) / (e_norm * r_norm)))

    if np.dot(r, v) < 0:
        nu = 360 - nu

    return a, e, i, Omega, argp, nu


def kepler2cart(a, e, i, Omega, argp, nu):
    #OE to Perifocal plane
    i, Omega, argp, nu = np.radians(i), np.radians(Omega), np.radians(argp), np.radians(nu)
    if hasattr(e, '__len__'):
        e_norm = norm(e)
    else:
        e_norm = e

    p = a * (1 - e_norm * e_norm)
    r = p / (1 + e_norm * np.cos(nu))
    eci = np.array([r * np.cos(nu), r * np.sin(nu), 0])

    c = np.sqrt(MU/p)
    eci_v = np.array([-c * np.sin(nu), c * (e + np.cos(nu)), 0])

    R = np.array([
[cos(Omega)*cos(argp)-sin(Omega)*sin(argp)*cos(i),-cos(Omega)*sin(argp)-sin(Omega)*cos(argp)*cos(i),sin(Omega)*sin(i)],
[sin(Omega)*cos(argp) + cos(Omega)*sin(argp)*cos(i), -sin(Omega)*sin(argp) + cos(Omega)*cos(argp)*cos(i), -cos(Omega) * sin(i)],
[sin(argp)*sin(i), cos(argp)*sin(i), cos(i)]
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