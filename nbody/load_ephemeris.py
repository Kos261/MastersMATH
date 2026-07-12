import spiceypy as sp
import numpy as np
import re

BODIES = {
    'SUN':                {'id': 10, 'mu': 132712440041.253311},
    'MERCURY':             {'id': 1,  'mu': 22031.868551},
    'VENUS':               {'id': 2,  'mu': 324858.592000},
    'EARTH BARYCENTER':    {'id': 3,  'mu': 398600.435544},
    'MARS BARYCENTER':     {'id': 4,  'mu': 42828.375816},
    'JUPITER BARYCENTER':  {'id': 5,  'mu': 126712764.100000},
    'SATURN BARYCENTER':   {'id': 6,  'mu': 37940584.841800},
    'URANUS BARYCENTER':   {'id': 7,  'mu': 5794556.400000},
    'NEPTUNE BARYCENTER':  {'id': 8,  'mu': 6836527.100580},
    'PLUTO BARYCENTER':    {'id': 9,  'mu': 975.500000},
}


def load_initial_states(et0, bodies=BODIES):
    names = list(bodies.keys())
    state0 = np.zeros((len(names), 6))
    mus = np.zeros(len(names))

    for i, name in enumerate(names):
        st, _ = sp.spkezr(name, et0, 'J2000', 'NONE', '0')
        state0[i] = st
        mus[i] = bodies[name]['mu']

    return names, state0, mus

def load_initial_states_with_asteroids(et0, asteroid_bsp_path, gm_by_number, planet_bodies=BODIES):
    planet_names = list(planet_bodies.keys())
    num_planets = len(planet_names)

    asteroid_gm = resolve_asteroid_ids(asteroid_bsp_path, gm_by_number)
    asteroid_ids = list(asteroid_gm.keys())
    num_asteroids = len(asteroid_ids)

    N = num_planets + num_asteroids
    state0 = np.zeros((N, 6))
    mus = np.zeros(N)

    for i, name in enumerate(planet_names):
        st, _ = sp.spkezr(name, et0, 'J2000', 'NONE', '0')
        state0[i] = st
        mus[i] = planet_bodies[name]['mu']

    for k, aid in enumerate(asteroid_ids):
        i = num_planets + k
        st, _ = sp.spkezr(str(aid), et0, 'J2000', 'NONE', '0')
        state0[i] = st
        mus[i] = asteroid_gm[aid]

    return planet_names, num_planets, state0, mus


def get_true_ephemeris(names, times):
    """
    Zwraca truth jako array (T, N, 6) - taki sam kształt jak states,
    żeby pasowało do istniejących funkcji plot_comparison/plot_errors/hamiltonian.
    Kolejność ciał (oś N) odpowiada kolejności w `names`.
    """
    T = len(times)
    N = len(names)
    truth = np.zeros((T, N, 6))

    for i, name in enumerate(names):
        states = [sp.spkezr(name, t, 'J2000', 'NONE', '0')[0] for t in times]
        truth[:, i, :] = np.array(states)

    return truth


def read_bsp_comments(path):
    handle = sp.dafopr(path)
    lines = []
    while True:
        n, buffer, done = sp.dafec(handle, bufsiz=50)
        lines.extend(buffer[:n])
        if done:
            break
    sp.dafcls(handle)
    return lines


def parse_asteroid_gm(lines):
    """
    Szuka linii typu:
    MA0001  1.396e-13   4.719e-10   62.628886
    i wyciąga (numer_asteroidy, GM w km^3/s^2) - ostatnia kolumna.
    """
    pattern = re.compile(
        r'^\s*MA(\d{4})\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)\s*$'
    )
    gm_by_number = {}
    for line in lines:
        m = pattern.match(line)
        if m:
            number = int(m.group(1))
            mu = float(m.group(2))
            gm_by_number[number] = mu
    return gm_by_number


def resolve_asteroid_ids(bsp_path, gm_by_number):
    """
    bsp_path: ścieżka do sb441-n16.bsp (lub -n373.bsp)
    gm_by_number: dict {numer: mu} sparsowany z komentarzy de442s.bsp
    """
    present_ids = sp.spkobj(bsp_path)

    resolved = {}
    for naif_id in present_ids:
        number = naif_id - 2_000_000
        if number in gm_by_number:
            resolved[naif_id] = gm_by_number[number]
        else:
            print(f"[warn] brak mu dla NAIF ID {naif_id} (numer {number})")

    print(f"[asteroids] dopasowano {len(resolved)} / {len(present_ids)}")
    return resolved


def inspect_kernel(kernel:str):
    ids = sp.spkobj(kernel)
    for id in ids:
        try:
            name = sp.bodc2n(id)
            print(f"{id} -> {name}")
        except:
            print(f"{id} -> brak nazwy w kernelu nazw")