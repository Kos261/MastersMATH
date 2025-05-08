from matplotlib import pyplot as plt
import numpy as np

from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time, TimeDelta
from astropy import units as u
from itertools import combinations
from poliastro.twobody.propagation import propagate


from poliastro.bodies import Earth, Moon
from poliastro.constants import rho0_earth, H0_earth

from poliastro.core.elements import rv2coe
from poliastro.core.perturbations import (
    atmospheric_drag_exponential,
    third_body,
    J2_perturbation,
)
from poliastro.core.propagation import func_twobody
from poliastro.core.elements import rv_pqw
from poliastro.ephem import build_ephem_interpolant
from poliastro.plotting import OrbitPlotter3D
from poliastro.twobody import Orbit
from poliastro.twobody.events import LithobrakeEvent
from poliastro.twobody.propagation import CowellPropagator
from poliastro.twobody.sampling import EpochsArray
from poliastro.util import norm

import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"

MU = 398600.4418    # geocentryczny parametr grawitacyjny Ziemi [km^3/s^2]
J2 = 1.08263e-3 
R = Earth.R.to(u.km).value
k = Earth.k.to(u.km**3 / u.s**2).value

orbit = Orbit.circular(Earth, 250 * u.km, epoch=Time(0.0, format="jd", scale="tdb"))

# parameters of a body
C_D = 2.2  # dimentionless (any value would do)
A_over_m = ((np.pi / 4.0) * (u.m**2) / (100 * u.kg)).to_value(u.km**2 / u.kg)  # km^2/kg
B = C_D * A_over_m

# parameters of the atmosphere
rho0 = rho0_earth.to(u.kg / u.km**3).value  # kg/km^3
H0 = H0_earth.to(u.km).value

tofs = TimeDelta(np.linspace(0 * u.h, 100000 * u.s, num=2000))
solar_system_ephemeris.set("de432s")

epoch = Time(2454283.0, format="jd", scale="tdb")  
# setting the exact event date is important

# create interpolant of 3rd body coordinates (calling in on every iteration will be just too slow)


def f(t0, state, k):
    # Two-body acceleration keplerian
    du_kep = func_twobody(t0, state, k)

    # Atmospheric drag
    ax, ay, az = atmospheric_drag_exponential(
        t0,
        state,
        k,
        R=R,
        C_D=C_D,
        A_over_m=A_over_m,
        H0=H0,
        rho0=rho0,
    )
    du_ad_atm = np.array([0, 0, 0, ax, ay, az])

    # J2 perturbation
    ax, ay, az = J2_perturbation(t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
    du_ad_j2 = np.array([0, 0, 0, ax, ay, az])


    #Moon
    body_r = build_ephem_interpolant(
    Moon,
    28 * u.day,
    (epoch.value * u.day, epoch.value * u.day + 60 * u.day),
    rtol=1e-2,
    )
    ax, ay, az = third_body(
        t0,
        state,
        k,          #grav comstant
        k_third = 1 * Moon.k.to(u.km**3 / u.s**2).value,
        perturbation_body=body_r,
    )
    du_ad_3rd = np.array([0, 0, 0, ax, ay, az])

    return du_kep + du_ad_atm + du_ad_j2 + du_ad_3rd





if __name__ == "__main__":
    altitude = 35786
    r = (R + altitude) * u.km
    v = np.sqrt(MU / r) # prędkość kołowa [km/s]

    initial = Orbit.from_classical(
    Earth,
    r,   # Pół oś wielka ~ 35 786 km nad pow
    0.0001 * u.one,   # Mimośród półoś mała / półoś wielka
    1 * u.deg,        # Inklinacja - wychylenie od równika
    0.0 * u.deg,      # Omega długość węzła wstępującego RAAN kąt między kierunkiem na punkt Barana
    0.0 * u.deg,      # omega argument perycentrum
    0.0 * u.rad,      # Anomalia prawdziwa 
    epoch=epoch,
    )

    sat_separation = 100
    r0, v0 = initial.rv()

   
    delta_r = [
        [0, 0, 0],         # sat 1
        [sat_separation * np.sqrt(3) * 0.5,
        -sat_separation * np.sqrt(3)/6,
         sat_separation * 0.5],         # sat 2: 1 km do przodu
        
        [sat_separation * np.sqrt(3) * 0.5,
         sat_separation * np.sqrt(3)/6, 
         sat_separation * 0.5],         # sat 3: 1 km na bok
        
        [sat_separation * np.sqrt(3) * 0.5,
          sat_separation * np.sqrt(3)/3, 0],         # sat 4: 1 km w górę
        ] * u.km

    satellites = []

    for dr in delta_r:
        r = r0 + dr
        v = v0  # ten sam wektor prędkości (lub dodaj perturbacje)
        sat = Orbit.from_vectors(Earth, r, v, epoch=epoch)
        satellites.append(sat)



####PLOTTING
    frame = OrbitPlotter3D()
    frame.set_attractor(Earth)
    fig = frame._figure

    for sat in satellites:
        r_start, _ = sat.rv()
        fig.add_trace(go.Scatter3d(
            x=[r_start[0].to_value(u.km)],
            y=[r_start[1].to_value(u.km)],
            z=[r_start[2].to_value(u.km)],
            mode='markers',
            marker=dict(size=5, color='red'),
            name="Start point"
        ))



        ephem = sat.to_ephem(
        EpochsArray(sat.epoch + tofs, method=CowellPropagator(rtol=1e-6, f=f)),
        )
        frame.plot_ephem(ephem, label="orbit influenced by Moon")
        






    times = [initial.epoch + TimeDelta(t * u.h) for t in [0, 2, 4, 6, 8]]

    # Dla każdego znacznika czasowego:
    for t in times:
        positions = []
        for sat in satellites:
            prop = sat.propagate(t - sat.epoch)
            r, _ = prop.rv()
            positions.append(r.to_value(u.km))

        # Dodaj linie między wszystkimi parami satelitów
        for i, j in combinations(range(len(satellites)), 2):
            x = [positions[i][0], positions[j][0]]
            y = [positions[i][1], positions[j][1]]
            z = [positions[i][2], positions[j][2]]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='blue', width=2),
                name=f"Link t={t.iso[:13]}"
            ))


    fig.write_html("orbit_plot.html", auto_open=True)
