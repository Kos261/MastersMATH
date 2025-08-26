# from poliastro.twobody import Orbit
# from poliastro.bodies import Earth
# from astropy import units as u
# from astropy.time import Time

# epoch = Time(2454283.0, format="jd", scale="tdb") 
R_EARTH = 6378.0 
# Low Earth Orbit
LEO = (
    6745.592,       # Semimajor Axis, Pół oś wielka ~ 367 km nad pow
    0.01,           # Eccentricity, Mimośród półoś mała / półoś wielka
    7.81,           # Inclination, Inklinacja - wychylenie od równika
    100.21,         # RAAN, Omega długość węzła wstępującego RAAN kąt między kierunkiem na punkt Barana
    152.83,         # Argument of Perigee, omega argument perycentrum
    0.0 ,           # True Anomaly, Anomalia prawdziwa 
    )

# Sun Synchronous Orbit
SSO = (
    7153.12,           
    0.00,        
    98.4469 ,       
    212.741,
    0.0,     
    0.0,        
    )

# GeoSynchronous Orbit
GEO = (
    42164.0,           
    0.001,       
    0.0,       
    0.0,
    0.0,     
    0.0,        
    )

# Molniya
HEO = (
    26164.0,           
    0.74,       
    63.4,       
    0.0,
    0.0,     
    0.0,
    )