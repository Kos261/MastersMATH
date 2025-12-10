
# MastersMATH

# Data
Ephemerides from: https://ssd.jpl.nasa.gov/ftp/eph/planets/ascii/
Ephem: de442s.bsp - Chebyshev polynomials great for any time
Time: naif0012.tls
Mission: Mars 2020: https://pl.wikipedia.org/wiki/Mars_2020
1. Start: 30 lipca 2020.
2. TCM-1: 14 sierpnia 2020 
3. TCM-2: 30 września 2020 
4. TCM-3: ok. 20 grudnia 2020
5. Lądowanie: 18 lutego 2021.

# ToDo
- Energia układu -> Lagrange -> Hamiltonian
- Sprawdzić dokładność R4 na orbicie keplerowskiej (czy energia się zachowuje?).
- Na końcu wiele satelit + integrator geometryczny (Verlet, Leapfrog).
- Najpierw J2 + RK4 (stabilny rdzeń).
- Sprawdzić, czy orbita precesuje (zmiana RAAN i argumentu perygeum).
- Potem opór atmosferyczny (dla LEO).
- Później rozszerzyć do modelu NRLMSISE-00.
- Następnie Księżyc i Słońce (efemerydy SPICE).
- Sprawdzić błąd energii/całek ruchu.