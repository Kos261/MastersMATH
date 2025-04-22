import spiceypy
from spiceypy import furnsh
import os

AU = 149597870.691

def getstsa():
    # utc_time = input("Input UTC Time (YYYY MM DD HH:MM:SS): ")
    utc_time = "2004 jun 11 19:32:00"

    et = spiceypy.str2et(utc_time)
    # str:16 -> box size 16, str.3f -> 3 digits
    print(f"Ephemeride Time in seconds past J2000: {et:16.3f}")

    # Computing apparent state of Phoebe as seen from CASSINI in the J2000 frame.
    # Units: km, km/s

    [state, ltime] = spiceypy.spkezr( 'PHOEBE', et,'J2000','LT+S','CASSINI')

    print('Apparent state of Phoebe as seen\nfrom CASSINI in the J2000 frame\n(km, km/s)')
    print(f"x: {state[0]}\ny: {state[1]}\nz: {state[2]}\n")
    print(f"vx: {state[3]}\nvy: {state[4]}\nvz: {state[5]}\n")
    
    print(f"One way Light Time CASS -> PHOEBE: {ltime}")


    [pos, ltime] = spiceypy.spkpos('EARTH', et,'J2000','LT+S','CASSINI')
    print('Earth position seen\nfrom CASSINI in the J2000 frame\n(km, km/s)')
    print(f"x: {pos[0]}\ny: {pos[1]}\nz: {pos[2]}\n")
    # print(f"vx: {pos[3]}\nvy: {pos[4]}\nvz: {pos[5]}\n")

    print(f"One way Light Time CASS -> EARTH: {ltime}")


    [pos, ltime] = spiceypy.spkpos('SUN', et,'J2000','NONE','PHOEBE')
    dist = spiceypy.vnorm(pos)
    dist_MY_AU = dist / AU
    print("Distance from SUN to PHOEBE my conversion: ",dist_MY_AU," AU")
    dist_AU = spiceypy.convrt( dist, 'KM', 'AU' )
    print("Distance from SUN to PHOEBE: ",dist_MY_AU," AU")


if __name__ == "__main__":
    kernel_path = "/home/konstanty/Projects/MastersMATH/kernels"
    kernels = ['naif0009.tls', 
               '981005_PLTEPH-DE405S.bsp',
               '020514_SE_SAT105.bsp',
               '030201AP_SK_SM546_T45.bsp']

    for kernel in kernels:
        print(f"Loaded {kernel}")
        furnsh(os.path.join(kernel_path, kernel))
        
    getstsa()