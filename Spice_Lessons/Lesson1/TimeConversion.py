import os
import spiceypy
from spiceypy import furnsh

def convtm(utc_time):
    SCLKID = -82

    # Universal Time standard
    # utc_time = input(" Input UTC time (ex. 2004 jan 11 19:32:00): ")


    print(f" Converting UTC time: {utc_time:s}")

    #Ephemeris time
    et_time = spiceypy.str2et(utc_time)
    print(f" Ephemeris time past J2000: {et_time:16.3f}")

    #ET calendar time
    cal_et = spiceypy.etcal(et_time)
    print(f" Calendar ET time: {cal_et}")

    #Spacecraft clock time
    sc_clock = spiceypy.sce2c(SCLKID, et_time)
    print(f" Spacecraft clock time: {sc_clock}")



if __name__ == "__main__":
    kernel_path = "/home/konstanty/Projects/MastersMATH/kernels" 
    kernels = ['naif0009.tls', 'cas00084.tsc']

    for kernel in kernels:
        print(f"Loaded {kernel}")
        furnsh(os.path.join(kernel_path, kernel))
        
    convtm("2004 jan 11 19:32:00")