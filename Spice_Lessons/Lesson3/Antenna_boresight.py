# The HGA boresight direction is direction of strongest antenna signal
import os
import spiceypy
from spiceypy import furnsh


def xform():
    utctime = "2004 jun 11 19:32:00"
    et = spiceypy.str2et(utctime)

    [state, ltime] = spiceypy.spkezr('PHOEBE', et,'J2000','LT+S','CASSINI')
    #
    # Now obtain the transformation matrix 6x6:
    # from the inertial J2000 frame to the non-inertial body-fixed IAU_PHOEBE frame.  
    # Since we want the apparent position, we need to subtract ltime from et.
    #
    sform = spiceypy.sxform( 'J2000', 'IAU_PHOEBE', et-ltime )

    # Multiplies the 6x6 matrix sform with the 6x1 vector state 
    # to transform the state from J2000 to IAU_PHOEBE.
    bfixst = spiceypy.mxvg(sform, state)

    print( '   Apparent state of Phoebe as seen from CASSINI in the IAU_PHOEBE\n'
           '      body-fixed frame (km, km/s):'      )
    print(f'X  = {bfixst[0]:19.6f}')
    print(f'Y  = {bfixst[1]:19.6f}')
    print(f'Z  = {bfixst[2]:19.6f}') 
    print(f'VX = {bfixst[3]:19.6f}')
    print(f'VY = {bfixst[4]:19.6f}')
    print(f'VZ = {bfixst[5]:19.6f}')


    # all of that ^ with sinle function
    #             |                                ref frame
    [state, ltime] = spiceypy.spkezr('PHOEBE',et,'IAU_PHOEBE','LT+S','CASSINI')


    # Now we are to compute the angular separation (angular difference) between
    # the apparent position of the Earth as seen from the
    # orbiter and the nominal boresight of the high gain
    # antenna.  First, compute the apparent position of
    # the Earth as seen from CASSINI in the J2000 frame.
    #
    [pos, ltime] = spiceypy.spkpos('EARTH',et,'J2000','LT+S','CASSINI')

    antenna_bsight = [ 0.0, 0.0, 1.0]

    # Now compute the rotation matrix from CASSINI_HGA into J2000.
    pform = spiceypy.pxform( 'CASSINI_HGA', 'J2000', et )
    # Antenna in J2000 ref frame
    antenna_bsight = spiceypy.mxv(pform, antenna_bsight)

    angular_sep =  spiceypy.convrt( spiceypy.vsep(antenna_bsight, pos), 'RADIANS', 'DEGREES')
    print(f'Angular separation between the apparent position of\nEarth and the CASSINI high gain antenna boresight (degrees): {angular_sep:16.3f}')


if __name__ ==  '__main__':
    kernel_path = "/home/konstanty/Projects/MastersMATH/kernels"

    kernels = ['naif0009.tls',
               'cas00084.tsc',
               '981005_PLTEPH-DE405S.bsp',
               '020514_SE_SAT105.bsp',
               '030201AP_SK_SM546_T45.bsp',
               'cas_v37.tf',
               '04135_04171pc_psiv2.bc',
               'cpck05Mar2004.tpc']
    
    for kernel in kernels:
        print(f"Loaded {kernel}")
        furnsh(os.path.join(kernel_path, kernel))
        
    xform()