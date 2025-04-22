import os
from spiceypy import furnsh

# Get the absolute path to the project root
kernel_path = "/home/konstanty/Projects/MastersMATH/kernels"  # Adjust as needed

# Load kernels dynamically
kernels = [
    'naif0009.tls',
    '981005_PLTEPH-DE405S.bsp',
    '020514_SE_SAT105.bsp',
    '030201AP_SK_SM546_T45.bsp'
]

for kernel in kernels:
    print(f"Loaded {kernel}")
    furnsh(os.path.join(kernel_path, kernel))