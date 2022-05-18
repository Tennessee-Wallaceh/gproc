import numpy as np
from gproc.example import addc, set_num_threads, parallel_test, add_numpyc

print(f'Adding: {addc(2, 3)}')

# Runs parallel threads with omp
print('\n============== PARALLEL TEST =============\n')
parallel_test(1000)

# Control omp
print('\n============== PARALLEL TEST 1 THREAD =============\n')
set_num_threads(1)
parallel_test(1000)

# Interact with numpy
print('\n============== NUMPY TEST =============\n')
a = np.linspace(-1, 3, 10)
b = np.linspace(0, 4, 10)

c = add_numpyc(a, b)

print(c)