from numba import njit

@njit
def add(a, b):
    return a + b

print(add(2, 3))
