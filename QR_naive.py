
import jax
from jax import numpy as np
from jax import random
from jax.numpy import linalg


seed = 1701
key = random.PRNGKey(seed)

N = 5
b = random.randint(key, (N, N), 0, 2000)
A = (b + b.T) / 2

T = A
print(T)
for i in range(100):
    
    Qi, Ri = linalg.qr(T)
    T = Ri @ Qi

print(np.diag(T).sort())
print(np.linalg.eigh(A).eigenvalues.sort())



