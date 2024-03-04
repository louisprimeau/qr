
import numpy as np

def hh_vec(x):
    assert x.dtype == np.float64
    s = np.sum(x[1:]**2)
    v = x.copy(); v[0] = 1

    if s == 0 and x[0] >= 0: b = 0
    elif s == 0 and x[0] < 0: b = -2
    else:
        mu = np.sqrt(x[0]**2 + s)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -s / (x[0] + mu)
        b = 2 * v[0]**2 / (s + v[0]**2)
        v = v / v[0]
    return b, v

def hh_mat(x):
    b, v = hh_vec(x)
    return np.eye(x.shape[0]) - b * np.outer(v, v)

def hh_r(b, v, A):
    return A - b * v @ (v.T @ A)

def hh_l(b, v, A):
    return A - np.outer((A @ v), b * v)

def hh_tridiag(B):
    A = B.copy()
    assert np.all(A == A.T)
    n = A.shape[0]
    print(A)
    for k in range(0, n-2):
        b, v = hh_vec(A[k+1:, k])
        p = b * A[k+1:, k+1:] @ v
        w = p - (b * (p.T @ v) / 2) * v
        print(w, v)
        A[k+1, k] = np.linalg.norm(A[k+1:, k])
        A[k, k+1] = A[k+1, k]
        A[k+1:, k+1:] -= (np.outer(v, w) + np.outer(w, v))

        print(A)
    return A


A = np.random.randn(5, 5)
A = A + A.T

T = hh_tridiag(A)
print(T)
