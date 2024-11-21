import numpy as np

from simulation import thomas_algorithm, _C, kapa_1, kapa_2, gamma_1, gamma_2


def test_three():
    N = 10000
    h, tau = 1/N, 0.00001
    C = _C(h, tau)

    y = (np.random.rand(N)+(1j*np.random.rand(N))).astype(dtype=complex)
    y[0] = kapa_1*y[1]+gamma_1
    y[N-1] = kapa_2*y[N-2]+gamma_2


    F = np.zeros(N, dtype=complex)
    for j in range(1, N-1):
        F[j] = (C*y[j]) - y[j+1] - y[j-1]

    alpha = np.zeros(N, dtype=np.complex128)
    alpha[1] = kapa_1
    for i in range(1, N-1):
        alpha[i+1] = (1 / (C - alpha[i]))

    y_new = thomas_algorithm(alpha, F, N)
    assert (np.max(abs(y_new - y)) < 1e-13)
    
