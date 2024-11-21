import pytest
import numpy as np

from simulation import _C, calculate_next, u_exact, kapa_1


def error(T, h, N, tau, C, a):
    t = 0
    u = np.array([u_exact(x, 0) for x in np.arange(0.0, 1.0, h)])
   
    alpha = np.zeros(N, dtype=np.complex128)
    alpha[1] = kapa_1
    for i in range(1, N-1):
        alpha[i+1] = (1 / (C - alpha[i]))

    max_error = 0
    while(t < T):
        u_next = calculate_next(u=u, t=t, h=h, tau=tau, N=N, alpha=alpha, a=a)
        
        u_next_tikslus = np.array([u_exact(x, t+tau) for x in np.arange(0.0, 1.0, h)])
        
        error = np.max(np.abs(u_next - u_next_tikslus))
        
        max_error = error if error > max_error else max_error
        u = u_next
        t=t+tau
    return max_error


@pytest.mark.parametrize("T", [0.5, 1])
def test_global(T):
    N = 1000
    h, tau = 1/N, 0.001
    C = _C(h, tau)
    a = 0.2
    errors = [error(T, h, N, tau, C, a)]
    error_change = []
    for i in range (1, 4):
        N = N * 2
        h = 1 / N
        tau = tau * 0.5
        C = _C(h, tau)
        errors.append(error(T, h, N, tau, C, a))
        error_change.append(errors[i-1]/errors[i])
        print(f"N: {N}, tau: {"{:.11f}".format(tau)}, Error Change: {"{:.11f}".format(errors[i-1]/errors[i])} times")

    
    assert np.max(error_change) > 1.8

