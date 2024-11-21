import pytest
import numpy as np

from simulation import _f, u_exact, _F, _C


def backward_error(x, t, h, tau, a):
    u, u_prev, u_next = u_exact(x, t), u_exact(x-h, t), u_exact(x+h, t)
    u_hat, u_hat_prev, u_hat_next = u_exact(x, t+tau), u_exact(x-h, t+tau), u_exact(x+h, t+tau)
    f, f_hat = _f(x, t, a), _f(x, t+tau, a)

    C = _C(h, tau)
    F = _F(u, u_prev, u_next, u_hat, u_hat_prev, u_hat_next, f, f_hat, h, tau, a)

    return np.abs((u_hat_next-(C*u_hat)+u_hat_prev)+F)

@pytest.mark.parametrize("x,t", [(0.43, 0.23), (0.67, 2.76), (0.93, 1.04)])
def test_two(x, t):
    step = 0.1
    h = 0.1
    tau = 0.1
    a = 0.2
    errors = [backward_error(x, t, h, tau, a)]
    error_change = []
    for i in range (1, 11):
        h = h * step
        tau = tau * step
        errors.append(backward_error(x, t, h, tau, a))
        error_change.append(errors[i-1]/errors[i])
        print(f"h: {"{:.11f}".format(h)}, tau: {"{:.11f}".format(tau)}, Error Change: {"{:.11f}".format(errors[i-1]/errors[i])} times")
    
    assert np.max(error_change) > 9500