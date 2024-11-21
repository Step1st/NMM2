import pytest
import numpy as np

from simulation import _f, u_exact


def backward_error(x, t, h, tau, a):
    u, u_prev, u_next = u_exact(x, t), u_exact(x-h, t), u_exact(x+h, t)
    u_hat, u_hat_prev, u_hat_next = u_exact(x, t+tau), u_exact(x-h, t+tau), u_exact(x+h, t+tau)
    f, f_hat = _f(x, t, a), _f(x, t+tau, a)

    ut_aproximation = ((u_hat - u) / tau)
    uxx_aproximation = (1.0j * 0.5 * (
        ((u_hat_next - (2*u_hat) + u_hat_prev) / (h**2))
        + 
        ((u_next - (2*u) + u_prev) / (h**2))
        )) 
    
    nonlinear_aproximation = (a*0.5*(
        ((abs(u_hat)**2)*((u_hat_next-u_hat_prev)/(2*h)))
        + 
        ((abs(u)**2)*((u_next-u_prev)/(2*h)))
        ))

    f_aproximation = ((f_hat + f) * 0.5)

    return abs(ut_aproximation - uxx_aproximation - nonlinear_aproximation - f_aproximation)

@pytest.mark.parametrize("x,t", [(0.43, 0.23), (0.67, 2.76), (0.93, 1.04)])
def test_one(x, t):
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
    
    assert np.max(error_change) > 95