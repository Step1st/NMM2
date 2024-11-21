import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser
from numpy import sin, cos, pi, exp, abs


parser = ArgumentParser("Model")
parser.add_argument("-a", "--alpha", nargs='*', type=float, default=[0.1])
parser.add_argument("-N", "--resolution", type=int, default=1000)
parser.add_argument("-T", "--duration", type=float, default=1.0)
parser.add_argument("-t", "--timestep", type=float, default=0.001)
parser.add_argument("-p", "--parallel", action='store_true')
parser.add_argument("--csv", action='store_true')
parser.add_argument("--fps", type=int)

args = parser.parse_args()

kapa_1, kapa_2 = 1, 1
gamma_1, gamma_2 = 0, 0

def u_exact(x, t) -> complex:
    return (cos(2*pi*x))+(exp(1j*(pi*t)))


def _f(x, t, a) -> complex:
    return pi*(2*a*((exp(1j*pi*t)*cos(2*pi*x) + cos(2*pi*x)**2 + 1)*exp(1j*pi*t) + cos(2*pi*x))*sin(2*pi*x) + 1j*(1.0*exp(1j*pi*t) + 4.0*pi*cos(2*pi*x))*exp(1j*pi*t))*exp(-1j*pi*t)


def _C(h, tau) -> complex:
    return 2-((1j*2*(h**2))/tau)


def _F(u, u_prev, u_next, y, y_prev, y_next, f, f_hat, h, tau, a):
    nonlinear = ((a*(-1j)*(h**2)) * (((abs(y)**2)*((y_next-y_prev)/(2*h))) + ((abs(u)**2)*((u_next-u_prev)/(2*h)))))

    functional = (-1j)*(h**2)*(f_hat+f)

    return ((((-1j)*2*(h**2))/tau)*u) - (2*u) + u_next + u_prev + nonlinear + functional


def thomas_algorithm(alpha, F: np.ndarray, N: int) -> np.ndarray[np.complex128]:
    beta = np.zeros(N, dtype=np.complex128)
    beta[1] = gamma_1
    for i in range(1, N-1):
        beta[i+1] = (beta[i] + F[i]) * alpha[i+1]

    y = np.zeros(N, dtype=np.complex128)
    y[N-1] = ((kapa_2 * beta[N-1]) + gamma_2) / (1 - (kapa_2 * alpha[N-1]))

    for i in range(N-1, 0, -1):
        y[i-1] = alpha[i] * y[i] + beta[i]

    return y


def calculate_next(u, t, h, tau, N, alpha, a) -> np.ndarray[np.complex128]:
    y_old = u
    while True:
        f, f_hat = np.zeros(N, dtype=np.complex128), np.zeros(N, dtype=np.complex128)
        x = np.arange(0, 1, h)
        f, f_hat = _f(x, t, a), _f(x, t+tau, a)

        F = np.zeros(N, dtype=np.complex128)
        F[1:N-1] = _F(u=u[1:N-1], u_prev=u[0:N-2], u_next=u[2:N],
                      y=y_old[1:N-1], y_prev=y_old[0:N-2], y_next=y_old[2:N],
                      f=f[1:N-1], f_hat=f_hat[1:N-1], h=h, tau=tau, a=a)

        y_new = thomas_algorithm(alpha=alpha, F=F, N=N)

        if (np.max(np.abs(y_new - y_old)) < 0.0000001):
            break

        if (np.isnan(np.max(np.abs(y_new - y_old)))):
            raise Exception("Error: Encountered NaN") 
        
        y_old = y_new
    return y_old


def compute_simulation(h, tau, N, T, a):
    C = _C(h, tau)

    kapa_1 = 1
    alpha = np.zeros(N, dtype=np.complex128)
    alpha[1] = kapa_1
    for i in range(1, N-1):
        alpha[i+1] = (1 / (C - alpha[i]))
    
    x = np.arange(0.0, 1.0, h)
    u_0 = u_exact(x, 0)
    u = u_0

    evolution = [u_0]

    t = 0
    while(t < T):
        u_next = calculate_next(u=u, t=t, h=h, tau=tau, N=N, alpha=alpha, a=a)

        evolution.append(u_next)
        
        u = u_next
        t=t+tau
   
    return np.array(evolution)


def make_animation(h, tau, a, evolution):
    fig, ax = plt.subplots()

    x = np.arange(0, 1, h)
    line, = ax.plot(x, evolution[0])

    def animate(i):
        line.set_ydata(np.absolute(i))
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=evolution, interval=1, blit=True)

    fps = args.fps if args.fps else int(1/tau)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(f"animations/animation_a_{a}.mp4", writer=writer)


def run_simulation(a):
    N = args.resolution
    T = args.duration
    h, tau = 1/N, args.timestep

    evolution = compute_simulation(h, tau, N, T, a)

    if (args.csv):
        np.savetxt(f"csv/csn_a_{a}.csv", evolution, delimiter=',')
    else:
        make_animation(h, tau, a, evolution)


def main():
    alpha = args.alpha
    if (args.parallel and len(alpha) > 1):
        
        with multiprocessing.Pool(min(len(alpha), multiprocessing.cpu_count())) as pool:
            pool.map(run_simulation, alpha)
    else:
        run_simulation(alpha[0])


if __name__ == "__main__":
    main()