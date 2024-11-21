import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, default=1000)
parser.add_argument("-t", "--timestep", type=float, default=0.001)
parser.add_argument("-f", "--file", nargs='*', type=argparse.FileType("r"), required=True)
parser.add_argument("--fps", type=int)
args = parser.parse_args()


def make_animation(h, tau, evolution):
    fig, ax = plt.subplots()

    x = np.arange(0, 1, h)
    line, = ax.plot(x, evolution[0])

    def animate(i):
        line.set_ydata(np.absolute(i))
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=evolution, interval=1, blit=True)

    fps = args.fps if args.fps else int(1/tau)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(f"animations/animation.mp4", writer=writer)


def main():
    N = args.resolution
    h, tau = 1/N, args.timestep
    evolution = np.loadtxt(args.file[0], delimiter=',', dtype=np.complex128)
    make_animation(h, tau, evolution)

if __name__ == "__main__":
    main()