#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_results(file):
    data = np.loadtxt(file, comments="#")
    # columns: d, N, estimate, stderr
    dcol = data[:,0].astype(int)
    N = data[:,1].astype(int)
    est = data[:,2]
    err = data[:,3]
    return dcol, N, est, err

def make_plot(files, analytic3=None, output="convergence.png"):
    """
    files: list of (filename, label) pairs
    analytic3: if provided, a float giving the analytic overlap in d=3
    """
    plt.figure(figsize=(8, 10))

    # Upper subplot: estimates with error bars
    ax1 = plt.subplot(2, 1, 1)
    for (fn, label) in files:
        dcol, N, est, err = load_results(fn)
        # all runs in a file share same d, so pick first
        d = dcol[0]
        ax1.errorbar(np.sqrt(N), est, yerr=err, fmt='o-', label=f"d={d}, {label}")
    if analytic3 is not None:
        # draw horizontal line
        ax1.axhline(analytic3, color='red', linestyle='--', label="analytic 3D")
    ax1.set_xlabel("sqrt(N)")
    ax1.set_ylabel("Estimated overlap volume")
    ax1.legend()
    ax1.grid(True)

    # Lower subplot: uncertainty vs N (log-log)
    ax2 = plt.subplot(2, 1, 2)
    for (fn, label) in files:
        dcol, N, est, err = load_results(fn)
        d = dcol[0]
        ax2.plot(np.sqrt(N), err, 'o-', label=f"d={d}, {label}")
    ax2.set_xlabel("sqrt(N)")
    ax2.set_ylabel("Statistical uncertainty (stderr)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Saved plot to {output}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot convergence from Monte Carlo output files")
    parser.add_argument("files", nargs="+", help="Result files, e.g. results_d3.txt results_d5.txt")
    parser.add_argument("--analytic3", type=float, default=None,
        help="Optional analytic overlap value in 3D to draw reference line")
    parser.add_argument("--out", type=str, default="convergence.png", help="Output image file")
    args = parser.parse_args()

    file_labels = [(fn, "") for fn in args.files]
    make_plot(file_labels, analytic3=args.analytic3, output=args.out)

if __name__ == "__main__":
    main()

