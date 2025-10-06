#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_results(file):
    data = np.loadtxt(file, comments="#")
    dcol = data[:,0].astype(int)
    N = data[:,1].astype(int)
    est = data[:,2]
    err = data[:,3]
    return dcol, N, est, err

def make_combined_plot(files, analytic3=None, output="convergence.png"):
    n = len(files)
    if n == 0:
        print("No files given.")
        return

    # Layout: n rows × 2 columns (left: estimate, right: stderr)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 4*n), squeeze=False)

    for i, (fn, label) in enumerate(files):
        dcol, N, est, err = load_results(fn)
        d = dcol[0]

        ax_est = axes[i][0]
        ax_err = axes[i][1]

        # Estimate plot
        ax_est.errorbar(np.sqrt(N), est, yerr=err, fmt='o-', label=f"d={d} {label}")
        if analytic3 is not None and d == 3:
            ax_est.axhline(analytic3, color='red', linestyle='--', label="analytic 3D")
        ax_est.set_xlabel("sqrt(N)")
        ax_est.set_ylabel("Estimated overlap")
        ax_est.legend()
        ax_est.grid(True)

        # Uncertainty (stderr) plot
        ax_err.plot(np.sqrt(N), err, 'o-', label=f"d={d} {label}")
        ax_err.set_xlabel("sqrt(N)")
        ax_err.set_ylabel("stderr")
        ax_err.legend()
        ax_err.grid(True)

        # You might add a title for each row
        ax_est.set_title(f"File: {fn}")
        ax_err.set_title(f"stderr (File {fn})")

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    fig.subplots_adjust(
        left=0.1, right=0.95, top=0.95, bottom=0.05,
        wspace=0.4,  # horizontal space between columns
        hspace=0.6   # vertical space between rows
    )
    print(f"Saved combined plot to {output}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot all results together in one figure")
    parser.add_argument("files", nargs="+", help="Result files, e.g. results_d3.txt results_d5.txt")
    parser.add_argument("--analytic3", type=float, default=None,
        help="Optional analytic overlap value in 3D to draw reference line")
    parser.add_argument("--out", type=str, default="convergence.png", help="Output image file")
    args = parser.parse_args()

    file_labels = [(fn, "") for fn in args.files]
    make_combined_plot(file_labels, analytic3=args.analytic3, output=args.out)

if __name__ == "__main__":
    main()

