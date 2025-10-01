#!/usr/bin/env python3
import sys
import math
import numpy as np
import argparse

def overlap_volume_mc_centers_along_axis(d, R1, R2, a, N, rng=None):
    """
    Monte Carlo estimate of overlap volume of two d-dimensional balls:
    Center1 at (0,0,…), Center2 at (a,0,0,…).
    Returns (estimate, standard_error).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Quick geometry checks
    D = abs(a)
    # No overlap
    if D >= R1 + R2:
        return 0.0, 0.0
    # One ball inside the other
    if D + min(R1, R2) <= max(R1, R2):
        # smaller ball entirely inside larger
        vol_small = (math.pi ** (d/2) / math.gamma(d/2 + 1)) * (min(R1, R2) ** d)
        return vol_small, 0.0

    # Build bounding box that encloses both balls
    # The min & max along each coordinate
    # For coordinate i = 0: min = min(0 - R1, a - R2), max = max(0 + R1, a + R2)
    # For other coordinates: min = -max(R1, R2), max = +max(R1, R2)
    mins = np.zeros(d)
    maxs = np.zeros(d)
    # For i = 0
    mins[0] = min(-R1, a - R2)
    maxs[0] = max(+R1, a + R2)
    # For i = 1..d−1
    for i in range(1, d):
        mins[i] = -max(R1, R2)
        maxs[i] = +max(R1, R2)

    widths = maxs - mins
    V_box = np.prod(widths)

    # Sample N points in the box
    u = rng.random((N, d))
    X = mins + u * widths  # each row is a sampled point

    # Distances
    # center1 = origin, center2 = (a, 0, 0, …)
    # So distance to center1 is norm(X)
    # Distance to center2 is norm(X - (a,0,0,…))
    d1 = np.linalg.norm(X, axis=1)
    # subtract a from the first coordinate
    X2 = np.copy(X)
    X2[:, 0] -= a
    d2 = np.linalg.norm(X2, axis=1)

    inside = (d1 <= R1) & (d2 <= R2)
    count = inside.sum()
    p = count / N

    est_vol = p * V_box
    # standard error (binomial) scaled to the volume
    std_err = math.sqrt(p * (1 - p) / N) * V_box

    return est_vol, std_err

def run_for_many_N(d, r1, r2, a, N_list, rng_seed=12345):
    """
    For a given dimension d, and geometry (r1, r2, a),
    run the Monte Carlo estimator for multiple N values,
    returning lists of (N, estimate, standard_error).
    """
    rng = np.random.default_rng(rng_seed)
    results = []
    for N in N_list:
        est, err = overlap_volume_mc_centers_along_axis(d, r1, r2, a, N, rng=rng)
        results.append((N, est, err))
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute overlap volume of two d-dimensional spheres (Monte Carlo)."
    )
    parser.add_argument("d", type=int, help="dimension d")
    parser.add_argument("N", type=int,
        help="number of Monte Carlo samples (if doing single-run mode)")
    parser.add_argument("r1", type=float, help="radius of first sphere")
    parser.add_argument("r2", type=float, help="radius of second sphere")
    parser.add_argument("a", type=float, help="center separation along one axis")
    parser.add_argument("--batch", nargs="*", type=int, metavar="N_i",
        help="(optional) list of N values to run batch mode (overrides single N mode)")
    parser.add_argument("--out", type=str, default=None,
        help="filename to write tabular results (columns: d, N, estimate, err)")
    return parser.parse_args()

def main():
    args = parse_args()

    d = args.d
    r1 = args.r1
    r2 = args.r2
    a = args.a

    # Decide whether to run a single N or a batch of N's
    if args.batch is not None and len(args.batch) > 0:
        N_list = args.batch
    else:
        N_list = [args.N]

    # Run
    results = run_for_many_N(d, r1, r2, a, N_list)

    # Print to stdout and optionally to file
    # Header
    print("# d   N   estimate   stderr")
    for (N, est, err) in results:
        print(f"{d}   {N}   {est:.12g}   {err:.12g}")

    if args.out is not None:
        try:
            with open(args.out, "a") as f:
                # if file is empty, print header
                if f.tell() == 0:
                    f.write("# d   N   estimate   stderr\n")
                for (N, est, err) in results:
                    f.write(f"{d}   {N}   {est:.12g}   {err:.12g}\n")
        except Exception as e:
            print(f"Error writing to {args.out}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

