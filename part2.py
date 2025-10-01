import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial

try:
    from scipy.stats import qmc
    have_sobol = True
except ImportError:
    have_sobol = False
    print("Warning: scipy.stats.qmc not available; need external Sobol library")

def true_volume_5d_unit_ball():
    # Volume of unit ball in dimension 5: π^{5/2} / Γ(5/2 + 1)
    return math.pi**(2.5) / math.gamma(2.5 + 1)

def estimate_mc_pseudorandom(N, dim=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # sample in [-1,1]^dim
    X = rng.random((N, dim)) * 2.0 - 1.0
    norms = np.linalg.norm(X, axis=1)
    inside = (norms <= 1.0)
    count = inside.sum()
    # volume of hypercube = 2^dim
    vol = count / N * (2.0**dim)
    return vol

def estimate_mc_sobol(N, dim=5):
    if not have_sobol:
        raise RuntimeError("Sobol sampler not available in scipy; install or use another library")
    sampler = qmc.Sobol(d=dim, scramble=False)  # or scramble=True
    # Sobol requires N to be power of 2 (for good properties)
    U = sampler.random_base2(m=int(math.log2(N)))
    # U ∈ [0,1]^dim, map to [-1,1]
    X = 2.0 * U - 1.0
    norms = np.linalg.norm(X, axis=1)
    inside = (norms <= 1.0)
    count = inside.sum()
    vol = count / N * (2.0**dim)
    return vol

def estimate_grid_fixed(m, dim=5):
    """
    Uniform grid in [-1,1]^dim with m points per axis.
    Evaluate indicator at each grid point, multiply by (2/m)^dim volume element.
    """
    # Use a meshgrid (careful with memory), or iterate
    # We'll do an iterative approach to avoid huge memory
    coords_1d = np.linspace(-1.0, 1.0, m)
    # The volume element
    dv = (2.0 / (m - 1))**dim  # note: if using endpoints inclusive
    count = 0
    total = m**dim
    # iterate via recursion or looping
    # Here’s a recursive python generator to yield all grid points:
    def rec_gen(dim_left, prefix):
        if dim_left == 0:
            yield np.array(prefix)
        else:
            for x in coords_1d:
                yield from rec_gen(dim_left - 1, prefix + [x])
    for point in rec_gen(dim, []):
        if np.linalg.norm(point) <= 1.0:
            count += 1
    vol = count * dv
    return vol

def run_experiments(N_list, grid_ms):
    true_vol = true_volume_5d_unit_ball()
    results = {'N': [], 'err_mc': [], 'err_sobol': [], 'err_grid': []}
    rng = np.random.default_rng(12345)

    for N in N_list:
        vol_mc = estimate_mc_pseudorandom(N, dim=5, rng=rng)
        err_mc = abs(vol_mc - true_vol) / true_vol
        results['N'].append(N)
        results['err_mc'].append(err_mc)

        if have_sobol:
            vol_sob = estimate_mc_sobol(N, dim=5)
            err_sob = abs(vol_sob - true_vol) / true_vol
        else:
            err_sob = None
        results['err_sobol'].append(err_sob)

    for m in grid_ms:
        vol_g = estimate_grid_fixed(m, dim=5)
        err_g = abs(vol_g - true_vol) / true_vol
        results['err_grid'].append((m, err_g))

    return true_vol, results

def plot_errors(true_vol, results):
    N = np.array(results['N'])
    err_mc = np.array(results['err_mc'])
    err_sob = np.array(results['err_sobol'], dtype=float)

    # For grid, we have (m, err) pairs
    m_vals, err_grid = zip(*results['err_grid'])
    sqrtN = np.sqrt(N)

    plt.figure(figsize=(8,6))
    # Monte Carlo error
    plt.plot(sqrtN, err_mc, 'o-', label="Pseudorandom MC")
    if have_sobol:
        plt.plot(sqrtN, err_sob, 's-', label="Sobol (quasirandom)")
    # For grid, convert m to approximate N = m^5
    N_grid = np.array(m_vals)**5
    plt.plot(np.sqrt(N_grid), err_grid, 'x-', label="Fixed grid")

    plt.xlabel(r"$\sqrt{N}$")
    plt.ylabel("Relative error")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Relative error vs sqrt(N) for 5D unit ball volume")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.savefig("methods.png", dpi=200)
    plt.show()

def main():
    # choose a list of sample sizes for MC / Sobol (powers of 2 if using Sobol)
    N_list = [2**k for k in range(6, 14)]  # from 64 up to 8192 (or higher as you like)
    # choose grid resolutions, e.g. from m=5 to m=15
    grid_ms = [5, 7, 9, 11, 13]  # so grid N = m^5
    true_vol, results = run_experiments(N_list, grid_ms)
    print("True 5D unit-ball volume:", true_vol)
    for (N, err) in zip(results['N'], results['err_mc']):
        print(f"N={N}, MC rel error = {err:.3e}")
    if have_sobol:
        for (N, err) in zip(results['N'], results['err_sobol']):
            print(f"N={N}, Sobol rel error = {err:.3e}")
    for (m, err) in results['err_grid']:
        print(f"m={m}, grid rel error = {err:.3e}")

    plot_errors(true_vol, results)

if __name__ == "__main__":
    main()
    
