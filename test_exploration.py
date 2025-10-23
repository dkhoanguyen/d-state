#!/usr/bin/env python3
"""
Dynamic Target Belief (agent speed + CIRCULAR sensing) — KALMAN-STYLE localization
----------------------------------------------------------------------------------
When object FOUND: 
- Create Gaussian blob at object location with variance σ = k / distance
- Closer robot = sharper peak (smaller σ)
- Background stays uniform (no purple patches)
- Belief continues evolving naturally
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Dynamic target belief with Kalman-style localization.")
    p.add_argument("--speed", type=int, default=1, help="Robot step size (cells per step).")
    p.add_argument("--sense", type=float, default=5.0, help="Sensing radius (cells, CIRCULAR).")
    p.add_argument("--bc", type=str, default="reflect", choices=["reflect", "wrap", "absorb"],
                   help="Boundary condition for belief motion model.")
    p.add_argument("--diff", type=float, default=0.7, help="Belief diffusion sigma (cells per step).")
    p.add_argument("--drift_i", type=int, default=0, help="Integer drift in i (row) per step.")
    p.add_argument("--drift_j", type=int, default=0, help="Integer drift in j (col) per step.")
    p.add_argument("--steps", type=int, default=2200, help="Number of simulation steps.")
    p.add_argument("--grid", type=int, default=40, help="Grid size N (NxN).")
    p.add_argument("--seed", type=int, default=2, help="Random seed.")
    p.add_argument("--kalman_k", type=float, default=8.0, help="Kalman scaling: σ = k / distance")
    return p.parse_args()

# -----------------------------
# Helpers
# -----------------------------
def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def renormalize(P):
    s = P.sum()
    if s <= 1e-15:
        return np.full_like(P, 1.0 / P.size)
    return P / s

def gaussian_kernel(size, sigma):
    k = int(size)
    ax = np.arange(-k//2 + 1, k//2 + 1)
    Xk, Yk = np.meshgrid(ax, ax)
    K = np.exp(-(Xk**2 + Yk**2) / (2 * max(1e-6, sigma)**2))
    K /= K.sum()
    return K

def pad_mode_array(P, pad, mode):
    if mode == "reflect":
        return np.pad(P, pad_width=pad, mode='reflect')
    elif mode == "wrap":
        return np.pad(P, pad_width=pad, mode='wrap')
    elif mode == "absorb":
        return np.pad(P, pad_width=pad, mode='constant', constant_values=0.0)
    else:
        raise ValueError("Unknown boundary mode")

def diffuse_with_kernel(P, sigma, mode="reflect"):
    if sigma <= 1e-6:
        return P
    k = max(3, int(np.ceil(6 * sigma)) | 1)  # odd
    K = gaussian_kernel(k, sigma)
    pad = k // 2
    Ppad = pad_mode_array(P, pad, mode)
    out = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            patch = Ppad[i:i + k, j:j + k]
            out[i, j] = np.sum(patch * K)
    return renormalize(out)

def shift_reflect(P, di, dj):
    out = np.zeros_like(P)
    ni, nj = P.shape
    ii = np.arange(ni) - di
    jj = np.arange(nj) - dj
    def reflect_index(idx, n):
        idx = np.array(idx, dtype=int)
        if n == 1: return np.zeros_like(idx)
        mod = 2 * (n - 1)
        r = idx % mod
        r = np.where(r < 0, r + mod, r)
        r = np.where(r <= (n - 1), r, 2 * (n - 1) - r)
        return r
    I2 = reflect_index(ii, ni)
    J2 = reflect_index(jj, nj)
    out[:, :] = P[I2][:, J2]
    return out

def shift_absorb(P, di, dj):
    out = np.zeros_like(P)
    ni, nj = P.shape
    I, J = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
    I2 = I - int(di)
    J2 = J - int(dj)
    mask = (I2 >= 0) & (I2 < ni) & (J2 >= 0) & (J2 < nj)
    out[I[mask], J[mask]] = P[I2[mask], J2[mask]]
    return renormalize(out)

def apply_drift(P, di, dj, mode="reflect"):
    if di == 0 and dj == 0:
        return P
    if mode == "wrap":
        return np.roll(P, shift=(int(di), int(dj)), axis=(0, 1))
    elif mode == "reflect":
        return shift_reflect(P, int(di), int(dj))
    elif mode == "absorb":
        return shift_absorb(P, int(di), int(dj))
    else:
        raise ValueError("Unknown boundary mode")

def circular_mask_indices(ri, rj, r_cells, N):
    """Boolean mask for a CIRCLE of radius r_cells around (ri, rj)."""
    I, J = np.ogrid[:N, :N]
    return (I - ri)**2 + (J - rj)**2 <= (r_cells**2)

def kalman_localization(P, obj_i, obj_j, ri, rj, k=8.0, eps=1e-8):
    """
    KALMAN-STYLE: When object found
    - Distance d = sqrt((ri-obj_i)^2 + (rj-obj_j)^2)
    - Variance σ = k / d  (closer = sharper peak)
    - Uniform background + Gaussian blob at object
    """
    # Distance from robot to object
    d = np.sqrt((ri - obj_i)**2 + (rj - obj_j)**2)
    # sigma = max(1.0, max(d, 0.001))  # min σ=0.3 for visibility
    sigma = 1.0
    
    # Create Gaussian centered at object
    I, J = np.ogrid[:P.shape[0], :P.shape[1]]
    gaussian = np.exp(-((I - obj_i)**2 + (J - obj_j)**2) / (2 * sigma**2))
    
    # Background level (tiny uniform)
    bg_level = eps
    
    # Combine: background + scaled Gaussian
    P = bg_level + (1.0 - P.size * bg_level) * (gaussian / gaussian.sum())
    
    return P

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    N = args.grid
    T = args.steps
    step_size = max(1, int(args.speed))
    sense_radius = max(0.0, float(args.sense))
    boundary_condition = args.bc
    sigma_move = float(args.diff)
    drift_vec = (int(args.drift_i), int(args.drift_j))
    kalman_k = float(args.kalman_k)

    # Negative evidence falloff
    neg_info_strength = 0.85
    neg_info_rho = max(1e-6, sense_radius * 0.8)

    # For plotting
    xs = np.linspace(0.5 / N, 1 - 0.5 / N, N)
    ys = np.linspace(0.5 / N, 1 - 0.5 / N, N)

    # Ground truth
    obj_i, obj_j = int(0.7 * N), int(0.25 * N)

    # Belief (uniform)
    P = np.full((N, N), 1.0 / (N * N))

    # Robot indices
    ri, rj = int(0.07 * N), int(0.07 * N)

    # Logs
    H_vals = [entropy(P)]
    visited = np.zeros((N, N), dtype=bool)
    object_found = False

    # --- Plot setup ---
    plt.ion()
    fig = plt.figure(figsize=(12.5, 5.0))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title(f"Kalman Belief (k={kalman_k}, diff={sigma_move}, speed={step_size}, sense={sense_radius})")
    img = ax1.imshow(P, origin="lower", extent=[0, 1, 0, 1],
                     cmap="viridis", interpolation="nearest", vmin=0, vmax=P.max())
    cbar = fig.colorbar(img, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("P(object at cell)")

    rob_plot, = ax1.plot([xs[rj]], [ys[ri]], 'wo', ms=6, label="Robot")
    obj_plot, = ax1.plot([xs[obj_j]], [ys[obj_i]], 'r*', ms=11, label="Object")
    sense_circle = plt.Circle((xs[rj], ys[ri]), sense_radius / N, color='w', fill=False, ls='--', alpha=0.8)
    ax1.add_patch(sense_circle)

    ax1.legend(loc="upper right")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Belief entropy H(p)")
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("H(p)")
    line_H, = ax2.plot([0], [H_vals[0]], '-')
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, np.log(N * N) * 1.05)

    # --- Simulation loop ---
    for t in range(1, T + 1):
        # 0) Random move
        if np.random.rand() < 0.5:
            ri = int(np.clip(ri + np.random.choice([-step_size, step_size]), 0, N - 1))
        else:
            rj = int(np.clip(rj + np.random.choice([-step_size, step_size]), 0, N - 1))

        # 1) Belief motion: drift then diffuse (EVEN AFTER FOUND!)
        if drift_vec != (0, 0):
            P = apply_drift(P, drift_vec[0], drift_vec[1], mode=boundary_condition)
        if sigma_move > 1e-6:
            P = diffuse_with_kernel(P, sigma=sigma_move, mode=boundary_condition)

        # 2) Circular sensing
        circle_mask = circular_mask_indices(ri, rj, sense_radius, N)
        visited |= circle_mask

        # 2a) Found? (object cell in circle)
        if (obj_i - ri)**2 + (obj_j - rj)**2 <= sense_radius**2:
            object_found = True
            # KALMAN UPDATE: Gaussian blob with distance-dependent variance
            P = kalman_localization(P, obj_i, obj_j, ri, rj, k=kalman_k, eps=1e-8)
        else:
            # 2b) Negative evidence INSIDE circle
            I, J = np.where(circle_mask)
            di = I - ri
            dj = J - rj
            d2 = di * di + dj * dj
            w = neg_info_strength * np.exp(-0.5 * d2 / (neg_info_rho ** 2))
            P[I, J] *= (1.0 - w)
            P = renormalize(P)

        # 3) Logging
        H_vals.append(entropy(P))

        # 4) Plot updates
        img.set_data(P)
        img.set_clim(vmin=0.0, vmax=max(P.max(), 1e-12))
        rob_plot.set_data([xs[rj]], [ys[ri]])
        sense_circle.center = (xs[rj], ys[ri])
        sense_circle.set_radius(sense_radius / N)

        line_H.set_data(np.arange(len(H_vals)), H_vals)
        ax2.set_ylim(0, max(ax2.get_ylim()[1], max(H_vals) * 1.05))

        # Show distance & sigma in xlabel
        d = np.sqrt((ri - obj_i)**2 + (rj - obj_j)**2)
        sigma_est = max(0.3, kalman_k / max(d, 0.1))
        ax1.set_xlabel(
            f"t={t:03d} | found={object_found} | dist={d:.1f} | σ={sigma_est:.2f} | mass@true={P[obj_i, obj_j]:.3f}"
        )
        plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()