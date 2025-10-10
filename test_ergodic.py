import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds

# ============================================================
# Modular, N-robot, decentralized Ergodic CLF-only controller
# - Tunable Gaussian Mixture rho(x)
# - Rectangular domain with cosine Fourier basis
# - Each robot solves a *local* CLF-QP
# - Low-bandwidth shared step = average of F_k(x_j) across robots
# ============================================================

# -------------------- Fourier basis --------------------
class FourierBasis:
    def __init__(self, L, Kmax):
        """
        L: array-like, domain size [Lx, Ly]
        Kmax: tuple (Kx, Ky) inclusive maximum orders
        """
        self.L = np.asarray(L, dtype=float)
        self.Kmax = tuple(int(k) for k in Kmax)
        self.modes = [(kx, ky)
                      for kx in range(self.Kmax[0] + 1)
                      for ky in range(self.Kmax[1] + 1)]
        self.nK = len(self.modes)
        # Sobolev weights Λ_k
        n = len(self.L)
        self.Lam = np.array([1.0 / (1.0 + kx*kx + ky*ky) ** ((n + 1) / 2.0)
                             for (kx, ky) in self.modes], dtype=float)

    def F_k(self, x, k):
        kx, ky = k
        Lx, Ly = self.L
        return np.cos(np.pi * kx * x[0] / Lx) * np.cos(np.pi * ky * x[1] / Ly)

    def gradF_k(self, x, k):
        kx, ky = k
        Lx, Ly = self.L
        gx = -(np.pi * kx / Lx) * np.sin(np.pi * kx * x[0] / Lx) * np.cos(np.pi * ky * x[1] / Ly)
        gy = -(np.pi * ky / Ly) * np.cos(np.pi * kx * x[0] / Lx) * np.sin(np.pi * ky * x[1] / Ly)
        return np.array([gx, gy], dtype=float)

    def F_vec(self, x):
        return np.array([self.F_k(x, m) for m in self.modes], dtype=float)

    def gradF_mat(self, x):
        # shape (nK, 2)
        return np.array([self.gradF_k(x, m) for m in self.modes], dtype=float)


# -------------------- Gaussian Mixture density over a grid --------------------
class GaussianMixtureDensity:
    def __init__(self, L, gridN=201):
        self.L = np.asarray(L, dtype=float)
        self.gridN = int(gridN)
        xs = np.linspace(0, self.L[0], self.gridN)
        ys = np.linspace(0, self.L[1], self.gridN)
        self.X, self.Y = np.meshgrid(xs, ys, indexing='xy')
        self.P = np.stack([self.X, self.Y], axis=-1)
        self.dxdy = (self.L[0] / (self.gridN - 1)) * (self.L[1] / (self.gridN - 1))
        self.rho = None

    @staticmethod
    def _gaussian_2d(P, mu, Sigma):
        d = P - mu
        Sinv = np.linalg.inv(Sigma)
        expo = np.einsum('...i,ij,...j->...', d, Sinv, d)
        det = np.linalg.det(Sigma)
        coef = 1.0 / (2.0 * np.pi * np.sqrt(det))
        return coef * np.exp(-0.5 * expo)

    @staticmethod
    def rotated_cov(sig1, sig2, theta_rad):
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        return R @ np.diag([sig1**2, sig2**2]) @ R.T

    def build(self, centers, covs, weights):
        if len(centers) != len(covs) or len(centers) != len(weights):
            raise ValueError("centers, covs, and weights must have equal length.")
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative.")
        w = w / (w.sum() + 1e-16)

        mix = np.zeros(self.P.shape[:2], dtype=float)
        for mu, Sigma, a in zip(centers, covs, w):
            mix += a * self._gaussian_2d(self.P, np.array(mu), np.array(Sigma))

        mix /= (mix.sum() * self.dxdy + 1e-16)  # normalize to integrate to 1
        self.rho = mix
        return self.rho

    def phi_coeffs(self, basis: FourierBasis):
        if self.rho is None:
            raise RuntimeError("Call build(...) first to create rho.")
        phi = np.zeros(basis.nK, dtype=float)
        for i, (kx, ky) in enumerate(basis.modes):
            phi[i] = np.sum(self.rho *
                            np.cos(np.pi * kx * self.X / basis.L[0]) *
                            np.cos(np.pi * ky * self.Y / basis.L[1])) * self.dxdy
        return phi


# -------------------- Multi-robot Ergodic CLF Controller --------------------
class MultiRobotErgodicCLF:
    """
    Decentralized CLF-only controller for N robots, single-integrator dynamics.
    Each robot solves:
        min 0.5 ||u_j||^2 + w_clf * delta_j^2
        s.t. gradE_j^T u_j <= -alpha_j * c_clf * E + delta_j
             |u_j|_inf <= u_max
             delta_j >= 0
    Shared (low-bandwidth): F_avg = (1/N) sum_j F(x_j)
    Running average coefficients: c <- c + dt * (F_avg - c)/t
    """
    def __init__(self, 
                 basis: FourierBasis, phi, u_max=1.0, w_clf=10.0, c_clf=0.5, alphas=None):
        self.basis = basis
        self.phi = np.asarray(phi, dtype=float)
        self.u_max = float(u_max)
        self.w_clf = float(w_clf)
        self.c_clf = float(c_clf)
        self.c = np.zeros_like(self.phi)  # running average coefficients
        self.t_now = 1e-2                 # avoid div-by-zero at start
        self.alphas = None if alphas is None else np.asarray(alphas, dtype=float)

    def ergodic_metric(self):
        return float(np.sum(self.basis.Lam * (self.c - self.phi) ** 2))

    def gradE_at(self, x, N_agents):
        # gradE_j = (2/(t*N)) * sum_k Lambda_k (c_k - phi_k) gradF_k(x_j)
        gradF = self.basis.gradF_mat(x)         # (nK, 2)
        weights = self.basis.Lam * (self.c - self.phi)  # (nK,)
        return (2.0 / (self.t_now * N_agents)) * (weights @ gradF)  # (2,)

    def local_qp(self, gradE_j, E, alpha_j):
        """
        Solve robot j's CLF-QP:
            min 0.5||u||^2 + w_clf*delta^2
            s.t. gradE_j^T u <= -alpha_j * c_clf * E + delta
                 |u_i| <= u_max
                 delta >= 0
        """
        def cost(z):
            u = z[:2]
            d = z[2]
            return 0.5 * np.dot(u, u) + self.w_clf * (d ** 2)

        Mclf = np.zeros(3)
        Mclf[:2] = gradE_j
        Mclf[2] = -1.0
        rhs_clf = -alpha_j * self.c_clf * E

        M = np.vstack([
            Mclf,
            np.array([ 1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([ 0.0, 1.0, 0.0]),
            np.array([ 0.0,-1.0, 0.0]),
        ])
        rhs = np.array([rhs_clf, self.u_max, self.u_max, self.u_max, self.u_max], dtype=float)

        con = LinearConstraint(M, lb=-np.inf * np.ones_like(rhs), ub=rhs)
        bnds = Bounds(lb=[-np.inf, -np.inf, 0.0], ub=[np.inf, np.inf, np.inf])

        z0 = np.zeros(3)
        res = minimize(cost, z0, method='SLSQP', constraints=[con], bounds=bnds)
        if not res.success:
            return np.zeros(2), 0.0
        return res.x[:2], res.x[2]

    def step(self, X):
        """
        One decentralized control step for all robots.
        X: array (N,2) current positions
        Returns:
            U: array (N,2) controls
            E: ergodic metric
            F_avg: vector (nK,) average Fourier evaluations
        """
        N = X.shape[0]
        if self.alphas is None:
            alphas = np.ones(N) / N
        else:
            alphas = self.alphas / (np.sum(self.alphas) + 1e-16)

        # LOCAL evaluations (each robot can compute its own F_k and send to network)
        F_list = np.stack([self.basis.F_vec(x) for x in X])  # (N, nK)
        # LOW-BANDWIDTH SHARE: average over robots (could be via consensus)
        F_avg = F_list.mean(axis=0)

        # Shared running-average update
        self.c = self.c + self.dt * ((F_avg - self.c) / self.t_now)

        # Build local gradients and solve local QPs
        E = self.ergodic_metric()
        U = np.zeros_like(X)
        for j in range(N):
            gE_j = self.gradE_at(X[j], N)
            U[j], _ = self.local_qp(gE_j, E, alphas[j])

        # Advance time
        self.t_now += self.dt
        return U, E, F_avg

    def set_dt(self, dt):
        self.dt = float(dt)


# -------------------- Example usage / full control loop --------------------
if __name__ == "__main__":
    # Domain and basis
    L = np.array([1.0, 1.0])
    Kmax = (4, 4)
    basis = FourierBasis(L=L, Kmax=Kmax)

    # Build a target Gaussian Mixture rho and its Fourier coefficients phi
    gmm = GaussianMixtureDensity(L=L, gridN=201)

    # Example mixture (edit freely)
    centers = [[0.25, 0.70], [0.75, 0.30], [0.55, 0.85]]
    covs = [
        [[0.02, 0.0], [0.0, 0.02]],
        [[0.02, 0.0], [0.0, 0.02]],
        GaussianMixtureDensity.rotated_cov(0.10, 0.03, theta_rad=np.deg2rad(35.0)),
    ]
    weights = [0.4, 0.4, 0.2]

    rho = gmm.build(centers, covs, weights)
    phi = gmm.phi_coeffs(basis)

    # Controller
    N = 2                                  # number of robots
    dt = 0.1
    T_steps = 2000
    u_max = 1.0
    w_clf = 200.0
    c_clf = 2.0
    alphas = np.ones(N) / N                 # equal CLF share

    ctrl = MultiRobotErgodicCLF(basis, phi, u_max=u_max, w_clf=w_clf, c_clf=c_clf, alphas=alphas)
    ctrl.set_dt(dt)

    # Initial states
    rng = np.random.default_rng(2)
    X = rng.uniform(low=[0.1, 0.1], high=[0.9, 0.9], size=(N, 2))

    # Logging
    traj = [X.copy()]
    E_hist = []

    # Live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=140)
    im = ax.imshow(rho.T, origin="lower", extent=[0, L[0], 0, L[1]], interpolation="bilinear")
    paths = [ax.plot([], [], lw=1.8, label=f"r{j+1}")[0] for j in range(N)]
    pts = ax.plot(X[:, 0], X[:, 1], 'o', ms=6)[0]
    ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1]); ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ρ(x)")
    ttl = ax.set_title("N-robot Ergodic CLF-only (Decentralized)   step 0/{}   E=0.0".format(T_steps))
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", ncols=2, fontsize=8)

    # Simulate
    for k in range(T_steps):
        U, E, F_avg = ctrl.step(X)
        X = X + dt * U
        X = np.minimum(np.maximum(X, 0.0), L)  # clamp to box

        traj.append(X.copy())
        E_hist.append(E)

        print(E)

        # Update plot
        arr = np.stack(traj, axis=0)  # (k+2, N, 2)
        for j in range(N):
            paths[j].set_data(arr[:, j, 0], arr[:, j, 1])
        pts.set_data(X[:, 0], X[:, 1])
        ttl.set_text(f"N-robot Ergodic CLF-only (Decentralized)   step {k+1}/{T_steps}   E={E:.3e}")
        plt.pause(0.001)

    plt.ioff()
    plt.show()