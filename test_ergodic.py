import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.stats import multivariate_normal as mvn

class MultiAgentErgodicDecentralized:
    """
    Decentralized N-agent ergodic controller on [0,Lx]x[0,Ly] with:
      • hk-normalized cosine basis
      • Per-agent CLF (always on) and optional High-Order CBF (relative-degree 2 on h=-E)
      • Each agent solves its OWN QP using only global scalars (c, E, dotE, beta) and its local gradient A_i

    Decision per agent i: z_i = [u_ix, u_iy, xi_i]
    Cost per agent: 0.5*||u_i||^2 + w_xi * xi_i^2

    If include_cbf=True:
      CLF   : A_i^T u_i - xi_i <= (-c_clf * E) / N
      HOCBF : A_i^T u_i - xi_i <= (beta + (k1+k2)*dotE + k1*k2*E) / N
    Else:
      CLF   : A_i^T u_i - xi_i <= (-c_clf * E) / N
    """

    def __init__(
        self,
        L=(1.0, 1.0),
        num_k_per_dim=10,
        num_cell=100,
        gmm_params=None,
        dt=0.05,
        T_steps=1200,
        u_max=0.1,
        w_xi=1000.0,
        c_clf=1.2,
        k1=2.0,
        k2=2.0,
        include_cbf=True,
        N_agents=4,
        x0=None,
        seed=7
    ):
        rng = np.random.default_rng(seed)
        self.L = np.array(L, dtype=float)
        self.dt = float(dt)
        self.T_steps = int(T_steps)
        self.u_max = float(u_max)
        self.w_xi = float(w_xi)
        self.c_clf = float(c_clf)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.include_cbf = bool(include_cbf)

        self.ks = self._build_modes(num_k_per_dim)
        self.K = self.ks.shape[0]
        self.gx, self.gy, self.grid, self.dxdy = self._build_grid(num_cell)

        self.hk, self.Fk_grid = self._build_hk_basis()

        if gmm_params is None:
            gmm_params = self._default_gmm()
        self.phi, self.pdf_gt_img = self._project_target_to_phi(gmm_params)
        self.pdf_recon_img = self._reconstruct_from_phi()

        self.Lam = np.array([self._sobolev_weight(k[0], k[1]) for k in self.ks])

        self.N = int(N_agents)
        if x0 is None:
            self.x = rng.uniform(low=0.1, high=0.9, size=(self.N, 2))
        else:
            x0 = np.asarray(x0, dtype=float)
            assert x0.shape == (self.N, 2)
            self.x = x0.copy()

        self.c = np.zeros(self.K, dtype=float)

        self.lam1 = np.zeros(self.N)
        self.lam2 = np.zeros(self.N)

        self.traj = [self.x.copy()]
        self.E_hist = []
        self.dotE_hist = []

        self.t_now = 1e-2

    @staticmethod
    def _default_gmm():
        mean1 = np.array([0.35, 0.38])
        cov1  = np.array([[0.01, 0.004],[0.004, 0.01]])
        w1    = 0.5
        mean2 = np.array([0.68, 0.25])
        cov2  = np.array([[0.005, -0.003],[-0.003, 0.005]])
        w2    = 0.2
        mean3 = np.array([0.56, 0.64])
        cov3  = np.array([[0.008, 0.0],[0.0, 0.004]])
        w3    = 0.3
        return [(w1, mean1, cov1), (w2, mean2, cov2), (w3, mean3, cov3)]

    def _build_modes(self, num_k_per_dim):
        kx, ky = np.meshgrid(np.arange(num_k_per_dim),
                             np.arange(num_k_per_dim),
                             indexing="xy")
        return np.stack([kx.ravel(), ky.ravel()], axis=1)

    def _build_grid(self, num_cell):
        gx, gy = np.meshgrid(np.linspace(0, self.L[0], num_cell),
                             np.linspace(0, self.L[1], num_cell),
                             indexing="xy")
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        dx = self.L[0] / (num_cell - 1)
        dy = self.L[1] / (num_cell - 1)
        dxdy = dx * dy
        return gx, gy, grid, dxdy

    def _build_hk_basis(self):
        K = self.ks.shape[0]
        Fk_grid = np.zeros((K, self.grid.shape[0]), dtype=float)
        hk = np.zeros(K, dtype=float)
        for i, k in enumerate(self.ks):
            fk_raw = np.prod(np.cos(np.pi * k / self.L * self.grid), axis=1)
            hk[i] = np.sqrt(np.sum(fk_raw**2) * self.dxdy)
            Fk_grid[i, :] = fk_raw / hk[i]
        return hk, Fk_grid

    def _gmm_pdf_vals(self, gmm_params):
        vals = 0.0
        for (w, m, C) in gmm_params:
            vals += w * mvn.pdf(self.grid, mean=m, cov=C)
        mass = np.sum(vals) * self.dxdy
        return vals / mass

    def _project_target_to_phi(self, gmm_params):
        rho = self._gmm_pdf_vals(gmm_params)
        phi = (self.Fk_grid @ rho) * self.dxdy
        return phi, rho.reshape(self.gx.shape)

    def _reconstruct_from_phi(self):
        recon = (self.phi @ self.Fk_grid)
        recon = np.maximum(recon, 0.0)
        s = np.sum(recon) * self.dxdy
        if s > 0:
            recon /= s
        return recon.reshape(self.gx.shape)

    @staticmethod
    def _sobolev_weight(kx, ky, n=2):
        return 1.0 / (1.0 + kx*kx + ky*ky)**((n + 1)/2.0)

    def F_stack(self, x):
        cx = np.cos(np.pi * self.ks[:, 0] * x[0] / self.L[0])
        cy = np.cos(np.pi * self.ks[:, 1] * x[1] / self.L[1])
        fk_raw = cx * cy
        return fk_raw / self.hk

    def gradF_stack(self, x):
        kx = self.ks[:, 0]; ky = self.ks[:, 1]
        ax = np.pi * kx / self.L[0]; ay = np.pi * ky / self.L[1]
        cosx, cosy = np.cos(ax * x[0]), np.cos(ay * x[1])
        sinx, siny = np.sin(ax * x[0]), np.sin(ay * x[1])
        dcosx = -ax * sinx
        dcosy = -ay * siny
        gx = (dcosx * cosy) / self.hk
        gy = (cosx * dcosy) / self.hk
        return np.stack([gx, gy], axis=1)

    def _global_quantities(self):
        F_each = np.array([self.F_stack(self.x[i]) for i in range(self.N)])
        F_bar  = np.mean(F_each, axis=0)
        E      = np.sum(self.Lam * (self.c - self.phi)**2)

        dotc   = (F_bar - self.c) / self.t_now
        dotE   = 2.0 * np.sum(self.Lam * (self.c - self.phi) * dotc)

        A_list = []
        for i in range(self.N):
            G_i = self.gradF_stack(self.x[i])
            A_i = (2.0 / self.t_now) * (self.Lam * (self.c - self.phi)) @ G_i
            A_list.append(A_i)
        A_list = np.array(A_list)

        beta = 2.0 * np.sum(
            self.Lam * (
                dotc**2
                + (self.c - self.phi) * (-(1.0/self.t_now)*dotc - (1.0/(self.t_now*self.t_now))*(F_bar - self.c))
            )
        )

        return F_bar, E, dotE, beta, A_list

    def _agent_qp(self, i, E, dotE, beta, A_i):
        rhs_clf_i = (-self.c_clf * E) / self.N
        if self.include_cbf:
            rhs_cbf_i = (beta + (self.k1 + self.k2)*dotE + (self.k1*self.k2)*E) / self.N

        gE_i = A_i.copy()

        def f_cost(z):
            u = z[:2]; xi = z[2]
            return 0.5*np.dot(u, u) + self.w_xi*(xi**2)

        def g1(z):
            u = z[:2]; xi = z[2]
            return float(np.dot(gE_i, u) - xi - rhs_clf_i)

        if self.include_cbf:
            def g2(z):
                u = z[:2]; xi = z[2]
                return float(np.dot(A_i, u) - xi - rhs_cbf_i)
        else:
            def g2(z):
                return -1.0

        def Phi(z, lam1, lam2):
            r1 = g1(z)
            r2 = g2(z)
            val = f_cost(z) + lam1*r1 + 0.5*r1**2
            if self.include_cbf:
                val += lam2*r2 + 0.5*max(0.0, r2)**2
            return val

        bnds = Bounds(lb=[-self.u_max, -self.u_max, 0.0],
                      ub=[ self.u_max,  self.u_max, np.inf])

        res = minimize(lambda z: Phi(z, self.lam1[i], self.lam2[i]),
                       x0=np.zeros(3), method="SLSQP", bounds=bnds)
        z = np.zeros(3) if not res.success else res.x

        u = z[:2]; xi = z[2]
        print(res.x)

        r1 = max(0.0, g1(z))
        self.lam1[i] = max(0.0, self.lam1[i] + 1.0*r1)
        if self.include_cbf:
            r2 = max(0.0, g2(z))
            self.lam2[i] = max(0.0, self.lam2[i] + 1.0*r2)

        return u, xi

    def step(self):
        F_bar, E, dotE, beta, A_list = self._global_quantities()

        U = np.zeros_like(self.x)
        for i in range(self.N):
            U[i], _ = self._agent_qp(i, E, dotE, beta, A_list[i])
            n = np.linalg.norm(U[i])
            if n > 0.05:
                U[i] = (U[i]/n) * 0.05

        self.x = self.x + self.dt * U
        self.x = np.minimum(np.maximum(self.x, 0.0), self.L)

        F_each_new = np.array([self.F_stack(self.x[i]) for i in range(self.N)])
        F_bar_new  = np.mean(F_each_new, axis=0)
        self.c     = self.c + self.dt * ((F_bar_new - self.c) / self.t_now)

        self.t_now += self.dt
        self.traj.append(self.x.copy())
        self.E_hist.append(E)
        self.dotE_hist.append(dotE)

    def run(self, show_live=True, plot_every=5):
        if show_live:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
            ax.imshow(self.pdf_recon_img, origin="lower",
                      extent=[0, self.L[0], 0, self.L[1]], interpolation="bilinear")
            title = "Decentralized N-Agent Ergodic Control (CLF{}{})".format(
                " + " if self.include_cbf else "",
                "HOCBF" if self.include_cbf else ""
            )
            ax.set_title(title)
            cb = plt.colorbar(ax.images[0], ax=ax)
            cb.set_label("Reconstructed ρ(x)")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            lines = [ax.plot([], [], '-', lw=2)[0] for _ in range(self.N)]
            pts   = ax.scatter([], [], s=36)
            ax.set_xlim(0, self.L[0]); ax.set_ylim(0, self.L[1])
            ax.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            plt.pause(0.1)

        for k in range(self.T_steps):
            self.step()
            if show_live and (k % plot_every == 0):
                arr = np.array(self.traj)  # (steps+1, N, 2)
                for i in range(self.N):
                    path = arr[:, i, :]
                    lines[i].set_data(path[:, 0], path[:, 1])
                pts.set_offsets(self.x)
                plt.pause(0.001)

        if show_live:
            plt.show()

    def plot_pdf_side_by_side(self, means=None):
        vmin = 0.0
        vmax = max(self.pdf_gt_img.max(), self.pdf_recon_img.max())
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=130, constrained_layout=True)

        ax = axes[0]
        im0 = ax.imshow(self.pdf_gt_img, origin="lower",
                        extent=[0, self.L[0], 0, self.L[1]],
                        interpolation="bilinear", vmin=vmin, vmax=vmax, cmap="Reds")
        ax.set_title("Ground Truth PDF")
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04, label="density")
        if means is not None:
            ax.scatter([m[0] for m in means], [m[1] for m in means], c="k", s=30, marker="x")

        ax = axes[1]
        im1 = ax.imshow(self.pdf_recon_img, origin="lower",
                        extent=[0, self.L[0], 0, self.L[1]],
                        interpolation="bilinear", vmin=vmin, vmax=vmax, cmap="Blues")
        ax.set_title("Reconstructed PDF from φ_k (hk-normalized)")
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="density")
        if means is not None:
            ax.scatter([m[0] for m in means], [m[1] for m in means], c="k", s=30, marker="x")

        plt.show()

    def plot_metrics(self):
        tt = np.arange(len(self.E_hist)) * self.dt
        fig, ax = plt.subplots(2, 1, figsize=(6, 5), dpi=140, constrained_layout=True)
        ax[0].plot(tt, self.E_hist, lw=2); ax[0].set_ylabel("E"); ax[0].set_title("Ergodic Metric")
        ax[1].plot(tt, self.dotE_hist, lw=2); ax[1].set_ylabel("dE/dt"); ax[1].set_xlabel("time"); ax[1].set_title("dE/dt")
        plt.show()

if __name__ == "__main__":
    gmm = [
        (0.5, np.array([0.35, 0.38]), np.array([[0.05, 0.004],[0.004, 0.01]])),
        # (0.2, np.array([0.68, 0.25]), np.array([[0.005, -0.003],[-0.003, 0.005]])),
        (0.5, np.array([0.56, 0.64]), np.array([[0.008, 0.0],[0.0, 0.004]])),
    ]

    ctrl = MultiAgentErgodicDecentralized(
        L=(1.0, 1.0),
        num_k_per_dim=10,
        num_cell=100,
        gmm_params=gmm,
        dt=0.1,
        T_steps=1200,
        u_max=0.8,
        w_xi=1000.0,
        c_clf=1.0,
        k1=2.0, k2=2.0,
        include_cbf=False,
        N_agents=1,
        x0=None,
        seed=7
    )

    ctrl.run(show_live=True, plot_every=5)
    # ctrl.plot_metrics()
