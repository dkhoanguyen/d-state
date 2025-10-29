import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from scipy.optimize import minimize, Bounds
from scipy.stats import multivariate_normal as mvn

# ============================================================
# Core, decentralized, function-first implementation
#   – Minimal shared state
#   – Each function takes what it needs and returns values
#   – Ready to integrate into an execute(...) framework
# ============================================================

# -----------------------------
# Data containers (lightweight)
# -----------------------------
@dataclass
class ErgodicParams:
    L: np.ndarray                    # domain, shape (2,)
    num_k_per_dim: int               # number of cosine modes per axis
    num_cell: int                    # grid resolution per axis
    c_clf: float = 1.0               # CLF gain
    include_cbf: bool = False        # keep False unless you also pass k1,k2
    k1: float = 2.0                  # HOCBF params (used only if include_cbf=True)
    k2: float = 2.0
    w_xi: float = 1000.0             # slack cost
    u_max: float = 0.1               # max control magnitude per axis
    dt: float = 0.1                  # agent time step
    drift_std: float = 0.0           # random drift for moving targets (0 = static)

@dataclass
class BasisPack:
    phi: np.ndarray      # (K,)
    Lam: np.ndarray      # (K,)
    ks: np.ndarray       # (K,2)
    Fk_grid: np.ndarray  # (K, M)
    dxdy: float
    hk: np.ndarray       # (K,)
    pdf_img: Optional[np.ndarray] = None
    gx_shape: Optional[Tuple[int,int]] = None

@dataclass
class AgentView:
    id: int
    state: np.ndarray          # shape (2,)
    c: Optional[np.ndarray]    # shape (K,), can be None on first call
    c_dot: Optional[np.ndarray]# shape (K,), can be None
    goal_lambda: float         # scalar dual for CLF


# ---------------------------------------
# Target / basis preprocessing utilities
# ---------------------------------------
def build_modes(num_k_per_dim: int) -> np.ndarray:
    kx, ky = np.meshgrid(np.arange(num_k_per_dim),
                         np.arange(num_k_per_dim), indexing="xy")
    return np.stack([kx.ravel(), ky.ravel()], axis=1)

def build_grid(num_cell: int, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    gx, gy = np.meshgrid(np.linspace(0, L[0], num_cell),
                         np.linspace(0, L[1], num_cell), indexing="xy")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    dx = L[0] / (num_cell - 1)
    dy = L[1] / (num_cell - 1)
    return gx, gy, grid, dx * dy

def build_hk_basis(ks: np.ndarray, grid: np.ndarray, dxdy: float, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = ks.shape[0]
    Fk_grid = np.zeros((K, grid.shape[0]), dtype=float)
    hk = np.zeros(K, dtype=float)
    for i, k in enumerate(ks):
        fk_raw = np.prod(np.cos(np.pi * k / L * grid), axis=1)
        hk[i] = np.sqrt(np.sum(fk_raw**2) * dxdy)
        Fk_grid[i, :] = fk_raw / hk[i]
    return hk, Fk_grid

def sobolev_weight(kx: int, ky: int, n: int = 2) -> float:
    return 1.0 / (1.0 + kx*kx + ky*ky)**((n + 1)/2.0)

def gmm_pdf_vals(grid: np.ndarray,
                 dxdy: float,
                 gmm_params: List[Tuple[float, np.ndarray, np.ndarray]]) -> np.ndarray:
    vals = 0.0
    for (w, m, C) in gmm_params:
        vals += w * mvn.pdf(grid, mean=m, cov=C)
    mass = np.sum(vals) * dxdy
    return vals / mass

def project_target_to_phi(Fk_grid: np.ndarray,
                          grid: np.ndarray,
                          gmm_params: List[Tuple[float, np.ndarray, np.ndarray]],
                          dxdy: float) -> Tuple[np.ndarray, np.ndarray]:
    rho = gmm_pdf_vals(grid, dxdy, gmm_params)
    phi = (Fk_grid @ rho) * dxdy
    return phi, rho

def reconstruct_from_phi(phi: np.ndarray, Fk_grid: np.ndarray, dxdy: float, gx_shape: Tuple[int,int]) -> np.ndarray:
    recon = (phi @ Fk_grid)
    recon = np.maximum(recon, 0.0)
    s = np.sum(recon) * dxdy
    if s > 0:
        recon /= s
    return recon.reshape(gx_shape)

def preprocess_target(centers: np.ndarray,
                      covs: np.ndarray,
                      weights: np.ndarray,
                      params: ErgodicParams,
                      rng: Optional[np.random.Generator] = None) -> BasisPack:
    # Optionally drift means a bit to simulate time-varying target
    gmm_params = []
    for w, m, C in zip(weights, centers, covs):
        m = np.array(m, dtype=float)
        if rng is not None and params.drift_std > 0:
            m = np.clip(m + rng.normal(0.0, params.drift_std, size=2),
                        a_min=[0.0, 0.0], a_max=params.L)
        gmm_params.append((float(w), m, np.array(C, dtype=float)))

    ks = build_modes(params.num_k_per_dim)
    gx, gy, grid, dxdy = build_grid(params.num_cell, params.L)
    hk, Fk_grid = build_hk_basis(ks, grid, dxdy, params.L)
    phi, rho_flat = project_target_to_phi(Fk_grid, grid, gmm_params, dxdy)
    pdf_img = rho_flat.reshape(gx.shape)
    Lam = np.array([sobolev_weight(int(k[0]), int(k[1])) for k in ks], dtype=float)
    return BasisPack(phi=phi, Lam=Lam, ks=ks, Fk_grid=Fk_grid, dxdy=dxdy, hk=hk, pdf_img=pdf_img, gx_shape=gx.shape)


# --------------------------------
# Local basis evaluation utilities
# --------------------------------
def F_stack(x: np.ndarray, ks: np.ndarray, L: np.ndarray, hk: np.ndarray) -> np.ndarray:
    cx = np.cos(np.pi * ks[:, 0] * x[0] / L[0])
    cy = np.cos(np.pi * ks[:, 1] * x[1] / L[1])
    return (cx * cy) / hk

def gradF_stack(x: np.ndarray, ks: np.ndarray, L: np.ndarray, hk: np.ndarray) -> np.ndarray:
    kx = ks[:, 0]; ky = ks[:, 1]
    ax = np.pi * kx / L[0]; ay = np.pi * ky / L[1]
    cosx, cosy = np.cos(ax * x[0]), np.cos(ay * x[1])
    sinx, siny = np.sin(ax * x[0]), np.sin(ay * x[1])
    dcosx = -ax * sinx
    dcosy = -ay * siny
    gx = (dcosx * cosy) / hk
    gy = (cosx * dcosy) / hk
    return np.stack([gx, gy], axis=1)


# -----------------------------------------
# Decentralized metric + QP per-agent block
# -----------------------------------------
def decentralized_metrics(agent: AgentView,
                          others: List[AgentView],
                          basis: BasisPack,
                          t_now: float,
                          L: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Returns:
      E, dotE, beta, A_i, dotc_i
    """
    K = basis.ks.shape[0]
    c_local = agent.c if agent.c is not None else np.zeros(K, dtype=float)

    # consensus on c (logical average of exchanged vectors)
    c_stack = [c_local]
    for ag in others:
        if ag.id == agent.id:
            continue
        c_stack.append(ag.c if ag.c is not None else np.zeros(K, dtype=float))
    mean_c = np.mean(np.vstack(c_stack), axis=0)

    resid = (mean_c - basis.phi)
    E = np.sum(basis.Lam * resid**2)

    F_i = F_stack(agent.state, basis.ks, L=L, hk=basis.hk)
    G_i = gradF_stack(agent.state, basis.ks, L=L, hk=basis.hk)
    A_i = (2.0 / max(t_now, 1e-6)) * (basis.Lam * resid) @ G_i

    # local dot c (only local info)
    dotc_i = (F_i - c_local) / max(t_now, 1e-6)

    # gather others' dotc if they publish; fallback to their own estimate
    dotc_list = [dotc_i]
    for ag in others:
        if ag.id == agent.id:
            continue
        if ag.c_dot is not None:
            dotc_list.append(ag.c_dot)
        else:
            # conservative fallback from their current state & c
            c_other = ag.c if ag.c is not None else np.zeros(K, dtype=float)
            F_other = F_stack(ag.state, basis.ks, L=L, hk=basis.hk)
            dotc_list.append((F_other - c_other) / max(t_now, 1e-6))

    dotc = np.mean(np.vstack(dotc_list), axis=0)
    dotE = 2.0 * np.sum(basis.Lam * resid * dotc)
    beta = 2.0 * np.sum(
        basis.Lam * (
            dotc**2 + resid * (-(1.0/max(t_now,1e-6))*dotc - (1.0/(max(t_now,1e-6)**2))*(max(t_now,1e-6)*dotc))
        )
    )
    return E, dotE, beta, A_i, dotc_i

def solve_local_qp(A_i: np.ndarray,
                   E: float,
                   dotE: float,
                   beta: float,
                   goal_lambda: float,
                   N_total: int,
                   params: ErgodicParams) -> Tuple[np.ndarray, float]:
    """
    Returns:
      u (2,), goal_lambda_new
    """
    rhs_clf = (-params.c_clf * E) / max(N_total, 1)
    gE_i = A_i.copy()

    def f_cost(z):
        u = z[:2]; xi = z[2]
        return 0.5*np.dot(u, u) + params.w_xi*(xi**2)

    def g1(z):
        u = z[:2]; xi = z[2]
        return float(np.dot(gE_i, u) - xi - rhs_clf)

    def Phi(z, lam1):
        r1 = g1(z)
        return f_cost(z) + lam1*r1 + 0.5*r1**2

    bnds = Bounds(lb=[-100, -100, 0.0],
                  ub=[ 100,  100, np.inf])

    res = minimize(lambda z: Phi(z, goal_lambda),
                   x0=np.zeros(3), method="SLSQP", bounds=bnds)
    z = np.zeros(3) if not res.success else res.x
    u = z[:2]
    n = np.linalg.norm(u)
    if n > 0.05:
        u = (u/n) * 0.05

    # dual update
    r1 = max(0.0, g1(z))
    goal_lambda_new = max(0.0, goal_lambda + 1.0*r1)
    return u, goal_lambda_new

def update_local_c_and_state(agent: AgentView,
                             basis: BasisPack,
                             params: ErgodicParams,
                             t_now: float,
                             u: np.ndarray,
                             motion_model: Callable[[np.ndarray, np.ndarray, float], np.ndarray]) -> AgentView:
    """
    Integrate state and update local coefficient vector using ONLY local quantities.
    Returns updated AgentView (copy).
    """
    K = basis.ks.shape[0]
    new_agent = AgentView(
        id=agent.id,
        state=motion_model(agent.state, u, params.dt),
        c=(agent.c if agent.c is not None else np.zeros(K, dtype=float)).copy(),
        c_dot=(agent.c_dot if agent.c_dot is not None else np.zeros(K, dtype=float)).copy(),
        goal_lambda=agent.goal_lambda
    )
    # clamp to domain
    new_agent.state = np.minimum(np.maximum(new_agent.state, 0.0), params.L)

    # local F_i and coefficient dynamics
    F_i_new = F_stack(new_agent.state, basis.ks, params.L, basis.hk)
    new_agent.c = new_agent.c + params.dt * ((F_i_new - new_agent.c) / max(t_now, 1e-6))
    new_agent.c_dot = (F_i_new - new_agent.c) / max(t_now, 1e-6)

    return new_agent


# ============================================================
# Example “execute-style” glue (standalone demo)
# ============================================================
@dataclass
class Scenario:
    L: np.ndarray
    dt: float
    seed: int = 7

@dataclass
class ErgodicCoverageTask:
    centers: np.ndarray     # shape (G, 2)
    covs: np.ndarray        # shape (G, 2, 2)
    weights: np.ndarray     # shape (G,)


# Example motion model for a single-integrator agent
def single_integrator(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    return x + dt * u


def main():
    # -------------------
    # World / experiment
    # -------------------
    scenario = Scenario(L=np.array([1.0, 1.0], dtype=float), dt=0.1, seed=7)

    # Two-mode GMM target
    centers = np.array([[0.35, 0.38],
                        [0.56, 0.64]], dtype=float)
    covs = np.array([
        [[0.05, 0.004],
         [0.004, 0.01]],
        [[0.008, 0.0],
         [0.0,   0.004]]
    ], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)
    task = ErgodicCoverageTask(centers=centers, covs=covs, weights=weights)

    # Ergodic controller parameters
    params = ErgodicParams(
        L=scenario.L,
        num_k_per_dim=10,
        num_cell=100,
        c_clf=1.0,
        include_cbf=False,
        w_xi=1000.0,
        u_max=0.8,
        dt=scenario.dt,
        drift_std=0.0   # set >0 to see a moving target over time
    )

    # -------------------
    # Agents
    # -------------------
    rng = np.random.default_rng(scenario.seed)
    agents: List[AgentView] = [
        AgentView(id=0, state=np.array([0.10, 0.10], dtype=float),
                  c=None, c_dot=None, goal_lambda=0.0),
        AgentView(id=1, state=np.array([0.90, 0.90], dtype=float),
                  c=None, c_dot=None, goal_lambda=0.0),
    ]
    N = len(agents)

    # -------------
    # Viz set-up
    # -------------
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    rng_local = np.random.default_rng(scenario.seed)

    # Precompute a basis once to get image shape for imshow initialization.
    basis0 = preprocess_target(task.centers, task.covs, task.weights, params, rng=rng_local)
    im = ax.imshow(basis0.pdf_img, origin="lower",
                   extent=[0, params.L[0], 0, params.L[1]],
                   interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("ρ(x)")
    ax.set_title("Decentralized Ergodic Control — Live")
    ax.set_xlim(0, params.L[0]); ax.set_ylim(0, params.L[1])
    ax.set_aspect('equal', adjustable='box')
    lines = [ax.plot([], [], '-', lw=2)[0] for _ in range(N)]
    pts   = ax.scatter([a.state[0] for a in agents],
                       [a.state[1] for a in agents], s=36)
    traj = [[a.state.copy()] for a in agents]
    plt.pause(0.1)

    # -------------
    # Run loop
    # -------------
    T_steps = 1000
    t_now = 1e-2

    for k in range(T_steps):
        # 1) Update target and basis (recomputed each step; supports moving target via drift_std)
        basis = preprocess_target(task.centers, task.covs, task.weights, params, rng=None)

        # 2) For each agent, compute decentralized metrics, solve local QP, and update
        new_agents: List[AgentView] = []
        for a in agents:
            # "others" here includes *all* agents; metrics ignores self by id
            E, dotE, beta, A_i, dotc_i = decentralized_metrics(
                agent=a, others=agents, basis=basis, t_now=t_now, L=params.L
            )
            u, new_lambda = solve_local_qp(
                A_i=A_i, E=E, dotE=dotE, beta=beta,
                goal_lambda=a.goal_lambda,
                N_total=N, params=params
            )
            a.goal_lambda = new_lambda

            # Update state and local coefficient vector using ONLY local data
            a_updated = update_local_c_and_state(
                agent=a, basis=basis, params=params, t_now=t_now,
                u=u, motion_model=single_integrator
            )
            new_agents.append(a_updated)

        agents = new_agents
        t_now += params.dt

        # 3) Viz update
        if k % 2 == 0:
            im.set_data(basis.pdf_img)
            for i, a in enumerate(agents):
                traj[i].append(a.state.copy())
                path = np.array(traj[i])
                lines[i].set_data(path[:, 0], path[:, 1])
            pts.set_offsets(np.array([a.state for a in agents]))
            plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
