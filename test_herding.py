import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

def f_repulsion(x_a, x_r, c=1.0, eps=1e-2):
    r = x_a - x_r
    norm_r = np.linalg.norm(r) + eps
    return c * r / (norm_r**3)

def J_repulsion(x_a, x_r, c=1.0, eps=1e-2):
    r = x_a - x_r
    norm_r = np.linalg.norm(r) + eps
    I = np.eye(2)
    outer = np.outer(r, r)
    J = (I / (norm_r**3)) - (3.0 * outer) / (norm_r**5)
    return c * J, -c * J

def build_goal_hocbf_terms(x_a, x_r, x_g, d, alpha1=2.0, alpha2=3.0, c=1.0, eps=1e-2):
    g = x_a - x_g
    h0 = d**2 - np.dot(g, g)
    fa = f_repulsion(x_a, x_r, c=c, eps=eps)
    dot_h0 = -2.0 * np.dot(g, fa)
    dfa_dxa, dfa_dxr = J_repulsion(x_a, x_r, c=c, eps=eps)
    ddot_h0 = -2.0 * np.dot(fa, fa) - 2.0 * np.dot(g, dfa_dxa @ fa)
    A_g = -2.0 * (g @ dfa_dxr)
    psi2_const = ddot_h0 + (alpha1 + alpha2) * dot_h0 + alpha1 * alpha2 * h0
    b_g = -psi2_const
    return A_g.reshape(1, 2), float(b_g), h0

def build_separation_cbf(x_a, x_r, r_min=0.22, gamma=1.0, c=1.0, eps=1e-2):
    r = x_a - x_r
    norm_r = np.linalg.norm(r) + eps
    h_s = norm_r - r_min
    rhat = r / norm_r
    fa = f_repulsion(x_a, x_r, c=c, eps=eps)
    A_s = -rhat.reshape(1, 2)
    b_s = -np.dot(rhat, fa) - gamma * h_s
    return A_s, float(b_s), h_s

def solve_qp_cbf(x_a, x_r, x_g, params):
    c = params["c"]
    eps = params["eps"]
    alpha1 = params["alpha1"]
    alpha2 = params["alpha2"]
    d = params["goal_radius"]
    r_min = params["r_min"]
    gamma = params["gamma"]
    w_g = params["w_g"]
    w_s = params["w_s"]
    u_max = params.get("u_max", None)
    A_g, b_g, _ = build_goal_hocbf_terms(x_a, x_r, x_g, d, alpha1, alpha2, c=c, eps=eps)
    A_s, b_s, _ = build_separation_cbf(x_a, x_r, r_min=r_min, gamma=gamma, c=c, eps=eps)
    def obj(z):
        u = z[:2]
        dg = z[2]
        ds = z[3]
        return 0.5 * np.dot(u, u) + w_g * (dg**2) + w_s * (ds**2)
    def con_goal(z):
        u = z[:2]
        dg = z[2]
        return float((A_g @ u)[0] - (b_g - dg))
    def con_safe(z):
        u = z[:2]
        ds = z[3]
        return float((A_s @ u)[0] - (b_s - ds))
    cons = [
        {"type": "ineq", "fun": con_goal},
        {"type": "ineq", "fun": con_safe},
        {"type": "ineq", "fun": lambda z: z[2]},
        {"type": "ineq", "fun": lambda z: z[3]},
    ]
    if u_max is not None:
        bounds = Bounds([-u_max, -u_max, 0.0, 0.0], [u_max, u_max, np.inf, np.inf])
    else:
        bounds = Bounds([-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf])
    z0 = np.array([0.0, 0.0, 0.1, 0.1])
    res = minimize(
        obj, z0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 200, "disp": False}
    )
    if not res.success:
        u_fallback = -0.5 * (x_r - x_a)
        u_fallback = u_fallback / (np.linalg.norm(u_fallback) + 1e-9) * 0.1
        return u_fallback, 0.0, 0.0, False
    u = res.x[:2]
    dg = float(res.x[2])
    ds = float(res.x[3])
    return u, dg, ds, True

def simulate_and_plot(
    x_a0=np.array([1.25, 0.50]),
    x_r0=np.array([0.10, 0.10]),
    x_g=np.array([1.00, 1.00]),
    T=20.0,
    dt=0.01,
    params=None,
    seed=1
):
    if params is None:
        params = {}
    p = {
        "c": params.get("c", 1.0),
        "eps": params.get("eps", 1e-2),
        "alpha1": params.get("alpha1", 2.0),
        "alpha2": params.get("alpha2", 3.0),
        "goal_radius": params.get("goal_radius", 0.15),
        "r_min": params.get("r_min", 0.22),
        "gamma": params.get("gamma", 1.0),
        "w_g": params.get("w_g", 1000.0),
        "w_s": params.get("w_s", 10.0),
        "u_max": params.get("u_max", 1.0),
        "robot_drift_std": params.get("robot_drift_std", 0.0),
    }
    rng = np.random.default_rng(seed)
    steps = int(np.ceil(T / dt))
    x_a_hist = np.zeros((steps + 1, 2))
    x_r_hist = np.zeros((steps + 1, 2))
    h0_hist = np.zeros(steps + 1)
    hs_hist = np.zeros(steps + 1)
    x_a = x_a0.copy()
    x_r = x_r0.copy()
    x_a_hist[0] = x_a
    x_r_hist[0] = x_r
    _, _, h0_hist[0] = build_goal_hocbf_terms(x_a, x_r, x_g, p["goal_radius"], p["alpha1"], p["alpha2"], p["c"], p["eps"])
    _, _, hs_hist[0] = build_separation_cbf(x_a, x_r, p["r_min"], p["gamma"], p["c"], p["eps"])
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 1.6)
    ax.set_ylim(-0.2, 1.6)
    ax.grid(True, alpha=0.3)
    theta = np.linspace(0, 2*np.pi, 200)
    goal_circle, = ax.plot(
        x_g[0] + p["goal_radius"] * np.cos(theta),
        x_g[1] + p["goal_radius"] * np.sin(theta),
        linestyle="--", linewidth=1.5
    )
    animal_traj, = ax.plot([], [], linewidth=1.5, label="Animal")
    robot_traj, = ax.plot([], [], linewidth=1.5, label="Robot")
    animal_scatter = ax.scatter([x_a[0]], [x_a[1]], s=60, marker="o", label="Animal (x_a)")
    robot_scatter = ax.scatter([x_r[0]], [x_r[1]], s=60, marker="s", label="Robot (x_r)")
    ax.scatter([x_g[0]], [x_g[1]], marker="*", s=140, label="Goal")
    ax.legend(loc="upper right")
    for k in range(steps):
        u, dg, ds, ok = solve_qp_cbf(x_a, x_r, x_g, p)
        drift = rng.normal(0.0, p["robot_drift_std"], size=2)
        x_r = x_r + (u + drift) * dt
        fa = f_repulsion(x_a, x_r, c=p["c"], eps=p["eps"])
        x_a = x_a + fa * dt
        x_a_hist[k + 1] = x_a
        x_r_hist[k + 1] = x_r
        _, _, h0_hist[k + 1] = build_goal_hocbf_terms(x_a, x_r, x_g, p["goal_radius"], p["alpha1"], p["alpha2"], p["c"], p["eps"])
        _, _, hs_hist[k + 1] = build_separation_cbf(x_a, x_r, p["r_min"], p["gamma"], p["c"], p["eps"])
        animal_traj.set_data(x_a_hist[:k + 2, 0], x_a_hist[:k + 2, 1])
        robot_traj.set_data(x_r_hist[:k + 2, 0], x_r_hist[:k + 2, 1])
        animal_scatter.set_offsets(np.array([[x_a[0], x_a[1]]]))
        robot_scatter.set_offsets(np.array([[x_r[0], x_r[1]]]))
        ax.set_title(
            f"Step {k+1}/{steps} | ||x_a - x_g||={np.linalg.norm(x_a - x_g):.3f} | "
            f"h0={h0_hist[k + 1]:.3e} | hs={hs_hist[k + 1]:.3e} | "
            f"dg={dg:.3e} ds={ds:.3e} | ok={ok}"
        )
        plt.pause(0.001)
        if np.linalg.norm(x_a - x_g) <= p["goal_radius"]:
            break
    plt.show()
    return x_a_hist[:k + 2], x_r_hist[:k + 2], h0_hist[:k + 2], hs_hist[:k + 2]

if __name__ == "__main__":
    x_a0 = np.array([1.25, 0.50])
    x_r0 = np.array([0.10, 0.10])
    x_g = np.array([1.00, 1.00])
    params = {
        "c": 1.0,
        "goal_radius": 0.15,
        "r_min": 0.22,
        "alpha1": 2.0,
        "alpha2": 3.0,
        "gamma": 1.0,
        "w_g": 1000.0,
        "w_s": 10.0,
        "u_max": 1.0,
        "robot_drift_std": 0.0
    }
    x_a_hist, x_r_hist, h0_hist, hs_hist = simulate_and_plot(
        x_a0=x_a0,
        x_r0=x_r0,
        x_g=x_g,
        T=20.0,
        dt=0.01,
        params=params,
        seed=1
    )