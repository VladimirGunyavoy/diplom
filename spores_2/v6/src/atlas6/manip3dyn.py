"""n-звенник, ДИНАМИКА 2n-мерная (ступень 5 manipulator_plan): плоский, горизонтально (без гравитации), точечные массы на концах звеньев.
Состояние (q1..qn, w1..wn), q — относительные углы; управления — моменты в суставах |τ_i| ≤ 1; слои — 2^n углов коробки моментов.
Параметры (решение агента): длины 1, массы звеньев MS = (1.5, 1.0, 0.5) для n=3 (тяжелее у основания)."""
import numpy as np
L = np.ones(3); MS = np.array([1.5, 1.0, 0.5])
LAYERS6 = [tuple(1.0 if (s >> i) & 1 else -1.0 for i in range(3)) for s in range(8)]


def _mats(n, m, ln):
    A = np.tril(np.ones((n, n)))                                   # θ = A q
    S = np.array([[m[max(i, j):].sum() * ln[i] * ln[j] for j in range(n)] for i in range(n)])
    return A, S


def accel(x, tau, m=MS, ln=L, g=0.0):
    """q̈ для x (...,2n), tau (n,). M_θ θ̈ + [S_ij sin(θi−θj) θ̇_j²] = A^{-T} τ; q̈ = A^{-1} θ̈ (θ̈ = A q̈ при θ = A q)."""
    n = len(tau); m = np.asarray(m[:n]); ln = np.asarray(ln[:n]); A, S = _mats(n, m, ln)
    q, w = x[..., :n], x[..., n:]; th = q @ A.T; thd = w @ A.T
    d = th[..., :, None] - th[..., None, :]
    Mt = S * np.cos(d); h = np.einsum('ij,...ij,...j->...i', S, np.sin(d), thd ** 2)
    Qt = np.linalg.solve(A.T, np.asarray(tau, float)) - g * ln * np.cos(th) * np.array([m[i:].sum() for i in range(n)])     # g: гравитация (θ от горизонтали, вертикальная плоскость)
    tdd = np.linalg.solve(Mt, (Qt - h)[..., None])[..., 0]
    return np.linalg.solve(A, tdd[..., None])[..., 0]


def f(x, tau, g=0.0):
    n = len(tau); return np.concatenate([x[..., n:], accel(x, tau, g=g)], -1)


def flow(x, s, t, dt_max=0.02, n=3):
    tau = LAYERS6[s][:n] if n == 3 else tuple(1.0 if (s >> i) & 1 else -1.0 for i in range(n))
    x = np.array(x, float); k = max(1, int(np.ceil(abs(t) / dt_max))); h = t / k
    for _ in range(k):
        k1 = f(x, tau); k2 = f(x + h / 2 * k1, tau); k3 = f(x + h / 2 * k2, tau); k4 = f(x + h * k3, tau)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def energy(x, m=MS, ln=L, g=0.0):
    n = x.shape[-1] // 2; A, S = _mats(n, np.asarray(m[:n]), np.asarray(ln[:n])); th = x[..., :n] @ A.T; thd = x[..., n:] @ A.T
    Mt = S * np.cos(th[..., :, None] - th[..., None, :]); mm = np.asarray(m[:n]); ln = np.asarray(ln[:n])
    pot = g * sum(mm[k] * (ln[:k + 1] * np.sin(th[..., :k + 1])).sum(-1) for k in range(n))
    return 0.5 * np.einsum('...i,...ij,...j->...', thd, Mt, thd) + pot
