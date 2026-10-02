"""research hub-v5chain-research-8: «бабочка» (идея пользователя 2026-10-02). (1) 2D ДИ: из точки вперёд+назад всеми u — бабочка с точкой в центре;
из нормального отрезка — с плотной серединой. (2) 4D: плоский ДИ (x, y, vx, vy), u = (ax, ay): якобиан D = [∂Φ/∂t, ∂Φ/∂ax, ∂Φ/∂ay], проектор
P = I − D(DᵀD)⁻¹Dᵀ, нормаль n; покрытие малого шара вокруг споры: из точки и из нормального отрезка (обратная задача — аналитически)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
# ---------- 4D ----------
c = np.array([0., 0., .6, .2]); tau = .5
def Phi(p, t, a): return np.r_[p[:2] + p[2:] * t + a * t * t / 2, p[2:] + a * t]
def D_of(p, t, a):
    dt = np.r_[p[2:] + a * t, a]; dax = np.r_[t * t / 2, 0, t, 0]; day = np.r_[0, t * t / 2, 0, t]; return np.stack([dt, dax, day], 1)
t0, a0 = .25, np.array([.3, -.5]); D = D_of(c, t0, a0); P = np.eye(4) - D @ np.linalg.solve(D.T @ D, D.T)
n_svd = np.linalg.svd(D.T)[2][-1]; wb = c[2:] + a0 * t0 / 2; n_an = np.r_[-wb[1], wb[0], wb[1] * t0 / 2, -wb[0] * t0 / 2]; n_an /= np.linalg.norm(n_an)
print('спора c =', c, ' t =', t0, ' a =', a0); print('D (4×3) =\n', D); print('ранг D =', np.linalg.matrix_rank(D)); print('P = I − D(DᵀD)⁻¹Dᵀ =\n', P)
print('ранг P =', np.linalg.matrix_rank(P), ' n (SVD) =', n_svd * np.sign(n_svd[1]), ' n (формула) =', n_an * np.sign(n_an[1]), ' Dᵀn =', D.T @ n_an)
n0 = np.r_[-c[3], c[2], 0, 0] / np.hypot(c[2], c[3]); print('предел t→0: n0 = (−vy, vx, 0, 0)/|v| =', n0)
def inverse(y, use_seg):
    w = c[2:] + y[2:]; d = y[:2] - c[:2]; cr = lambda a, b: a[0] * b[1] - a[1] * b[0]
    s = cr(d, w) / cr(n0[:2], w) if use_seg else 0.
    if not use_seg and abs(cr(d, w)) > 1e-12: return None
    dd = d - s * n0[:2]; t = 2 * dd @ w / (w @ w); a = (y[2:] - c[2:]) / t if abs(t) > 1e-12 else np.full(2, np.inf)
    ok = abs(t) <= tau and np.all(np.abs(a) <= 1) and abs(s) <= .15 and np.allclose(Phi(c + s * n0, t, a), y, atol=1e-9); return (s, t, a) if ok else None
rng = np.random.default_rng(0); Y = c + rng.normal(size=(20000, 4)) * .08
for seg in (False, True):
    print('покрытие облака вокруг c (σ = .08):', 'нормальный отрезок |s|≤.15' if seg else 'из точки', '→', round(np.mean([inverse(y, seg) is not None for y in Y]), 3))
# ---------- 2D картинка ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5)); c2 = np.array([0., .5])
for k, (A, ttl) in enumerate(((0., 'из точки: вперёд и назад всеми u ∈ [−1,1] — бабочка с точкой в центре'),
                               (.12, 'из нормального отрезка (нормаль к f(c,0) = (v,0)): середина плотная'))):
    rng2 = np.random.default_rng(1); N = 40000; s = rng2.uniform(-A, A, N); t = rng2.uniform(-tau, tau, N); u = rng2.uniform(-1, 1, N)
    x0, v0 = c2[0] + 0 * s, c2[1] + s                                                   # нормаль к (v,0) — вертикаль
    X = x0 + v0 * t + u * t * t / 2; V = v0 + u * t
    ax[k].scatter(X, V, s=.3, c=np.sign(t), cmap='coolwarm', alpha=.4); ax[k].plot(*c2, 'k*', ms=12)
    if A: ax[k].plot([0, 0], [c2[1] - A, c2[1] + A], 'k', lw=3)
    ax[k].set_title(ttl, fontsize=10); ax[k].set_xlabel('x'); ax[k].set_ylabel('v'); ax[k].set_aspect('equal'); ax[k].grid(alpha=.3)
plt.tight_layout(); plt.savefig('figs/explain_butterfly.png', dpi=110)
