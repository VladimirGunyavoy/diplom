"""research hub-v5chain-research-9: споры-«бабочки» на ДВОЙНОМ МАЯТНИКЕ / двухзвенном манипуляторе (динамика 4D: q1, q2, w1, w2; моменты |τ_i| ≤ 1).
Модель = v6 `src/atlas6/manip3dyn.py` (n = 2, массы 1.5/1.0, длины 1, g — гравитация, θ от горизонтали). Задача-эталон research-7 (`dp_ocp_ref.md`):
g = 1, старт «висит» (−π/2, 0, 0, 0), цель q ∈ (π/2, 0) ± .3, |w| ≤ .5, |w| ≤ 3 вдоль пути; OCP 5.098, v6 5.28.
Спора: точка c + отрезок s·n, n = (−w2, w1, 0, 0)/|w| (⟂ скорости в пространстве q; при |w| < .05 — случайное направление в q), m узлов.
Обратная задача (y → c + s·n одной дугой с постоянным τ): 4 уравнения, 4 неизвестных (t, τ1, τ2, s) — Ньютон по точному потоку (rk4),
∂/∂t = f даром, ∂/∂τ — разностями, ∂/∂s = −n; старт — формула ДИ (t0 = 2Δq·w̄/|w̄|², a = Δw/t0, τ0 — обратная динамика в середине).
Беллман: V(узел) = min по годным дугам [t + V_c(s)] (V_c линейно по узлам), Якоби. Агент — держит τ шаг dt и перепланирует."""
import numpy as np, sys, os, time, json
G = float(os.environ.get('G', 1.)); M1, M2 = 1.5, 1.0; S11, S12, S22 = M1 + M2, M2, M2; MS = np.array([M1 + M2, M2])
C3 = np.array([np.pi / 2, 0.]); RQ, RW_, WMAX = .3, .5, 3.; BIG = 1e3
TL = float(os.environ.get('TL', 1.5)); WIN = float(os.environ.get('WIN', .5))
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def acc(q1, q2, w1, w2, u1, u2):
    th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = np.cos(th1 - th2); s = np.sin(th1 - th2)
    Q1 = u1 - u2 - G * MS[0] * np.cos(th1) - S12 * s * d2 ** 2; Q2 = u2 - G * MS[1] * np.cos(th2) + S12 * s * d1 ** 2
    det = S11 * S22 - (S12 * c) ** 2; a1 = (S22 * Q1 - S12 * c * Q2) / det; a2 = (S11 * Q2 - S12 * c * Q1) / det
    return a1, a2 - a1
def invdyn(q1, q2, w1, w2, a1, a2):
    """τ, дающие q̈ = (a1, a2) в состоянии."""
    th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = np.cos(th1 - th2); s = np.sin(th1 - th2); t1, t2 = a1, a1 + a2
    Q1 = S11 * t1 + S12 * c * t2; Q2 = S12 * c * t1 + S22 * t2
    u2 = Q2 + G * MS[1] * np.cos(th2) - S12 * s * d1 ** 2; u1 = Q1 + u2 + G * MS[0] * np.cos(th1) + S12 * s * d2 ** 2; return u1, u2
def f(z, u1, u2): a1, a2 = acc(z[0], z[1], z[2], z[3], u1, u2); return np.stack([z[2], z[3], a1, a2])
def flow(z, u1, u2, t, n=6):
    h = t / n
    for _ in range(n):
        k1 = f(z, u1, u2); k2 = f(z + h / 2 * k1, u1, u2); k3 = f(z + h / 2 * k2, u1, u2); k4 = f(z + h * k3, u1, u2); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z
def ingoal(z): return (np.abs(wrap(z[0] - C3[0])) <= RQ) & (np.abs(wrap(z[1] - C3[1])) <= RQ) & (np.abs(z[2]) <= RW_) & (np.abs(z[3]) <= RW_)

def solve_arcs(Y, C, Nn, newton=8):
    """Y (4,n) старты, C (4,n) центры спор, Nn (4,n) нормали — попарно. Возвращает t, u1, u2, s, ok."""
    dq = wrap(C[:2] - Y[:2]); wb = Y[2:] + C[2:]; t = 2 * (dq * wb).sum(0) / np.maximum((wb ** 2).sum(0), 1e-9)
    r = dq - wb * t / 2; s = (r * Nn[:2]).sum(0)                                                          # остаток вдоль нормали ⇒ s
    t = np.clip(t, 1e-3, TL); a1, a2 = (C[2] - Y[2]) / t, (C[3] - Y[3]) / t; zm = (Y + np.r_[dq + Y[:2], C[2:]][[0, 1, 2, 3]]) / 2
    u1, u2 = invdyn(zm[0], zm[1], zm[2], zm[3], a1, a2); Ct = C.copy(); Ct[:2] = Y[:2] + dq                    # цель без скачка 2π
    for _ in range(newton):
        Z = flow(Y, u1, u2, t); R = Z - (Ct + s * Nn)
        e = 1e-5; Z1 = flow(Y, u1 + e, u2, t); Z2 = flow(Y, u1, u2 + e, t); J = np.stack([f(Z, u1, u2), (Z1 - Z) / e, (Z2 - Z) / e, -Nn], -1)   # (4, n, 4)
        J = np.moveaxis(J, 1, 0); d = np.linalg.solve(J + 1e-12 * np.eye(4), np.moveaxis(R, 0, 1)[..., None])[..., 0]
        t = np.clip(t - d[:, 0], 1e-4, 2 * TL); u1 = np.clip(u1 - d[:, 1], -3, 3); u2 = np.clip(u2 - d[:, 2], -3, 3); s = s - d[:, 3]
    Z = flow(Y, u1, u2, t); res = np.abs(Z - (Ct + s * Nn)).max(0)
    ok = (res < 1e-7) & (t > 1e-3) & (t <= TL) & (np.abs(u1) <= 1 + 1e-9) & (np.abs(u2) <= 1 + 1e-9)
    return t, u1, u2, s, ok

if __name__ == '__main__':
    sys.path.insert(0, os.path.expanduser('~/claude-work/projects/spore/spores_2/v6'))
    from src.atlas6.manip3dyn import f as f6
    rng = np.random.default_rng(0); n = 2000
    Z = np.r_[rng.uniform(-np.pi, np.pi, (2, n)), rng.uniform(-2, 2, (2, n))]; U = rng.uniform(-1, 1, (2, n))
    mine = f(Z, U[0], U[1]); ref = np.stack([f6(Z[:, i], U[:, i], g=G) for i in range(200)], 1)
    print('динамика против v6 (200 точек): max |Δ| =', float(np.abs(mine[:, :200] - ref).max()))
    T = rng.uniform(.1, 1.2, n); E = flow(Z, U[0], U[1], T); Nn = np.zeros((4, n)); w = E[2:]; nn = np.hypot(*w); Nn[0], Nn[1] = -w[1] / nn, w[0] / nn
    t, u1, u2, s, ok = solve_arcs(Z, E, Nn); good = ok & (np.abs(t - T) < 1e-5) & (np.abs(u1 - U[0]) < 1e-5) & (np.abs(u2 - U[1]) < 1e-5) & (np.abs(s) < 1e-5)
    print('обратная задача (s = 0, случайные дуги t ≤ 1.2): найдено', round(float(ok.mean()), 3), 'точно та же дуга', round(float(good.mean()), 3))
