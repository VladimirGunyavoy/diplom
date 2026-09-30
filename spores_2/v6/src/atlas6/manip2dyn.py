"""Манипулятор 2 звена, ДИНАМИКА 4D (ступень 3 manipulator_plan): горизонтально, без гравитации, M(q)q̈ + C(q,q̇)q̇ = τ, |τ_i| ≤ 1.
Параметры: d=1, b=.5, a=2.5 (M = [[a+2b cos q2, d+b cos q2],[d+b cos q2, d]]). Слои — 4 угла коробки моментов. Состояние (q1, q2, w1, w2), q — тор."""
import numpy as np
A, B, D = 2.5, 0.5, 1.0
LAYERS4 = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]


def accel(x, tau):
    c, s = np.cos(x[..., 1]), np.sin(x[..., 1]); w1, w2 = x[..., 2], x[..., 3]; h = -B * s
    r1 = tau[0] + h * (2 * w1 * w2 + w2 ** 2); r2 = tau[1] - h * w1 ** 2
    m11, m12, m22 = A + 2 * B * c, D + B * c, D * np.ones_like(c); det = m11 * m22 - m12 ** 2
    return np.stack([(m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det], -1)


def f4(x, tau):
    return np.concatenate([x[..., 2:], accel(x, tau)], -1)


def flow4(x, s, t, dt_max=0.02):
    x = np.array(x, float); tau = LAYERS4[s]; n = max(1, int(np.ceil(abs(t) / dt_max))); h = t / n
    for _ in range(n):
        k1 = f4(x, tau); k2 = f4(x + h / 2 * k1, tau); k3 = f4(x + h / 2 * k2, tau); k4 = f4(x + h * k3, tau)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def clearance(P, obstacles, L1=1.0, L2=1.0):
    """Зазор звеньев-отрезков до дисков ((cx,cy),r): min по звеньям и дискам (dist − r), P (...,≥2) — углы q1,q2. <0 — столкновение. Векторно."""
    P = np.asarray(P, float); q1, q2 = P[..., 0], P[..., 1]
    e = np.stack([L1 * np.cos(q1), L1 * np.sin(q1)], -1); t = e + L2 * np.stack([np.cos(q1 + q2), np.sin(q1 + q2)], -1)
    def seg(a, b, c):
        d = b - a; u = np.clip(np.sum((c - a) * d, -1) / np.maximum(np.sum(d * d, -1), 1e-12), 0, 1)
        return np.linalg.norm(a + u[..., None] * d - c, axis=-1)
    z = np.zeros_like(e); out = np.full(q1.shape, np.inf)
    for c, r in obstacles:
        c = np.array(c, float); out = np.minimum(out, np.minimum(seg(z, e, c), seg(e, t, c)) - r)
    return out
