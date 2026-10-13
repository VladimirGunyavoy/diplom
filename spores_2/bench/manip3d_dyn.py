"""research-21: трёхмерный двузвенный манипулятор (слово пользователя 2026-10-07 03:4x): поворот основания q0 вокруг вертикали,
плечо q1 (тангаж от горизонта), локоть q2 (относительный); 3 мотора |τ_i| ≤ 1, без гравитации и препятствий; состояние 6D (q0, q1, q2, w0, w1, w2).
Точечные массы: m1 в локте, m2 на конце; l1 = .5, l2 = 1, m1 = 5, m2 = 1 ⇒ подсистема (q1, q2) при w0 = 0, τ0 = 0 — РОВНО плоский
манипулятор v6 `manip2dyn.py` (a = 2.5, b = .5, d = 1). Основание — диск I0 = .5 (иначе при вертикальной руке M00 = 0, вырождение).
Лагранжиан: L = ½ M00(q1, q2) w0² + ½ wpᵀ Mp(q2) wp, M00 = I0 + m1 r1² + m2 r2², r1 = l1 cos q1, r2 = l1 cos q1 + l2 cos(q1 + q2) (радиусы от оси).
⇒ M00 ẇ0 + (∂1M00 w1 + ∂2M00 w2) w0 = τ0;   Mp ẇp + Cp wp − ½ ∂M00 w0² = τp  (центробежная связь: вращение основания «раскрывает» руку)."""
import numpy as np
I0, L1, L2, M1, M2 = .5, .5, 1., 5., 1.
A, B, D = (M1 + M2) * L1 ** 2 + M2 * L2 ** 2, M2 * L1 * L2, M2 * L2 ** 2          # = 2.5, .5, 1 (manip2dyn)
US = np.array([[a, b, c] for a in (-1., 1.) for b in (-1., 1.) for c in (-1., 1.)])  # 8 вершин коробки моментов
def m00(q1, q2):
    r1 = L1 * np.cos(q1); r2 = r1 + L2 * np.cos(q1 + q2); return I0 + M1 * r1 ** 2 + M2 * r2 ** 2, r1, r2
def f(x, tau):
    """x (..., 6), tau (..., 3) → ẋ"""
    q1, q2, w0, w1, w2 = x[..., 1], x[..., 2], x[..., 3], x[..., 4], x[..., 5]
    J, r1, r2 = m00(q1, q2); s1, s12 = np.sin(q1), np.sin(q1 + q2)
    d1 = -2 * (M1 * r1 * L1 * s1 + M2 * r2 * (L1 * s1 + L2 * s12)); d2 = -2 * M2 * r2 * L2 * s12   # ∂M00/∂q1, ∂M00/∂q2
    a0 = (tau[..., 0] - (d1 * w1 + d2 * w2) * w0) / J
    c, s = np.cos(q2), np.sin(q2); h = -B * s
    r1_ = tau[..., 1] - h * (2 * w1 * w2 + w2 ** 2) + .5 * d1 * w0 ** 2; r2_ = tau[..., 2] + h * w1 ** 2 + .5 * d2 * w0 ** 2   # знак Кориолиса: в manip2dyn v6 и growN manip — обратный (энергия не сохраняется, research-21)
    m11, m12 = A + 2 * B * c, D + B * c; det = m11 * D - m12 ** 2
    return np.stack([w0, w1, w2, a0, (D * r1_ - m12 * r2_) / det, (-m12 * r1_ + m11 * r2_) / det], -1)
def energy(x):
    J = m00(x[..., 1], x[..., 2])[0]; c = np.cos(x[..., 2]); w1, w2 = x[..., 4], x[..., 5]
    return .5 * J * x[..., 3] ** 2 + .5 * ((A + 2 * B * c) * w1 ** 2 + 2 * (D + B * c) * w1 * w2 + D * w2 ** 2)
def rk4(x, tau, h, n):
    for _ in range(n):
        k1 = f(x, tau); k2 = f(x + h / 2 * k1, tau); k3 = f(x + h / 2 * k2, tau); k4 = f(x + h * k3, tau); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
if __name__ == '__main__':
    rng = np.random.default_rng(0); X = np.c_[rng.uniform(-np.pi, np.pi, (200, 3)), rng.uniform(-2, 2, (200, 3))]
    E0 = energy(X); Y = rk4(X, np.zeros((200, 3)), .001, 3000); print('τ=0, 3 с: max |ΔE|/E =', np.abs(energy(Y) - E0).max() / E0.min())
    # мощность: dE/dt = τ·w при постоянном τ — проверка интегралом
    T = rng.uniform(-1, 1, (200, 3)); x = X.copy(); W = 0.; h = .001
    for _ in range(1000): xn = rk4(x, T, h, 1); W += h / 2 * ((T * x[:, 3:]).sum(1) + (T * xn[:, 3:]).sum(1)); x = xn
    print('τ≠0, 1 с: max |ΔE − ∫τ·w| =', np.abs(energy(x) - E0 - W).max())
    q = np.array([[0, np.pi / 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0.]]); print('M00 рука вверх / горизонтально:', m00(q[:, 1], q[:, 2])[0], '; ускорение рыскания при τ0=1:', 1 / m00(q[:, 1], q[:, 2])[0])
    x = np.zeros((1, 6)); x[0, 4] = 0; xs = rk4(np.array([[0, 0, 0, 1., 0, 0]]), np.zeros((1, 3)), .001, 1000); print('вращение основания w0=1, рука горизонтально, 1 с без моментов → q1, q2:', xs[0, 1:3].round(4), '(равновесие: должно остаться 0)')
    xs = rk4(np.array([[0, .3, .2, 1., 0, 0]]), np.zeros((1, 3)), .001, 1000); print('… рука под углом (q1 .3, q2 .2) → q1, q2, w0 через 1 с:', xs[0, [1, 2, 3]].round(4), '(раскрывается к горизонту, w0 падает)')
def flow(P, s, t, dt_max=.05, g=0.):
    """API коридора v6 (как manip3dyn.flow): слой s — вершина US[s], время t (может быть < 0), rk4 с шагом ≤ dt_max"""
    P = np.asarray(P, float); n = max(1, int(np.ceil(abs(t) / dt_max))); return rk4(P, US[s], t / n, n)
import math
def _flow_scalar(x, tau, t, dt_max):
    """одна точка на math (SLSQP зовёт поток точкой; numpy-накладные ×13–22, `query_speed.md`)"""
    q0, q1, q2, w0, w1, w2 = (float(v) for v in x); t0, t1, t2 = (float(v) for v in tau); n = max(1, int(math.ceil(abs(t) / dt_max))); h = t / n
    def fd(q1, q2, w0, w1, w2):
        c1, s1 = math.cos(q1), math.sin(q1); c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2); r1 = L1 * c1; r2 = r1 + L2 * c12; J = I0 + M1 * r1 * r1 + M2 * r2 * r2
        d1 = -2 * (M1 * r1 * L1 * s1 + M2 * r2 * (L1 * s1 + L2 * s12)); d2 = -2 * M2 * r2 * L2 * s12; a0 = (t0 - (d1 * w1 + d2 * w2) * w0) / J
        c, s = math.cos(q2), math.sin(q2); hh = -B * s; ra = t1 - hh * (2 * w1 * w2 + w2 * w2) + .5 * d1 * w0 * w0; rb = t2 + hh * w1 * w1 + .5 * d2 * w0 * w0
        m11, m12 = A + 2 * B * c, D + B * c; det = m11 * D - m12 * m12; return a0, (D * ra - m12 * rb) / det, (-m12 * ra + m11 * rb) / det
    for _ in range(n):
        a1 = fd(q1, q2, w0, w1, w2); v1 = (w0, w1, w2)
        v2 = (w0 + h / 2 * a1[0], w1 + h / 2 * a1[1], w2 + h / 2 * a1[2]); a2 = fd(q1 + h / 2 * v1[1], q2 + h / 2 * v1[2], *v2)
        v3 = (w0 + h / 2 * a2[0], w1 + h / 2 * a2[1], w2 + h / 2 * a2[2]); a3 = fd(q1 + h / 2 * v2[1], q2 + h / 2 * v2[2], *v3)
        v4 = (w0 + h * a3[0], w1 + h * a3[1], w2 + h * a3[2]); a4 = fd(q1 + h * v3[1], q2 + h * v3[2], *v4)
        q0 += h / 6 * (v1[0] + 2 * v2[0] + 2 * v3[0] + v4[0]); q1 += h / 6 * (v1[1] + 2 * v2[1] + 2 * v3[1] + v4[1]); q2 += h / 6 * (v1[2] + 2 * v2[2] + 2 * v3[2] + v4[2])
        w0 += h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]); w1 += h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]); w2 += h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
    return np.array([q0, q1, q2, w0, w1, w2])
_flow_np = flow
def flow(P, s, t, dt_max=.05, g=0.):
    P = np.asarray(P, float); return _flow_scalar(P, US[s], t, dt_max) if P.ndim == 1 else _flow_np(P, s, t, dt_max)
