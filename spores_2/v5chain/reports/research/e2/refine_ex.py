"""e3: SLSQP коридора с ТОЧНЫМ якобианом ограничений (прямая чувствительность по тому же RK4, что и flow) вместо численного.
Значения ограничений — те же, что у corridor_nd.refine: g(конец)=hw²−wr(x)² и (WM−WMARG)−|w_i| в ns точках каждого сегмента; шаг RK4 в под-интервале — ceil(t/ns/.02) шагов, как flow.
Производная f(x,u) и шага — casadi AD (SX). Дифф с corridor_nd.refine: только jac= для constraints. Фиксированное число шагов RK4 на под-интервал берётся по текущему d (как flow)."""
import numpy as np, casadi as ca
from scipy.optimize import minimize
from src.atlas6.corridor_nd import merge
import common

def make_step(name):
    if name == 'dp1':
        G1, MS, S11, S12, S22 = 1., (2.5, 1.), 2.5, 1., 1.
        def fca(x, u):
            q1, q2, w1, w2 = x[0] + np.pi / 2, x[1], x[2], x[3]; th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2; c = ca.cos(th1 - th2); sn = ca.sin(th1 - th2)
            Q1 = u[0] - u[1] - G1 * MS[0] * ca.cos(th1) - S12 * sn * d2 ** 2; Q2 = u[1] - G1 * MS[1] * ca.cos(th2) + S12 * sn * d1 ** 2
            det = S11 * S22 - (S12 * c) ** 2; a1 = (S22 * Q1 - S12 * c * Q2) / det; a2 = (S11 * Q2 - S12 * c * Q1) / det
            return ca.vertcat(w1, w2, a1, a2 - a1)
    elif name == 'manip':
        AA, BB, DD = 2.5, .5, 1.
        def fca(x, u):
            c, sn = ca.cos(x[1]), ca.sin(x[1]); w1, w2 = x[2], x[3]; h = -BB * sn; r1 = u[0] - h * (2 * w1 * w2 + w2 ** 2); r2 = u[1] + h * w1 ** 2
            m11, m12, m22 = AA + 2 * BB * c, DD + BB * c, DD; det = m11 * m22 - m12 ** 2
            return ca.vertcat(w1, w2, (m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det)
    else: raise ValueError(name)
    x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 2); h = ca.SX.sym('h')
    k1 = fca(x, u); k2 = fca(x + h / 2 * k1, u); k3 = fca(x + h / 2 * k2, u); k4 = fca(x + h * k3, u); xn = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function('st', [x, u, h], [ca.vertcat(xn, ca.reshape(ca.jacobian(xn, x), 16, 1), ca.jacobian(xn, h))])

class EX:
    def __init__(s, name, US, hw, per, ns):
        s.F = make_step(name); s.US = np.asarray(US, float); s.hw = hw; s.per = np.asarray(per, float); s.ns = ns; s.last = None; s.stat = dict(n=0, t=0.)
    def ev(s, x0, seq, d, wl):
        key = d.tobytes()
        if s.last is not None and s.last[0] == key: return s.last[1]
        n = len(seq); S = np.zeros((4, n)); x = np.array(x0, float); ns = s.ns; W = []; JW = []; wi, wm = wl
        for k, (sg, t) in enumerate(zip(seq, d)):
            r = max(1, int(np.ceil(abs(t / ns) / .02))); h = t / ns / r; u = s.US[sg]
            for _ in range(ns):
                for _ in range(r):
                    o = np.asarray(s.F(x, u, h)).ravel(); x = o[:4]; S = o[4:20].reshape(4, 4, order="F") @ S; S[:, k] += o[20:] / (ns * r)
                W.append(x[list(wi)].copy()); JW.append(S[list(wi)].copy())
        xe = common.wrap(x, s.per); g = s.hw ** 2 - xe ** 2; Jg = -2 * xe[:, None] * S
        W = np.array(W); JW = np.array(JW)                                  # (ns*n, nw), (ns*n, nw, n)
        val = np.concatenate([g, (wm - np.abs(W)).ravel()]); jac = np.vstack([Jg, (-np.sign(W)[:, :, None] * JW).reshape(-1, n)])
        s.last = (key, (val, jac)); return s.last[1]
    def refine(s, flow, x0, path, g, tries=3, seed=0, clear=None, ns=16, wlim=None):
        seq, d0 = merge(path); rng = np.random.default_rng(seed); best = (np.inf, None)
        fun = lambda d: s.ev(x0, seq, np.asarray(d, float), wlim)[0]; jac = lambda d: s.ev(x0, seq, np.asarray(d, float), wlim)[1]
        con = {'type': 'ineq', 'fun': fun, 'jac': jac}
        for k in range(tries):
            di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
            r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, 1.5 * d0.sum() + 1)] * len(d0), constraints=[con], method='SLSQP', options=dict(maxiter=80, ftol=1e-9))
            if np.all(fun(r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
        ok = best[1] is not None
        return (best[0] if ok else float(d0.sum())), seq, (best[1] if ok else d0), ok

def install(name, US, hw, per, ns):
    from src.atlas6 import corridor_nd
    c = EX(name, US, hw, per, ns); corridor_nd.refine = c.refine; return c
