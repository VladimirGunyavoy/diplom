"""Коридор (dt_i, u_i) для общей системы (маятник): последовательность u из траектории Q-жадного агента, короткие сегменты (дребезг) сливаются,
длительности уточняет SLSQP: min Σdt_i при равенстве «конец = цель». Число переключений ≥ 1 (раскачка)."""
import numpy as np
from scipy.optimize import minimize
from .gcell import rk4


def agent_trace(P, x, dt=0.02, T_max=60.0, R=0.5):
    """Траектория Q-жадного агента `PendAtlas.run_agent`: (список u по шагам dt, конечное состояние с НЕсвёрнутым θ, дошёл)."""
    x = np.array(x, float); us = []; t = 0.0
    while t < T_max:
        if P.in_goal(x, R):
            return us, x, True
        q = [P.value(rk4(P.f, x, s * P.umax, P.tau)) for s in (+1, -1)]
        u = P.umax * (+1 if q[0] <= q[1] else -1); us.append(u); x = rk4(P.f, x, u, dt); t += dt
    return us, x, False


def segments(us, dt=0.02, min_len=0.15):
    """RLE по u; сегменты короче min_len сливаются с предыдущим (дребезг), соседние с равным u склеиваются."""
    seg = []
    for u in us:
        if seg and seg[-1][1] == u:
            seg[-1][0] += dt
        else:
            seg.append([dt, u])
    out = []
    for d, u in seg:
        if out and d < min_len:
            out[-1][0] += d
        elif out and out[-1][1] == u:
            out[-1][0] += d
        else:
            out.append([d, u])
    return [(d, u) for d, u in out]


def endpoint(f, x0, corridor):
    x = np.array(x0, float)
    for d, u in corridor:
        x = rk4(f, x, u, max(d, 0.0))
    return x


def optimize(f, x0, corridor, target, dt_max=0.01):
    """SLSQP: min Σdt при endpoint = target (θ — как в target, без свёртки). Возвращает (коридор, время, невязка)."""
    us = [u for _, u in corridor]; d0 = np.array([d for d, _ in corridor])
    ep = lambda d: endpoint(f, x0, list(zip(d, us))) - np.asarray(target, float)
    r = minimize(lambda d: d.sum(), d0, jac=lambda d: np.ones_like(d), bounds=[(0, None)] * len(d0),
                 constraints=[{'type': 'eq', 'fun': ep}], method='SLSQP', options={'maxiter': 300, 'ftol': 1e-12})
    return [(float(d), u) for d, u in zip(r.x, us)], float(r.x.sum()), float(np.hypot(*ep(r.x)))
