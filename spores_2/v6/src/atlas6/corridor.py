"""Коридор (dt_i, u_i) (source_doc §8): последовательность управлений из траектории агента, дребезг схлопывается в «u, затем −u»
(для DI по принципу максимума ≤ 1 переключения), длительности уточняет оптимизатор так, чтобы конец = цель."""
import numpy as np
from scipy.optimize import least_squares
from .cell import flow
from .agent import T_star


def agent_controls(x, v, cells, R_goal, dt=0.01, tol=0.05, umax=1.0):
    """Траектория агента (как run_agent), но возвращает список u по шагам dt и признак «дошёл»."""
    C = np.array([cv.cell.c for cv in cells]); G = np.array([cv.grad for cv in cells])
    T0 = T_star(x, v); t = 0.0; us = []
    while t < 3 * T0 + 1:
        if np.hypot(x, v) < tol:
            return us, True
        if np.hypot(x, v) < R_goal:
            s = x + v * abs(v) / 2; u = -np.sign(s) if abs(s) > 1e-9 else -np.sign(v)
        else:
            i = int(np.argmin((C[:, 0] - x) ** 2 + (C[:, 1] - v) ** 2)); u = -np.sign(G[i, 1]) or 1.0
        us.append(u); x, v = flow(np.array([x, v]), u * umax, dt); t += dt
    return us, False


def compress(us, dt=0.01):
    """Дребезг → коридор из ≤ 2 сегментов [(dt1, u1), (dt2, −u1)]: u1 — управление первого сегмента, dt1 — время до последней смены знака, dt2 — остаток."""
    if not us:
        return []
    u1 = us[0]
    last = max((k for k in range(len(us)) if us[k] == u1), default=0)      # последний шаг с u1
    return [(dt * (last + 1), u1), (dt * (len(us) - last - 1), -u1)]


def endpoint(x0, v0, corridor, umax=1.0):
    p = np.array([x0, v0], float)
    for d, u in corridor:
        p = flow(p, u * umax, max(d, 0.0))
    return p


def optimize(x0, v0, corridor, umax=1.0):
    """Длительности dt_i при фиксированных u_i: минимизируем невязку конца (x, v) → (0, 0). Порядок управлений берётся из коридора, но
    у агента первые шаги могут дребезжать — поэтому пробуем и обратный порядок и берём допустимый (невязка < 1e-6) с меньшим временем.
    Возвращает (коридор, время, невязка)."""
    best = None
    d0 = [d for d, _ in corridor]; us0 = [u for _, u in corridor]
    for us, ds in ((us0, d0), ([-u for u in us0], d0[::-1]), (us0, [1.0, 1.0]), ([-u for u in us0], [1.0, 1.0])):
        f = lambda d: endpoint(x0, v0, [(di, ui) for di, ui in zip(d, us)], umax)
        r = least_squares(f, ds, bounds=(0, np.inf), xtol=1e-14, ftol=1e-14)
        res = float(np.hypot(*f(r.x))); T = float(sum(r.x))
        cand = ([(float(d), u) for d, u in zip(r.x, us)], T, res)
        if best is None or (res < 1e-6 <= best[2]) or ((res < 1e-6) == (best[2] < 1e-6) and (T if res < 1e-6 else res) < (best[1] if best[2] < 1e-6 else best[2])):
            best = cand
    return best
