"""A4: управления из произвольной точки. Жадный по T с шагом dt: из 8 режимов берётся тот, что минимизирует dt + T(конец) (T — interp_T3).
Возвращает [(режим, время)] и траекторию; конец — когда state в допуске цели (или дальше T не падает)."""
import numpy as np
from math import hypot, pi
from .modes import Straight, Rotate, Arc
from .lattice import interp_T3


def rollout(state, T, n, h=1.0, vmax=1.0, wmax=1.0, dt=0.25, tol_xy=0.15, tol_th=0.15, max_steps=200):
    modes = [('f', Straight(1, vmax)), ('b', Straight(-1, vmax)), ('r+', Rotate(1, wmax)), ('r-', Rotate(-1, wmax))] + \
            [(f'a{sv:+d}{sw:+d}', Arc(sv, sw, vmax, wmax)) for sv in (1, -1) for sw in (1, -1)]
    st, ctrl, traj = tuple(state), [], [tuple(state)]
    for _ in range(max_steps):
        x, y, th = st
        if hypot(x, y) < tol_xy and abs((th + pi) % (2 * pi) - pi) < tol_th:
            break
        best = None
        for name, m in modes:
            lab, s = m.to_chart(st)
            nx = m.from_chart(lab, s + m.rate * dt)
            t = interp_T3(nx[0], nx[1], nx[2], T, n, h)
            if not np.isnan(t) and (best is None or t < best[0]):
                best = (t, name, nx)
        cur = interp_T3(x, y, th, T, n, h)
        if best is None or (not np.isnan(cur) and best[0] >= cur - 1e-9):
            break
        if ctrl and ctrl[-1][0] == best[1]:
            ctrl[-1] = (best[1], ctrl[-1][1] + dt)
        else:
            ctrl.append((best[1], dt))
        st = best[2]; traj.append(st)
    return ctrl, traj
