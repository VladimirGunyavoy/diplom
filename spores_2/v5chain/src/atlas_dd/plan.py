"""A4: управления из произвольной точки. Жадный по T с шагом dt: из 8 режимов берётся тот, что минимизирует dt + T(конец) (T — interp_T3).
Возвращает [(режим, время)] и траекторию; конец — когда state в допуске цели (или дальше T не падает)."""
import numpy as np
from math import hypot, pi
from .modes import Straight, Rotate, Arc
from .lattice import interp_T3


def rsr_controls(st, vmax=1.0, wmax=1.0):
    """Точный финиш в (0,0,0): поворот к цели (или задом) – прямая – поворот; [(режим, время)] и конечное состояние."""
    x, y, th = st
    d = hypot(x, y)
    if d < 1e-9:
        dth = (0.0 - th + pi) % (2 * pi) - pi
        return ([('r+' if dth > 0 else 'r-', abs(dth) / wmax)] if abs(dth) > 1e-12 else []), (x, y, 0.0)
    bearing = np.arctan2(-y, -x); best = None
    for hd, m in ((bearing, 'f'), (bearing + pi, 'b')):
        a1 = (hd - th + pi) % (2 * pi) - pi; a2 = (0.0 - hd + pi) % (2 * pi) - pi
        t = (abs(a1) + abs(a2)) / wmax + d / vmax
        if best is None or t < best[0]:
            best = (t, m, a1, a2)
    _, m, a1, a2 = best
    c = []
    for a in (a1,):
        if abs(a) > 1e-12: c.append(('r+' if a > 0 else 'r-', abs(a) / wmax))
    c.append((m, d / vmax))
    if abs(a2) > 1e-12: c.append(('r+' if a2 > 0 else 'r-', abs(a2) / wmax))
    return c, (0.0, 0.0, 0.0)


def _search(st, modes, T, n, h, dt, depth, cur):
    """Лучшая последовательность из depth шагов dt: минимизирует depth·dt + T(конец); возвращает [(имя, состояние)] если лучше cur."""
    best = [cur - 1e-9, None]
    def rec(s, seq, d):
        if d == depth:
            t = interp_T3(s[0], s[1], s[2], T, n, h)
            if not np.isnan(t) and depth * dt + t < best[0]:
                best[0], best[1] = depth * dt + t, list(seq)
            return
        for name, m in modes:
            lab, sv = m.to_chart(s)
            nx = m.from_chart(lab, sv + m.rate * dt)
            rec(nx, seq + [(name, nx)], d + 1)
    rec(st, [], 0)
    return best[1]


def rollout(state, T, n, h=1.0, vmax=1.0, wmax=1.0, dt=0.25, tol_xy=0.15, tol_th=0.15, max_steps=200, finish=False, refine_dt=False, lookahead=0, la_dt=0.15):
    modes = [('f', Straight(1, vmax)), ('b', Straight(-1, vmax)), ('r+', Rotate(1, wmax)), ('r-', Rotate(-1, wmax))] + \
            [(f'a{sv:+d}{sw:+d}', Arc(sv, sw, vmax, wmax)) for sv in (1, -1) for sw in (1, -1)]
    st, ctrl, traj = tuple(state), [], [tuple(state)]
    for _ in range(max_steps * (4 if refine_dt else 1)):
        x, y, th = st
        if hypot(x, y) < tol_xy and abs((th + pi) % (2 * pi) - pi) < tol_th and not refine_dt:
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
            if refine_dt and dt > 0.011:            # тонкий финиш: T перестал падать на шаге dt — уменьшить шаг
                dt /= 5
                continue
            if lookahead:                            # бокового сдвига одним шагом не убрать — перебор последовательностей глубины lookahead
                seq = _search(st, modes, T, n, h, la_dt, lookahead, cur)
                if seq:
                    for name, nx in seq:
                        if ctrl and ctrl[-1][0] == name:
                            ctrl[-1] = (name, ctrl[-1][1] + la_dt)
                        else:
                            ctrl.append((name, la_dt))
                        st = nx; traj.append(st)
                    continue
            break
        if ctrl and ctrl[-1][0] == best[1]:
            ctrl[-1] = (best[1], ctrl[-1][1] + dt)
        else:
            ctrl.append((best[1], dt))
        st = best[2]; traj.append(st)
    if finish:                     # тонкий финиш: с места остановки — точный RSR до (0,0,0)
        x, y, th = st
        if hypot(x, y) > 1e-9 or abs((th + pi) % (2 * pi) - pi) > 1e-9:
            c, e = rsr_controls(st, vmax, wmax)
            ctrl += c; traj.append(e)
    return ctrl, traj


def rollout_multi(state, fields, **kw):
    """A5-lite: многоуровневый финиш. fields = [(T, n, h), ...] от грубого к тонкому; каждый следующий rollout стартует с места остановки предыдущего.
    Тонкое поле у цели исправляет недооценку линейной интерполяции (боковой сдвиг ε стоит ~√ε, а не ~ε)."""
    ctrl, traj, st = [], [tuple(state)], tuple(state)
    for T, n, h in fields:
        x, y, _ = st
        if abs(x) >= (n - 1) * h or abs(y) >= (n - 1) * h:
            continue                                   # состояние вне области тонкого поля
        c, tr = rollout(st, T, n, h, **kw)
        for name, t in c:
            if ctrl and ctrl[-1][0] == name:
                ctrl[-1] = (name, ctrl[-1][1] + t)
            else:
                ctrl.append((name, t))
        traj += tr[1:]; st = tr[-1]
    return ctrl, traj
