"""Схема B (5D): состояние (x, y, θ, v, ω), управления — ускорения (a, α), |a|≤amax, |α|≤αmax, |v|≤vmax, |ω|≤ωmax.
Решётка: (x, y) шага h на [−n·h, n·h]², θ равномерно nth значений (период 2π), v и ω — уровни шага dv=amax·dt, dω=αmax·dt (dt — шаг управления),
поэтому (v, ω) после шага снова точно на уровнях, интерполяция только по (x, y, θ). Шаг: v'=v+a·dt, ω'=ω+α·dt, θ'=θ+ω̄dt, сдвиг по среднему курсу и средней скорости.
Цена — итерации Беллмана (T только убывает) от цели; вне поля — штраф BIG."""
import numpy as np
BIG = 1e3


def make_grid(n, h, nth, vmax=1.0, wmax=1.0, amax=1.0, almax=1.0, dt=0.5):
    dv, dw = amax * dt, almax * dt
    mv, mw = int(round(vmax / dv)), int(round(wmax / dw))
    return dict(n=n, h=h, nth=nth, vmax=vmax, wmax=wmax, amax=amax, almax=almax, dt=dt, mv=mv, mw=mw, dv=dv, dw=dw)


def _interp(Tv, x, y, th, g):
    """Tv[nx, ny, nth] — срез при фиксированных (v, ω). Билинейно по (x, y), линейно по θ (циклически). Вне поля BIG."""
    n, h, nth = g['n'], g['h'], g['nth']
    fx, fy = x / h + n, y / h + n
    inside = (fx >= 0) & (fy >= 0) & (fx <= 2 * n) & (fy <= 2 * n)
    fx = np.clip(fx, 0, 2 * n - 1e-9); fy = np.clip(fy, 0, 2 * n - 1e-9)
    i, j = fx.astype(int), fy.astype(int); u, w = fx - i, fy - j
    ft = (th % (2 * np.pi)) / (2 * np.pi) * nth
    k = ft.astype(int) % nth; s = ft - np.floor(ft); k1 = (k + 1) % nth
    def bil(kk):
        return ((1 - u) * (1 - w) * Tv[i, j, kk] + u * (1 - w) * Tv[i + 1, j, kk] + (1 - u) * w * Tv[i, j + 1, kk] + u * w * Tv[i + 1, j + 1, kk])
    if g.get('nearest'):                       # ребро без интерполяции: конец округляется до узла (T — цена настоящего графа)
        i, j = np.clip(np.rint(fx).astype(int), 0, 2 * n), np.clip(np.rint(fy).astype(int), 0, 2 * n)
        kk = np.rint(ft).astype(int) % nth
        return np.where(inside, Tv[i, j, kk], BIG)
    r = (1 - s) * bil(k) + s * bil(k1)
    return np.where(inside, r, BIG)


def solve5(g, iters=200, tol=1e-6, verbose=False):
    """T[iv, iw, ix, iy, ith]; индексы v: −mv..mv → 0..2mv. Цель: (0,0,θ=0,0,0). Возвращает (T, число итераций)."""
    n, h, nth, dt = g['n'], g['h'], g['nth'], g['dt']
    mv, mw = g['mv'], g['mw']
    xs = (np.arange(2 * n + 1) - n) * h
    X, Y, TH = np.meshgrid(xs, xs, np.arange(nth) * 2 * np.pi / nth, indexing='ij')
    T = np.full((2 * mv + 1, 2 * mw + 1) + X.shape, BIG)
    goal = (mv, mw, n, n, 0)
    T[goal] = 0.0
    for it in range(iters):
        Tn = T.copy(); ch = 0.0
        for iv in range(2 * mv + 1):
            for iw in range(2 * mw + 1):
                v, w = (iv - mv) * g['dv'], (iw - mw) * g['dw']
                best = T[iv, iw].copy()
                for da in (-1, 0, 1):
                    for dal in (-1, 0, 1):
                        iv2, iw2 = iv + da, iw + dal
                        if not (0 <= iv2 <= 2 * mv and 0 <= iw2 <= 2 * mw):
                            continue
                        v2, w2 = (iv2 - mv) * g['dv'], (iw2 - mw) * g['dw']
                        vb, wb = (v + v2) / 2, (w + w2) / 2
                        thm = TH + wb * dt / 2
                        x2 = X + vb * dt * np.cos(thm); y2 = Y + vb * dt * np.sin(thm); th2 = TH + wb * dt
                        best = np.minimum(best, dt + _interp(T[iv2, iw2], x2, y2, th2, g))
                Tn[iv, iw] = best
        Tn[goal] = 0.0
        ch = float(np.max(T - Tn)); T = Tn
        if verbose:
            print(it, ch)
        if ch < tol:
            break
    return T, it + 1


def _step5(st, iv, iw, da, dal, g):
    mv, mw, dt = g['mv'], g['mw'], g['dt']
    iv2, iw2 = iv + da, iw + dal
    if not (0 <= iv2 <= 2 * mv and 0 <= iw2 <= 2 * mw):
        return None
    x, y, th, v, w = st
    v2, w2 = (iv2 - mv) * g['dv'], (iw2 - mw) * g['dw']
    vb, wb = (v + v2) / 2, (w + w2) / 2
    thm = th + wb * dt / 2
    return (x + vb * dt * np.cos(thm), y + vb * dt * np.sin(thm), th + wb * dt, v2, w2), iv2, iw2


def _snap(st, g, frac):
    h, nth = g['h'], g['nth']; x, y, th, v, w = st
    xs, ys = round(x / h) * h, round(y / h) * h
    dth = 2 * np.pi / nth; ts = round(th / dth) * dth
    if abs(x - xs) < frac * h: x = xs
    if abs(y - ys) < frac * h: y = ys
    if abs(th - ts) < frac * dth: th = ts
    return (x, y, th, v, w)


def rollout5(state, T, g, max_steps=40, depth=1, snap=0.0):
    """Жадно по T с перебором на depth шагов вперёд: минимизирует depth·dt + T(конец); выполняется первый шаг лучшей последовательности.
    snap>0: если (x,y,θ) ближе к узлу решётки, чем snap·шаг — состояние притягивается к узлу (лечит боковой сдвиг ~0.01, который интерполяция T раздувает: √ε). Возвращает (управления [(a,α)], траектория). Останов: в цели или max_steps; возвращается префикс до точки с наименьшей ошибкой (позиция+курс+v+ω)."""
    dt, mv, mw = g['dt'], g['mv'], g['mw']
    st = tuple(state)
    iv, iw = int(round(st[3] / g['dv'])) + mv, int(round(st[4] / g['dw'])) + mw
    ctrl, tr = [], [st]
    def err(s):
        return float(np.hypot(s[0], s[1]) + abs(np.sin(s[2] / 2)) * 2 + abs(s[3]) + abs(s[4]))
    def val(s, a, b, d):
        best = None
        for da in (-1, 0, 1):
            for dal in (-1, 0, 1):
                r = _step5(s, a, b, da, dal, g)
                if r is None:
                    continue
                s2, a2, b2 = r
                if err(s2) < 1e-6:                       # цель поглощает: дальше цена 0 (иначе «ждать» и «ехать сейчас» дают ничью и rollout топчется)
                    t = dt
                elif d == 1:
                    t = dt + float(_interp(T[a2, b2], np.array(s2[0]), np.array(s2[1]), np.array(s2[2]), g))
                else:
                    t = dt + val(s2, a2, b2, d - 1)[0]
                if best is None or t < best[0]:
                    best = (t, da, dal)
        return best
    for _ in range(max_steps):
        if err(st) < 1e-6:
            break
        t, da, dal = val(st, iv, iw, depth)
        st, iv, iw = _step5(st, iv, iw, da, dal, g)
        if snap > 0:
            st = _snap(st, g, snap)
        ctrl.append((da * g['amax'], dal * g['almax'])); tr.append(st)
    k = int(np.argmin([err(s) for s in tr]))                 # лучшая точка траектории (зависание/колебание отбрасываем)
    return ctrl[:k], tr[:k + 1]


def nearest_reachable(state, T, g):
    """Старт из покоя на недостижимом узле (T≥BIG/2, чётность подрешётки) → ближайший достижимый узел покоя по (x, y, θ) в индексах ±1. Возвращает (state', сдвиг xy, сдвиг θ)."""
    n, h, nth, mv, mw = g['n'], g['h'], g['nth'], g['mv'], g['mw']
    x, y, th, v, w = state
    ix, iy, k = int(round(x / h)) + n, int(round(y / h)) + n, int(round(th / (2 * np.pi / nth))) % nth
    best = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dk in (-1, 0, 1):
                jx, jy, kk = ix + dx, iy + dy, (k + dk) % nth
                if 0 <= jx <= 2 * n and 0 <= jy <= 2 * n and T[mv + int(round(v / g['dv'])), mw + int(round(w / g['dw'])), jx, jy, kk] < BIG / 2:
                    c = abs(dx) + abs(dy) + abs(dk)
                    if best is None or c < best[0]:
                        best = (c, jx, jy, kk)
    if best is None:
        return state, None, None
    _, jx, jy, kk = best
    return ((jx - n) * h, (jy - n) * h, kk * 2 * np.pi / nth, v, w), np.hypot((jx - ix) * h, (jy - iy) * h), abs(((kk - k + nth / 2) % nth) - nth / 2) * 2 * np.pi / nth


def rollout5_best(state, T, g, depths=(3, 4), **kw):
    """Несколько глубин перебора, берётся траектория с наименьшей конечной ошибкой (при равной — короче). Глубина 3 и 4 останавливаются в разных местах."""
    best = None
    for d in depths:
        c, tr = rollout5(state, T, g, depth=d, **kw)
        e = tr[-1]
        err = float(np.hypot(e[0], e[1]) + 2 * abs(np.sin(e[2] / 2)) + abs(e[3]) + abs(e[4]))
        if best is None or (err, len(c)) < (best[0], len(best[1])):
            best = (err, c, tr)
    return best[1], best[2]
