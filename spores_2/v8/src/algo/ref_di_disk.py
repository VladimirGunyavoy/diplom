"""Эталон времени быстродействия двойного интегратора до цели-КРУГА (v8, worker-1, по ТЗ research-1 «solve8»).

ДИ  ẋ = v, v̇ = u, |u| ≤ 1; цель — круг ‖(x, v)‖ ≤ RHO. Оптимальное управление bang-bang и (принцип максимума: p_v линейна по t) имеет не более одного
переключения. Поэтому
    T_ref(x0) = min( T_dir(+1), T_dir(−1),  min_{s=±1, t1 ≥ 0} [ t1 + T_in(state_s(t1), −s) ] ),
где state_s(t1) — состояние после t1 под u = s, T_in(y, u) — время ПЕРВОГО входа в круг под постоянным u (0, если y уже в круге), T_dir(s) = T_in(x0, s).
Вход в круг — точно: x(t)² + v(t)² − RHO² = 0 — квартика по t (коэффициенты ниже), первый неотрицательный вещественный корень (корни — собственные числа
сопровождающей матрицы, пакетно). Минимизация по t1 — крупная сетка + 4 раунда «сгущения» вокруг лучшей точки (шаг 5e-3 → 5e-9), без scipy; векторно по стартам.

    T_ref(X, rho=.2)        X: (K, 2) или (2,) — (x, v); возврат (K,) или скаляр
    T_point(X)              классическое время до НАЧАЛА координат (кривая переключения x = −v|v|/2) — верхняя оценка для T_ref (круг содержит начало)
    starts(K=200, seed=1, rho=.2, lim=2.5)  K равномерных точек в [−lim, lim]² вне круга, rng(seed)
"""
import numpy as np

TMAX1 = 12.0                # верхняя граница t1 (поле ±2.5: время до начала координат ≤ ~8)
GRID0 = 4801                # узлов крупной сетки по t1
ZOOM_N = 201                # узлов в каждом раунде сгущения
ZOOM_ROUNDS = 4
_TOL_IM = 1e-7              # |Im λ| ≤ TOL ⇒ корень вещественный (касание круга считается входом)


def _as2(X):
    X = np.asarray(X, float); one = X.ndim == 1
    return (X[None] if one else X), one


def T_point(X):
    """время до начала координат: x > −v|v|/2 ⇒ сначала u = −1: T = v + 2√(x + v²/2); иначе T = −v + 2√(−x + v²/2)"""
    X, one = _as2(X); x, v = X[:, 0], X[:, 1]
    above = x > -0.5 * v * np.abs(v)
    T = np.where(above, v + 2 * np.sqrt(np.maximum(x + 0.5 * v * v, 0.)), -v + 2 * np.sqrt(np.maximum(-x + 0.5 * v * v, 0.)))
    return float(T[0]) if one else T


def _flow(x, v, u, t):
    """точное состояние через t при постоянном u"""
    return x + v * t + 0.5 * u * t * t, v + u * t


def entry_time(x, v, u, rho):
    """первое t ≥ 0 с x(t)² + v(t)² = rho² (0, если уже внутри; inf, если круг не достигается). Массивы любой формы, u ∈ {+1, −1} (скаляр)."""
    x = np.asarray(x, float); v = np.asarray(v, float); shp = np.broadcast_shapes(x.shape, v.shape); x = np.broadcast_to(x, shp).ravel(); v = np.broadcast_to(v, shp).ravel()
    # g(t) = (x + v t + u t²/2)² + (v + u t)² − rho²  = ¼t⁴ + u v t³ + (v² + u x + 1) t² + 2 v (x + u) t + (x² + v² − rho²)   (u² = 1)
    c3 = u * v; c2 = v * v + u * x + 1.0; c1 = 2 * v * (x + u); c0 = x * x + v * v - rho * rho
    inside = c0 <= 0
    out = np.full(len(x), np.inf); out[inside] = 0.
    k = ~inside
    if k.any():
        n = int(k.sum()); M = np.zeros((n, 4, 4)); M[:, 0, 0] = -4 * c3[k]; M[:, 0, 1] = -4 * c2[k]; M[:, 0, 2] = -4 * c1[k]; M[:, 0, 3] = -4 * c0[k]
        M[:, 1, 0] = M[:, 2, 1] = M[:, 3, 2] = 1.0
        lam = np.linalg.eigvals(M); ok = (np.abs(lam.imag) <= _TOL_IM * np.maximum(1., np.abs(lam.real))) & (lam.real >= -1e-12)
        t = np.where(ok, np.maximum(lam.real, 0.), np.inf).min(1); out[k] = t
    return out.reshape(shp)


def _switch_total(x0, v0, s, t1, rho):
    """полное время t1 + T_in(state_s(t1), −s); либо вход под u = s раньше t1 (тогда total = время входа, ≤ t1)"""
    x1, v1 = _flow(x0, v0, s, t1)
    return t1 + entry_time(x1, v1, -s, rho)


def _t_switch_point(x, v, s):
    """классическое время первого участка до начала координат для первого управления s (если s не оптимально — просто разумная пробная точка ≥ 0)"""
    if s < 0: return np.maximum(0., v + np.sqrt(np.maximum(x + 0.5 * v * v, 0.)))
    return np.maximum(0., -v + np.sqrt(np.maximum(-x + 0.5 * v * v, 0.)))


def T_ref(X, rho=0.2, return_details=False):
    X, one = _as2(X); K = len(X); x0 = X[:, 0:1]; v0 = X[:, 1:2]
    best = np.full(K, np.inf); info = [None] * K
    for s in (1.0, -1.0):
        td = entry_time(X[:, 0], X[:, 1], s, rho)                                   # без переключения
        upd = td < best; best = np.where(upd, td, best)
        for i in np.flatnonzero(upd): info[i] = (s, None, td[i])
        t1 = np.linspace(0, TMAX1, GRID0)[None, :]
        F = _switch_total(x0, v0, s, t1, rho); j = np.argmin(F, 1); tg = t1[0, j]; fg = F[np.arange(K), j]; stp = TMAX1 / (GRID0 - 1)
        # локальная сетка вокруг классического времени переключения (до начала координат): при малом RHO окно попадания в круг в t1 уже шага крупной сетки
        tpt = _t_switch_point(X[:, 0], X[:, 1], s); w = max(8 * rho, 1e-4); nl = int(np.clip(2 * w / min(1e-3, rho / 10), ZOOM_N, 6001)); tl = np.maximum(0., tpt[:, None] + w * np.linspace(-1, 1, nl)[None, :])
        Fl = _switch_total(x0, v0, s, tl, rho); jl = np.argmin(Fl, 1); tlb = tl[np.arange(K), jl]; flb = Fl[np.arange(K), jl]; hl = (tl[:, 1] - tl[:, 0]) if tl.shape[1] > 1 else np.full(K, stp)
        use_l = flb < fg; tg = np.where(use_l, tlb, tg); stp_k = np.where(use_l, np.maximum(hl, 1e-12), stp)
        lo = np.maximum(0., tg - stp_k); hi = tg + stp_k
        for _ in range(ZOOM_ROUNDS):
            tt = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, ZOOM_N)[None, :]
            F = _switch_total(x0, v0, s, tt, rho); jj = np.argmin(F, 1); tb = tt[np.arange(K), jj]; h = (hi - lo) / (ZOOM_N - 1)
            lo = np.maximum(0., tb - h); hi = tb + h
        fb = F[np.arange(K), jj]; upd = fb < best - 0.0
        for i in np.flatnonzero(upd): info[i] = (s, tb[i], fb[i])
        best = np.where(upd, fb, best)
    if return_details: return (float(best[0]), info[0]) if one else (best, info)
    return float(best[0]) if one else best


def starts(K=200, seed=1, rho=0.2, lim=2.5):
    """K стартов равномерно по полю [−lim, lim]² вне круга цели (повторная выборка), rng(seed)"""
    rng = np.random.default_rng(seed); out = []
    while len(out) < K:
        p = rng.uniform(-lim, lim, 2)
        if np.hypot(*p) > rho: out.append(p)
    return np.array(out)


def brute_two_switch(x0, v0, rho, step=0.05, tmax=7.0):
    """независимая сверка: перебор последовательностей bang-bang до 2 переключений, шаги длительностей step; вход в круг — на ПОСЛЕДНЕМ участке точно.
    Возвращает минимальное найденное время (верхняя оценка оптимума с точностью ~ step)."""
    best = np.inf; ts = np.arange(0., tmax + 1e-9, step)
    for s in (1.0, -1.0):
        best = min(best, float(entry_time(x0, v0, s, rho)))                                                   # 0 переключений
        T1, T2 = np.meshgrid(ts, ts, indexing='ij')
        # 1 переключение: s на t1, затем −s до входа
        x1, v1 = _flow(x0, v0, s, ts); best = min(best, float((ts + entry_time(x1, v1, -s, rho)).min()))
        # 2 переключения: s на t1, −s на t2, затем s до входа
        xa, va = _flow(x0, v0, s, T1); xb, vb = _flow(xa, va, -s, T2)
        best = min(best, float((T1 + T2 + entry_time(xb, vb, s, rho)).min()))
    return best
