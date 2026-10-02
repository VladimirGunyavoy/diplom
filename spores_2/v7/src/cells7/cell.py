"""Клетка v7: сегмент ⊥ полю слоя через спору c (полуширина r) × время τ вдоль потока + ЛИНЕЙНАЯ модель потока из вариационных уравнений.
Φ(c + s·n, t) ≈ X(t) + s·V(t), V = J·n, V' = A(X)·V. Ядро: |s|≤r, 0≤t≤τ; гало: ×HALO (1.1) — на 10% шире/длиннее (PLAN 0з п.2а).
Принадлежность x клетке — Ньютон по (s,t) на модели (без rk4)."""
import numpy as np

R_MAX, TAU_MAX, TOL, HALO, NT = 0.2, 1.0, 1e-3, 1.1, 32


def herm(t, ts, P, D):
    """Кубический Эрмит по узлам ts (равномерным), значения P (nt+1,d), производные D → (значение, производная) в точках t (N,)."""
    h = ts[1] - ts[0]; i = np.clip(np.floor((t - ts[0]) / h).astype(int), 0, len(ts) - 2); u = ((t - ts[i]) / h)[:, None]
    p0, p1, d0, d1 = P[i], P[i + 1], D[i] * h, D[i + 1] * h
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    dh00, dh10, dh01, dh11 = 6 * u**2 - 6 * u, 3 * u**2 - 4 * u + 1, -6 * u**2 + 6 * u, 3 * u**2 - 2 * u
    return h00 * p0 + h10 * d0 + h01 * p1 + h11 * d1, (dh00 * p0 + dh10 * d0 + dh01 * p1 + dh11 * d1) / h


class Cell:
    def __init__(self, S, k, c, r=R_MAX, tau=TAU_MAX, tol=TOL):
        self.S, self.k, self.c = S, k, np.array(c, float)
        f0 = S.f(self.c, k); self.n = np.array([-f0[1], f0[0]]) / max(np.linalg.norm(f0), 1e-12)
        self.r, self.tau = r, tau; self.err = np.nan
        self._shrink(tol)
        self._build()

    def _traj(self, tau):
        """Центральная траектория X(t) и V(t) на NT+1 узлах (rk4 вместе с вариационным уравнением)."""
        S, k = self.S, self.k; h = tau / NT; X = [self.c.copy()]; V = [self.n.copy()]; x, v = X[0].copy(), V[0].copy()
        def der(x, v): return S.f(x, k), S.A(x, k) @ v
        for _ in range(NT):
            a1, b1 = der(x, v); a2, b2 = der(x + h / 2 * a1, v + h / 2 * b1); a3, b3 = der(x + h / 2 * a2, v + h / 2 * b2); a4, b4 = der(x + h * a3, v + h * b3)
            x = x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4); v = v + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4); X.append(x.copy()); V.append(v.copy())
        return np.array(X), np.array(V)

    def _model_err(self, r, tau):
        """Макс. ошибка модели в краевых точках сегмента (s=±r, ±r/2 · t=τ/2, τ) против точного rk4, нормированные координаты."""
        S, k = self.S, self.k; X, V = self._traj(tau); e = 0.0
        for s in (-r, -r / 2, r / 2, r):
            for j in (NT // 2, NT):
                ex = S.rk4(self.c + s * self.n, k, tau * j / NT); e = max(e, float(np.max(np.abs(S.wrap(ex - (X[j] + s * V[j]))))))
        return e

    def _shrink(self, tol):
        r, tau = self.r, self.tau
        for _ in range(40):
            e = self._model_err(r, tau)
            if e <= tol or (r < 1e-3 and tau < 1e-2): break
            if r / R_MAX >= tau / TAU_MAX: r *= 0.7
            else: tau *= 0.8
        self.r, self.tau, self.err = r, tau, e

    def _build(self):
        S, k = self.S, self.k; self.ts = np.linspace(0, self.tau, NT + 1); self.X, self.V = self._traj(self.tau)
        self.Xd = np.array([S.f(x, k) for x in self.X]); self.Vd = np.array([S.A(x, k) @ v for x, v in zip(self.X, self.V)])
        self.exit = self.X[-1].copy()

    def locate(self, Y, halo=1.0, it=6):
        """Для точек Y (N,2): (s, t, внутри) по модели; внутри — |s|≤halo·r и −(halo−1)τ ≤ t ≤ halo·τ. Ньютон по (s,t)."""
        S = self.S; Y = np.atleast_2d(Y); N = len(Y)
        d = S.wrap(Y[:, None, :] - self.X[None, :, :]); d2 = np.sum(d * d, -1); j0 = np.argmin(d2, 1)
        thr = 1.5 * (max(halo, 1.0) * self.r * np.max(np.linalg.norm(self.V, axis=1)) + max(self.ts[1] - self.ts[0], (max(halo, 1.0) - 1) * self.tau) * np.max(np.linalg.norm(self.Xd, axis=1))) + 1e-9      # гало по t выходит за [0,τ] — отступ от крайних выборок траектории
        near = d2[np.arange(N), j0] < thr * thr
        if not near.any(): return np.zeros(N), np.zeros(N), np.zeros(N, bool)
        if not near.all():                                   # предфильтр: Ньютон только для точек рядом с центральной траекторией
            s_, t_, in_ = self.locate(Y[near], halo, it); s = np.zeros(N); t = np.zeros(N); inside = np.zeros(N, bool); s[near], t[near], inside[near] = s_, t_, in_
            return s, t, inside
        j = j0; t = self.ts[j].copy(); s = np.zeros(N)
        for _ in range(it):
            tt = np.clip(t, -0.3 * self.tau, 1.3 * self.tau); X, Xp = herm(tt, self.ts, self.X, self.Xd); V, Vp = herm(tt, self.ts, self.V, self.Vd)
            res = S.wrap(X + s[:, None] * V - Y); a = Xp + s[:, None] * Vp; b = V; det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]; det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dt = -(res[:, 0] * b[:, 1] - res[:, 1] * b[:, 0]) / det; ds = -(a[:, 0] * res[:, 1] - a[:, 1] * res[:, 0]) / det
            t = np.clip(t + dt, -0.3 * self.tau, 1.3 * self.tau); s = np.clip(s + ds, -3 * self.r, 3 * self.r)
        tt = np.clip(t, -0.3 * self.tau, 1.3 * self.tau); X, _ = herm(tt, self.ts, self.X, self.Xd); V, _ = herm(tt, self.ts, self.V, self.Vd)
        ok_res = np.max(np.abs(S.wrap(X + s[:, None] * V - Y)), 1) < 5e-3
        e_ = 2e-2; inside = ok_res & (np.abs(s) <= halo * self.r * (1 + e_)) & (t >= -(halo - 1) * self.tau - e_ * self.tau) & (t <= halo * self.tau * (1 + e_))   # допуск на краях гало: узлы сетки лежат ровно на границе
        return s, t, inside

    def outline(self, m=24):
        """Контур ядра по модели: левая кромка s=−r, правая s=+r (m точек по t)."""
        t = np.linspace(0, self.tau, m); X, _ = herm(t, self.ts, self.X, self.Xd); V, _ = herm(t, self.ts, self.V, self.Vd)
        return X - self.r * V, X + self.r * V
