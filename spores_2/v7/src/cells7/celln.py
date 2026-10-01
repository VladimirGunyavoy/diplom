"""Клетка v7 в nD (PLAN 0з-7): сегмент (n−1)-диск ⊥ полю f_k(c) радиуса r × время τ вдоль потока, КВАДРАТИЧНАЯ локальная модель (research-4):
Φ(c+Qδ, t) ≈ X(t) + J(t)δ + ½H(t)[δ,δ]; J, H — конечные разности точного потока по узлам t_j (один раз при постройке, шаг HS).
Принадлежность — Гаусс–Ньютон по (δ, t) на модели. r, τ ≤ (R_MAX, TAU_MAX) уменьшаются, пока ошибка модели против точного потока ≤ TOL."""
import numpy as np

R_MAX, TAU_MAX, TOL, HALO, NT, HS = 0.2, 1.0, 1e-2, 1.1, 16, 0.08


class SysN:
    """flow(y,k,t): y (N,n) нормированные координаты → (N,n). per: периоды (0 — нет). box: (lo, hi) нормированные."""
    def __init__(self, name, n, K, flow, box, per=None, umax=None):
        self.name, self.n, self.K, self.flow, self.box = name, n, K, flow, (np.array(box[0], float), np.array(box[1], float))
        self.per = np.zeros(n) if per is None else np.array(per, float)
    def wrap(self, d):
        d = np.array(d, float); m = self.per > 0
        if m.any(): d[..., m] = (d[..., m] + self.per[m] / 2) % self.per[m] - self.per[m] / 2
        return d
    def sample(self, rng, N): return self.box[0] + (self.box[1] - self.box[0]) * rng.random((N, self.n))


class CellN:
    def __init__(self, S, k, c, r=R_MAX, tau=TAU_MAX, tol=TOL, rng=None):
        self.S, self.k, self.c = S, k, np.array(c, float); n = S.n; self.m = n - 1
        e = 1e-3; f0 = (S.flow(self.c[None], k, e)[0] - S.flow(self.c[None], k, -e)[0]) / (2 * e); self.f0 = f0
        u, _, _ = np.linalg.svd(f0[:, None]); self.Q = u[:, 1:]                  # ортонормированный базис ⊥ f0
        self._model(); self.r, self.tau = r, tau; self.err = np.nan
        self._shrink(tol, rng or np.random.default_rng(0)); self._finish()

    def _model(self):
        """X (NT+1,n), J (NT+1,n,m), H (NT+1,n,m,m) на узлах 0..TAU_MAX — центральные разности точного потока (HS)."""
        S, m, n = self.S, self.m, self.S.n; h = HS; E = np.eye(m); pts = [np.zeros(m)]; idx = {}
        for a in range(m): pts += [h * E[a], -h * E[a]]
        for a in range(m):
            for b in range(a + 1, m): pts += [h * (E[a] + E[b]), h * (E[a] - E[b]), h * (-E[a] + E[b]), -h * (E[a] + E[b])]
        D = np.array(pts); Y = self.c[None] + D @ self.Q.T; self.ts = np.linspace(0, TAU_MAX, NT + 1); dt = TAU_MAX / NT
        X, J, H = [], [], []
        for j in range(NT + 1):
            if j: Y = S.flow(Y, self.k, dt)
            P = S.wrap(Y - Y[0]) + Y[0]; x0 = Y[0]; Jj = np.zeros((n, m)); Hj = np.zeros((n, m, m)); q = 1
            for a in range(m):
                p, mm = P[q] - x0, P[q + 1] - x0; Jj[:, a] = (p - mm) / (2 * h); Hj[:, a, a] = (p + mm) / h**2; q += 2
            for a in range(m):
                for b in range(a + 1, m):
                    v = (P[q] - P[q + 1] - P[q + 2] + P[q + 3]) / (4 * h * h); Hj[:, a, b] = Hj[:, b, a] = v; q += 4
            X.append(x0); J.append(Jj); H.append(Hj)
        self.X, self.J, self.H = np.array(X), np.array(J), np.array(H)

    def phi(self, d, t):
        """Модель Φ(δ, t) для d (N,m), t (N,) → (N,n); линейная интерполяция по узлам."""
        dt = self.ts[1]; i = np.clip(np.floor(t / dt).astype(int), 0, NT - 1); u = np.clip(t / dt - i, -0.5, 1.5)[:, None]
        X = self.X[i] * (1 - u) + self.X[i + 1] * u; J = self.J[i] * (1 - u[..., None]) + self.J[i + 1] * u[..., None]; H = self.H[i] * (1 - u[..., None, None]) + self.H[i + 1] * u[..., None, None]
        return X + np.einsum('nij,nj->ni', J, d) + 0.5 * np.einsum('nijk,nj,nk->ni', H, d, d)

    def _shrink(self, tol, rng):
        r, tau = self.r, self.tau; S = self.S
        U = rng.normal(size=(10, self.m)); U /= np.linalg.norm(U, axis=1)[:, None]; U = np.vstack([U, np.eye(self.m), -np.eye(self.m)])
        for _ in range(40):
            e = 0.0
            for j in (NT // 2, NT):
                tj = tau * j / NT; D = r * U; Yx = S.flow(self.c[None] + D @ self.Q.T, self.k, tj)
                # модель в момент tj при δ=D: tj может не попадать в узлы — берём phi
                e = max(e, float(np.max(np.abs(S.wrap(Yx - self.phi(D, np.full(len(D), tj)))))))
            if e <= tol or (r < 2e-3 and tau < 2e-2): break
            if r / R_MAX >= tau / TAU_MAX: r *= 0.7
            else: tau *= 0.8
        self.r, self.tau, self.err = r, tau, e

    def _finish(self):
        self.exit = self.phi(np.zeros((1, self.m)), np.array([self.tau]))[0]
        tt = np.linspace(0, self.tau, 9); self.path = self.phi(np.zeros((9, self.m)), tt)
        self.rad = float(np.max(np.linalg.norm(self.S.wrap(self.path - self.c), axis=1)) + self.r * float(np.max(np.linalg.norm(self.J[:, :, :], axis=(1, 2)))) + 1e-3)

    def locate(self, Y, halo=1.0, it=6):
        """(δ, t, внутри) по модели; Гаусс–Ньютон по (δ,t); внутри: |δ| ≤ halo·r, −(halo−1)τ ≤ t ≤ halo·τ."""
        S = self.S; Y = np.atleast_2d(Y); N, n, m = len(Y), S.n, self.m
        inside = np.zeros(N, bool); dd = np.zeros((N, m)); tt = np.zeros(N)
        d0 = np.linalg.norm(S.wrap(Y - self.c), axis=1); near = np.nonzero(d0 <= self.rad * 1.5)[0]
        if not len(near): return dd, tt, inside
        Z = Y[near]; P = S.wrap(Z[:, None, :] - self.path[None]); j = np.argmin(np.sum(P * P, -1), 1); t = np.linspace(0, self.tau, 9)[j]; d = np.zeros((len(Z), m)); hdt = 1e-3
        for _ in range(it):
            R = S.wrap(self.phi(d, t) - Z); M = np.empty((len(Z), n, n))
            for a in range(m):
                dp = d.copy(); dp[:, a] += 1e-4; M[:, :, a] = S.wrap(self.phi(dp, t) - self.phi(d, t)) / 1e-4
            M[:, :, m] = S.wrap(self.phi(d, t + hdt) - self.phi(d, t)) / hdt
            try: step = np.linalg.solve(M + 1e-9 * np.eye(n), -R[..., None])[..., 0]
            except np.linalg.LinAlgError: break
            d = d + step[:, :m]; t = np.clip(t + step[:, m], -0.3 * self.tau, 1.3 * self.tau); d = np.clip(d, -3 * self.r, 3 * self.r)
        R = np.max(np.abs(S.wrap(self.phi(d, t) - Z)), 1)
        ok = (R < 2 * TOL) & (np.linalg.norm(d, axis=1) <= halo * self.r) & (t >= -(halo - 1) * self.tau) & (t <= halo * self.tau)
        dd[near], tt[near], inside[near] = d, t, ok
        return dd, tt, inside

    def kernel_samples(self, rng, N=48):
        U = rng.normal(size=(N, self.m)); U /= np.linalg.norm(U, axis=1)[:, None]; U *= self.r * 0.97 * rng.random((N, 1)) ** (1 / self.m)
        return self.phi(U, rng.random(N) * self.tau * 0.97 + 0.01 * self.tau)
