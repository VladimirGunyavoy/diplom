"""V по клеткам спор (рекомендация hub-research-7 `knowledge/research/v7_spore_cells.md`, идея пользователя): клетки `cover_layer` (ядро+гало 10%),
узлы (s,t) в клетке, V*(точка) = min по клеткам ЛЮБОГО слоя, содержащим точку, билинейно по узлам клетки;
V*(узел) = min_k [Δt + V*(φ_k(узел, Δt))] — шаг ТОЧНЫМ потоком, никаких переходов «та же точка в другой клетке» за нулевое время.
Неизвестное — большое КОНЕЧНОЕ BIG, Якоби до сходимости. Агент: каждые dt слой argmin_k [dt + V*(φ_k(q, dt))]."""
import numpy as np
from scipy.spatial import cKDTree

BIG = 1e3


def rk4v(S, k, Y, T, n=12):
    """Поток слоя k точек Y (N,2) за время T (N,) (любого знака), n шагов rk4."""
    Y = np.array(Y, float); h = (np.asarray(T, float) / n)[:, None]
    for _ in range(n):
        k1 = S.f(Y, k); k2 = S.f(Y + h / 2 * k1, k); k3 = S.f(Y + h / 2 * k2, k); k4 = S.f(Y + h * k3, k); Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Y


class SporeV:
    def __init__(self, S, cells, goal, m=7, nt=9, halo=1.1):
        """cells — список Cell (оба слоя, поле .k); goal(Y)->bool-маска цели в нормированных координатах."""
        self.S, self.cells, self.goal, self.m, self.nt, self.halo = S, cells, goal, m, nt, halo
        K = len(cells); self.K = K
        self.sg = np.linspace(-1, 1, m); self.tg = np.linspace(-(halo - 1), halo, nt)          # доли r и τ
        self.r = np.array([c.r for c in cells]); self.tau = np.array([c.tau for c in cells]); self.kk = np.array([c.k for c in cells])
        self.cen = np.array([c.c for c in cells])
        sj = self.sg[None, :, None] * self.r[:, None, None] * halo; ti = self.tg[None, None, :] * self.tau[:, None, None]
        n_ = np.array([c.n for c in cells])
        base = self.cen[:, None, None, :] + sj[..., None] * n_[:, None, None, :]                   # (K,m,1,2)
        base = np.broadcast_to(base, (K, m, nt, 2)); T = np.broadcast_to(ti, (K, m, nt))
        P = np.empty((K, m, nt, 2))
        for k in (0, 1):
            msk = self.kk == k
            if msk.any(): P[msk] = rk4v(S, k, base[msk].reshape(-1, 2), T[msk].ravel(), n=24).reshape(-1, m, nt, 2)
        self.P = P.reshape(-1, 2); self.N = len(self.P)
        d = np.linalg.norm(S.wrap(P - self.cen[:, None, None, :]), axis=-1).reshape(K, -1); self.rad = d.max(1) * 1.05 + 1e-6
        self.tree = cKDTree(self.cen); self.Rmax = self.rad.max()
        self.dtc = (self.tau * halo - (-(halo - 1) * self.tau)) / (nt - 1)                          # шаг узлов по t в клетке
        self.G = goal(self.P)

    def _pairs(self, Q):
        """Для точек Q (n,2): (qi, k, idx4, w4) по всем клеткам, содержащим точку (ядро+гало)."""
        qt = cKDTree(Q); lst = qt.query_ball_point(self.cen, self.rad)                                  # по клеткам — точки в её радиусе (память ∝ числу реальных пар)
        CI = np.repeat(np.arange(self.K), [len(l) for l in lst]); QI = np.concatenate([np.array(l, int) for l in lst]).astype(int) if len(CI) else np.zeros(0, int)
        o = np.argsort(CI, kind='stable'); QI, CI = QI[o], CI[o]; bnd = np.nonzero(np.diff(CI))[0] + 1; out = ([], [], [], [])
        for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(CI)]):
            if a == b: continue
            c = CI[a]; q = QI[a:b]; s, t, ins = self.cells[c].locate(Q[q], self.halo)
            if not ins.any(): continue
            q, s, t = q[ins], s[ins], t[ins]
            fs = (s / (self.r[c] * self.halo) + 1) / 2 * (self.m - 1); ft = (t / self.tau[c] + (self.halo - 1)) / (2 * self.halo - 1) * (self.nt - 1)
            j = np.clip(np.floor(fs).astype(int), 0, self.m - 2); i = np.clip(np.floor(ft).astype(int), 0, self.nt - 2); u = np.clip(fs - j, 0, 1); v = np.clip(ft - i, 0, 1)
            b0 = (c * self.m + j) * self.nt + i; out[0].append(q); out[1].append(np.full(len(q), c))
            out[2].append(np.stack([b0, b0 + 1, b0 + self.nt, b0 + self.nt + 1], 1)); out[3].append(np.stack([(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v], 1))
        if not out[0]: return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4))
        return tuple(np.concatenate(x) for x in out)

    def _vmin(self, V, npts, pr):
        qi, _, idx, w = pr; val = np.sum(V[idx] * w, 1); out = np.full(npts, BIG); np.minimum.at(out, qi, val); return out

    def build(self, log=None):
        """Рёбра: для каждого узла и слоя k — точка φ_k(узел, Δt клетки) и её пары (клетка, веса)."""
        self.E = []; Tn = np.repeat(self.dtc, self.m * self.nt)
        for k in (0, 1):
            Y = rk4v(self.S, k, self.P, Tn, n=12); self.E.append((self._pairs(Y), len(Y)))
            if log: log('edges', k, len(self.E[-1][0][0]))

    def solve(self, it=20000, tol=1e-9, log=None):
        V = np.full(self.N, BIG); V[self.G] = 0.0; Tn = np.repeat(self.dtc, self.m * self.nt)
        for n in range(it):
            Vn = np.full(self.N, BIG)
            for (pr, npts) in self.E: Vn = np.minimum(Vn, Tn + self._vmin(V, npts, pr))
            Vn[self.G] = 0.0; Vn = np.minimum(Vn, BIG); d = np.max(np.abs(Vn - V)); V = Vn
            if log and n % 200 == 0: log(n, d)
            if d < tol: break
        self.Vn = V; self.iters = n; return V

    def value(self, Q):
        Q = np.atleast_2d(Q); return self._vmin(self.Vn, len(Q), self._pairs(Q))

    def rollout(self, Q0, dt=0.02, tmax=40.0, eps=0.0):
        """Q0 (n,2): агент — каждые dt слой argmin [dt + V*(φ_k(q,dt))]; T (inf — не дошёл), число переключений."""
        Y = np.array(Q0, float); n = len(Y); T = np.full(n, np.inf); act = np.ones(n, bool); last = np.full(n, -1); sw = np.zeros(n, int); t = 0.0
        done = self.goal(Y); T[done] = 0; act &= ~done
        while act.any() and t < tmax:
            ia = np.nonzero(act)[0]; Yn = [rk4v(self.S, k, Y[ia], np.full(len(ia), dt), n=2) for k in (0, 1)]
            vk = np.stack([self.value(Yn[k]) for k in (0, 1)], 1); kb = np.argmin(vk, 1); ok = vk[np.arange(len(ia)), kb] < BIG / 2
            if eps > 0:                                                    # гистерезис (hub-research-7): держать прежний слой, если он хуже лучшего не более чем на eps
                li = last[ia]; hold = (li >= 0) & (vk[np.arange(len(ia)), np.maximum(li, 0)] <= vk[np.arange(len(ia)), kb] + eps); kb = np.where(hold, li, kb)
            ch = (last[ia] >= 0) & (kb != last[ia]) & ok; sw[ia[ch]] += 1; last[ia] = np.where(ok, kb, last[ia])
            Y[ia] = np.where(kb[:, None] == 0, Yn[0], Yn[1]); t += dt; fin = self.goal(Y[ia]); T[ia[fin]] = t; act[ia[fin]] = False; act[ia[~ok]] = False
        return T, sw
