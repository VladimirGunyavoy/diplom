"""V по клеткам со струями (PLAN 0з п.3, явный градиент): узлы — точки центральной линии каждой клетки (NT+1 на клетку), в узле (V, p=∇V).
Рёбра: stay (узел j → j+1 той же клетки, цена h, p_j = Jᵀ p_{j+1}, J = [Lv f]_{j+1}[Lv f]_j⁻¹ — поток по модели, т.е. вариационные уравнения);
switch (точка узла лежит в ядре/гало клетки D по модели → V(x) ≈ V_m + p_m·(x − X_m), m — ближайший узел D по t; цена 0). Беллман до сходимости."""
import numpy as np
from scipy.spatial import cKDTree


class Jet:
    def __init__(self, S, layers, goal, halo=1.1, seed_lat=0.25, chains=(), s_tol=0.3, lat='lin', lam=0.0):
        self.lam = lam
        self.lat = lat
        self.s_tol = s_tol
        self.S, self.layers, self.goal = S, layers, np.array(goal, float)
        self.chains = [(C, T) for ch in chains for C, T in ch]
        cells = [(k, C) for k, L in enumerate(layers) for C in L]; self.cells = [C for C, _ in self.chains] + [C for _, C in cells]
        NT1 = len(self.cells[0].ts); self.NT1 = NT1; nc = len(self.cells); self.N = nc * NT1
        self.X = np.concatenate([C.X for C in self.cells]); self.F = np.concatenate([C.Xd for C in self.cells]); self.L = np.concatenate([C.V for C in self.cells])
        self.h = np.array([C.ts[1] - C.ts[0] for C in self.cells]); self.hn = np.repeat(self.h, NT1)
        self.t = np.concatenate([C.ts for C in self.cells]); self.cid = np.repeat(np.arange(nc), NT1)
        # stay: n -> n+1 внутри клетки
        j = np.arange(self.N) % NT1; self.stay = np.nonzero(j < NT1 - 1)[0]
        A = np.stack([self.L, self.F], -1)                                   # (N,2,2) столбцы Lv, f
        s = self.stay; Jm = A[s + 1] @ np.linalg.inv(A[s]); self.Jt = np.swapaxes(Jm, 1, 2)
        self._switch(halo)
        self._seed(seed_lat)

    def _wrapped_tree(self):
        S = self.S; P = self.X.copy(); box = []
        for i in range(2):
            if S.per[i] > 0: P[:, i] = np.mod(P[:, i], S.per[i]); box.append(S.per[i])
            else: P[:, i] = P[:, i] - P[:, i].min() + 1.0; box.append(P[:, i].max() * 4 + 100)
        self._off = np.array([0.0 if S.per[i] > 0 else (self.X[:, i].min() - 1.0) for i in range(2)])
        return cKDTree(P, boxsize=box)

    def _pt(self, y):
        S = self.S; y = np.array(y, float)
        for i in range(2):
            y[..., i] = np.mod(y[..., i], S.per[i]) if S.per[i] > 0 else y[..., i] - self._off[i] * 1.0
        return y

    def _switch(self, halo):
        S = self.S; tree = self._wrapped_tree(); src, dst, dd = [], [], []
        for d, D in enumerate(self.cells):
            R = float(np.max(np.linalg.norm(S.wrap(D.X - D.c), axis=1)) + 1.2 * D.r * np.max(np.linalg.norm(D.V, axis=1))) + 0.05
            idx = np.array(tree.query_ball_point(self._pt(D.c), R), int)
            idx = idx[self.cid[idx] != d] if len(idx) else idx
            if not len(idx): continue
            s, t, ins = D.locate(self.X[idx], halo)
            ins = ins & (np.abs(s) <= self.s_tol * D.r); q = idx[ins]
            if not len(q): continue
            jm = np.clip(np.rint(t[ins] / self.h[d]).astype(int), 0, self.NT1 - 1); m = d * self.NT1 + jm
            src.append(q); dst.append(m); dd.append(S.wrap(self.X[q] - self.X[m]))
        self.src = np.concatenate(src); self.dst = np.concatenate(dst); self.d = np.concatenate(dd)
        B = np.stack([self.F[self.dst], self.L[self.dst]], -1); co = np.linalg.solve(B, self.d[:, :, None])[:, :, 0]; self.ea, self.es = co[:, 0], co[:, 1]   # d = a·f + s·L в системе узла m

    def _seed(self, lat):
        self.V = np.full(self.N, np.inf); self.P = np.zeros((self.N, 2)); self.nseed = 0
        for c, (C, Te) in enumerate(self.chains):                       # цепочки клеток вдоль линий слоя, приходящих в цель: V = Te − t
            for j in range(self.NT1):
                n = c * self.NT1 + j; self.V[n] = Te - C.ts[j]; self.P[n] = np.linalg.solve(np.array([self.F[n], self.L[n]]), [-1.0, self.lam]); self.nseed += 1
        for c, C in enumerate(self.cells):
            s, t, ins = C.locate(self.goal, 1.0)
            if not (ins[0] and abs(s[0]) <= lat * C.r): continue
            for j in range(self.NT1):
                if C.ts[j] > t[0]: break
                n = c * self.NT1 + j; self.V[n] = t[0] - C.ts[j]
                self.P[n] = np.linalg.solve(np.array([self.F[n], self.L[n]]), [-1.0, 0.0]); self.nseed += 1

    def solve(self, maxit=4000, eps=1e-9):
        V, P = self.V, self.P; s = self.stay
        for it in range(maxit):
            Vb = np.full(self.N, np.inf); cand_stay = self.hn[s] + V[s + 1]; np.minimum.at(Vb, s, cand_stay)
            Pd = P[self.dst]; pl = np.einsum('ij,ij->i', Pd, self.L[self.dst]) * self.es
            vs = V[self.dst] + (np.einsum('ij,ij->i', Pd, self.F[self.dst]) * self.ea + np.abs(pl) if self.lat == 'abs' else np.einsum('ij,ij->i', Pd, self.d)); vs = np.where(np.isfinite(V[self.dst]), np.maximum(vs, 0.0), np.inf); np.minimum.at(Vb, self.src, vs)
            imp = Vb < V - eps
            if not imp.any(): break
            Pn = P.copy()
            ms = imp[s] & (cand_stay == Vb[s]); Pn[s[ms]] = np.einsum('nij,nj->ni', self.Jt[ms], P[s[ms] + 1])
            mw = imp[self.src] & (vs == Vb[self.src]); Pn[self.src[mw]] = P[self.dst[mw]]
            V = np.where(imp, Vb, V); P = Pn
        self.V, self.P, self.sweeps = V, P, it + 1

    def query(self, y, halo=1.0):
        y = np.atleast_2d(y); best = np.full(len(y), np.inf)
        for c, C in enumerate(self.cells):
            s, t, ins = C.locate(y, halo); ins = ins & (np.abs(s) <= self.s_tol * C.r)
            if not ins.any(): continue
            jm = np.clip(np.rint(t[ins] / self.h[c]).astype(int), 0, self.NT1 - 1); m = c * self.NT1 + jm
            v = self.V[m] + np.einsum('ij,ij->i', self.P[m], self.S.wrap(y[ins] - self.X[m])); v = np.where(np.isfinite(self.V[m]), np.maximum(v, 0), np.inf)
            ii = np.nonzero(ins)[0]; best[ii] = np.minimum(best[ii], v)
        return best


def goal_chain(S, k, goal, total=3.0, tau0=0.6):
    """Клетки слоя k, центральные линии которых лежат на линии потока, приходящей в цель (обратная ветвь); [(клетка, V в конце клетки)]."""
    from .cell import Cell
    out = []; T = 0.0; goal = np.array(goal, float)
    while T < total:
        tau = tau0
        for _ in range(6):
            C = Cell(S, k, S.rk4(goal, k, -(T + tau)), tau=tau)
            if C.tau >= tau - 1e-9: break
            tau = C.tau
        C = Cell(S, k, S.rk4(goal, k, -(T + tau)), r=C.r, tau=tau, tol=1e9)
        out.append((C, T + tau)); T += tau
        if np.any(np.abs(C.c) > 3) or tau < 1e-3: break
    return out
