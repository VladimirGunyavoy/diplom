"""V на решётке внутри клеток (PLAN 0з п.3, замена graph.py): узел (клетка C, s_i, t_j), s_i∈[−r,r], t_j∈[0,τ]; V(узел) = min( Δt + V(следующий узел вдоль потока),
V в клетке другого слоя, содержащей эту точку (билинейно по (s,t) соседа) ); конец клетки (t=τ) — переход в клетку того же слоя, содержащую X(τ,s).
Цель: узлы в радиусе eps от цели — 0. Итерация значений (Беллман–Форд), всё векторно. Боковая поправка — сама решётка по s."""
import numpy as np

NSV, NTV = 5, 9


def _nodes(C):
    s = np.linspace(-C.r, C.r, NSV); t = np.linspace(0, C.tau, NTV)
    X = np.array([np.interp(t, C.ts, C.X[:, d]) for d in range(2)]).T; V = np.array([np.interp(t, C.ts, C.V[:, d]) for d in range(2)]).T
    return s, t, X[None, :, :] + s[:, None, None] * V[None, :, :]       # (NSV, NTV, 2)


class Field:
    def __init__(self, S, layers, goal, eps=0.06, halo=1.0):
        self.S, self.layers, self.eps = S, layers, eps
        self.cells = [(k, C) for k, L in enumerate(layers) for C in L]; n = len(self.cells)
        self.Y = np.zeros((n, NSV, NTV, 2)); self.k = np.array([k for k, _ in self.cells])
        for i, (k, C) in enumerate(self.cells): self.Y[i] = _nodes(C)[2]
        flat = self.Y.reshape(-1, 2)
        # преемники: для каждого узла — переключение в клетки другого слоя; для t=τ — в клетки того же слоя (кроме своей)
        self.sw_idx = np.full((len(flat), 4), -1); self.sw_w = np.zeros((len(flat), 4)); best = np.full(len(flat), np.inf)
        for j, (kj, D) in enumerate(self.cells):
            s, t, ins = D.locate(flat, halo)
            ok = np.nonzero(ins)[0]
            for q in ok:
                i = q // (NSV * NTV); ki = self.k[i]; nt = q % NTV
                if kj == ki and not (nt == NTV - 1 and i != j): continue    # свой слой: только выход из конца клетки в соседнюю
                if kj == ki and i == j: continue
                score = abs(s[q]) / D.r
                if score < best[q]: best[q] = score; self.sw_idx[q], self.sw_w[q] = self._interp(j, D, s[q], t[q])
        self.goal = np.linalg.norm(S.wrap(self.Y - np.array(goal, float)), axis=-1) < eps
        self.V = np.where(self.goal, 0.0, 1e9)
        self.dt = np.array([C.tau / (NTV - 1) for _, C in self.cells])

    def _interp(self, j, D, s, t):
        """Индексы 4 соседних узлов решётки клетки j (плоские) и билинейные веса для (s,t)."""
        fs = np.clip((s + D.r) / (2 * D.r) * (NSV - 1), 0, NSV - 1 - 1e-9); ft = np.clip(t / D.tau * (NTV - 1), 0, NTV - 1 - 1e-9)
        a, b = int(fs), int(ft); u, v = fs - a, ft - b; base = j * NSV * NTV
        idx = [base + a * NTV + b, base + (a + 1) * NTV + b, base + a * NTV + b + 1, base + (a + 1) * NTV + b + 1]
        return np.array(idx), np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])

    def solve(self, sweeps=400):
        n = len(self.cells); V = self.V.reshape(-1).copy(); goal = self.goal.reshape(-1); has = self.sw_idx[:, 0] >= 0
        idx = np.where(self.sw_idx >= 0, self.sw_idx, 0); w = np.where(self.sw_idx >= 0, self.sw_w, 0.0)
        step = np.repeat(self.dt, NSV * NTV)
        for it in range(sweeps):
            old = V.copy(); Vc = V.reshape(n, NSV, NTV)
            nxt = np.full_like(Vc, 1e9); nxt[:, :, :-1] = Vc[:, :, 1:] + self.dt[:, None, None]       # вдоль потока
            fin = (V[idx] < 1e8) & (w > 1e-12); ws = np.where(fin, w, 0).sum(1)                       # билинейно по конечным углам (остальные не учитываются)
            sw = np.where(has & (ws > 1e-12), (np.where(fin, V[idx] * w, 0).sum(1)) / np.maximum(ws, 1e-12), 1e9)   # переключение/выход
            V = np.minimum(np.minimum(nxt.reshape(-1), sw), V); V[goal] = 0
            if np.max(np.abs(np.where(V < 1e8, V, 0) - np.where(old < 1e8, old, 0))) < 1e-9 and ((V < 1e8) == (old < 1e8)).all(): break
        self.V = V.reshape(n, NSV, NTV); self.sweeps = it + 1; return self

    def query(self, y):
        best = np.inf
        for j, (k, C) in enumerate(self.cells):
            s, t, ins = C.locate(y, 1.0)
            if ins[0]:
                idx, w = self._interp(j, C, s[0], t[0]); v = float((self.V.reshape(-1)[idx] * w).sum()); best = min(best, v)
        return best
