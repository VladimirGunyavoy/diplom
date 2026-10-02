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
    def __init__(self, S, cells, goal, m=None, nt=None, halo=1.1, dt_edge=None, hs=0.02, ht=0.02, mmax=41):
        """cells — список Cell (оба слоя, поле .k); goal(Y)->bool-маска цели в нормированных координатах.
        Густота узлов адаптивна: шаг по s ≈ hs (норм. ед.), по t ≈ ht·(τ-единица времени); m/nt заданы числом — одинаковы для всех (тест)."""
        self.S, self.cells, self.goal, self.halo = S, cells, goal, halo; self.dt_edge = dt_edge
        K = len(cells); self.K = K
        self.r = np.array([c.r for c in cells]); self.tau = np.array([c.tau for c in cells]); self.kk = np.array([c.k for c in cells]); self.cen = np.array([c.c for c in cells])
        self.mc = np.full(K, m) if m else np.clip(np.ceil(2 * halo * self.r / hs).astype(int) + 1, 5, mmax)
        self.ntc = np.full(K, nt) if nt else np.clip(np.ceil((2 * halo - 1) * self.tau / ht).astype(int) + 1, 5, mmax)
        self.cnt = self.mc * self.ntc; self.off = np.r_[0, np.cumsum(self.cnt)]; self.N = int(self.off[-1])
        self.P = np.empty((self.N, 2)); self.rad = np.empty(K)
        for c, cell in enumerate(cells):
            mc, nc = self.mc[c], self.ntc[c]; sj = np.linspace(-1, 1, mc) * self.r[c] * halo; ti = np.linspace(-(halo - 1), halo, nc) * self.tau[c]
            base = cell.c[None, None, :] + sj[:, None, None] * cell.n[None, None, :]; base = np.broadcast_to(base, (mc, nc, 2)).reshape(-1, 2); T = np.broadcast_to(ti[None, :], (mc, nc)).ravel()
            self.P[self.off[c]:self.off[c + 1]] = rk4v(S, cell.k, base, T, n=24)
            self.rad[c] = np.linalg.norm(S.wrap(self.P[self.off[c]:self.off[c + 1]] - cell.c), axis=1).max() * 1.05 + 1e-6
        self.tree = cKDTree(self.cen); self.Rmax = self.rad.max()
        self.dtc = (self.tau * (2 * halo - 1)) / (self.ntc - 1)                                     # шаг узлов по t в клетке
        self.G = goal(self.P)

    def _pairs(self, Q):
        """Для точек Q (n,2): (qi, k, idx4, w4) по всем клеткам, содержащим точку (ядро+гало)."""
        Q = self.S.wrap(Q); n0 = len(Q); per = np.asarray(self.S.per, float)                      # точки приводим в [−период/2, период/2) (агент может накручивать обороты)
        Qi = np.concatenate([Q + sh * per for sh in (-1, 0, 1)]) if per.any() else Q               # периодичность: образы точек со сдвигами ±период (locate сам оборачивает)
        qt = cKDTree(Qi); lst = qt.query_ball_point(self.cen, self.rad)                                  # по клеткам — точки в её радиусе (память ∝ числу реальных пар)
        CI = np.repeat(np.arange(self.K), [len(l) for l in lst]); QI = np.concatenate([np.array(l, int) for l in lst]).astype(int) if len(CI) else np.zeros(0, int)
        QI = QI % n0; pk = np.unique(QI.astype(np.int64) * self.K + CI); QI, CI = pk // self.K, pk % self.K                      # образы → исходные точки, без дублей
        o = np.argsort(CI, kind='stable'); QI, CI = QI[o], CI[o]; bnd = np.nonzero(np.diff(CI))[0] + 1; out = ([], [], [], [])
        for a, b in zip(np.r_[0, bnd], np.r_[bnd, len(CI)]):
            if a == b: continue
            c = CI[a]; q = QI[a:b]; s, t, ins = self.cells[c].locate(Q[q], self.halo)
            if not ins.any(): continue
            q, s, t = q[ins], s[ins], t[ins]
            mc, nc = self.mc[c], self.ntc[c]
            fs = (s / (self.r[c] * self.halo) + 1) / 2 * (mc - 1); ft = (t / self.tau[c] + (self.halo - 1)) / (2 * self.halo - 1) * (nc - 1)
            j = np.clip(np.floor(fs).astype(int), 0, mc - 2); i = np.clip(np.floor(ft).astype(int), 0, nc - 2); u = np.clip(fs - j, 0, 1); v = np.clip(ft - i, 0, 1)
            b0 = self.off[c] + j * nc + i; out[0].append(q); out[1].append(np.full(len(q), c))
            out[2].append(np.stack([b0, b0 + 1, b0 + nc, b0 + nc + 1], 1)); out[3].append(np.stack([(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v], 1))
        if not out[0]: return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4))
        return tuple(np.concatenate(x) for x in out)

    def _vmin(self, V, npts, pr):
        qi, _, idx, w = pr; val = np.sum(V[idx] * w, 1); out = np.full(npts, BIG); np.minimum.at(out, qi, val); return out

    def _Tn(self):
        """Шаг по времени ребра узла: свой dtc клетки или заданный dt_edge (полулагранжева диффузия ∝ h²/dt — при мелких клетках шаг надо брать крупнее)."""
        return np.full(self.N, self.dt_edge) if self.dt_edge else np.repeat(self.dtc, self.cnt)

    def build(self, log=None):
        """Рёбра: для каждого узла и слоя k — точка φ_k(узел, Δt клетки) и её пары (клетка, веса)."""
        self.E = []; Tn = self._Tn()
        for k in (0, 1):
            Y = rk4v(self.S, k, self.P, Tn, n=12); self.E.append((self._pairs(Y), len(Y), self.goal(Y)))        # конец ребра в цели — V=0 (узлы цели редки: траектория проскакивает цель между узлами)
            if log: log('edges', k, len(self.E[-1][0][0]))

    def solve(self, it=20000, tol=1e-9, log=None):
        V = np.full(self.N, BIG); V[self.G] = 0.0; Tn = self._Tn()
        for n in range(it):
            Vn = np.full(self.N, BIG)
            for (pr, npts, eg) in self.E: Vn = np.minimum(Vn, Tn + np.where(eg, 0.0, self._vmin(V, npts, pr)))
            Vn[self.G] = 0.0; Vn = np.minimum(Vn, BIG); d = np.max(np.abs(Vn - V)); V = Vn
            if log and n % 200 == 0: log(n, d)
            if d < tol: break
        self.Vn = V; self.iters = n; return V

    def value(self, Q):
        Q = np.atleast_2d(Q); return np.where(self.goal(Q), 0.0, self._vmin(self.Vn, len(Q), self._pairs(Q)))

    def rollout(self, Q0, dt=0.02, tmax=40.0, eps=0.0, sub=4):
        """Q0 (n,2): агент — каждые dt слой argmin [dt + V*(φ_k(q,dt))]; T (inf — не дошёл), число переключений."""
        Y = np.array(Q0, float); n = len(Y); T = np.full(n, np.inf); act = np.ones(n, bool); last = np.full(n, -1); sw = np.zeros(n, int); t = 0.0
        done = self.goal(Y); T[done] = 0; act &= ~done
        while act.any() and t < tmax:
            ia = np.nonzero(act)[0]; Yn = [rk4v(self.S, k, Y[ia], np.full(len(ia), dt), n=2) for k in (0, 1)]
            vk = np.stack([self.value(Yn[k]) for k in (0, 1)], 1); kb = np.argmin(vk, 1); ok = vk[np.arange(len(ia)), kb] < BIG / 2
            if eps > 0:                                                    # гистерезис (hub-research-7): держать прежний слой, если он хуже лучшего не более чем на eps
                li = last[ia]; hold = (li >= 0) & (vk[np.arange(len(ia)), np.maximum(li, 0)] <= vk[np.arange(len(ia)), kb] + eps); kb = np.where(hold, li, kb)
            ch = (last[ia] >= 0) & (kb != last[ia]) & ok; sw[ia[ch]] += 1; last[ia] = np.where(ok, kb, last[ia])
            kk = kb.copy(); yy = Y[ia].copy(); fin = np.zeros(len(ia), bool); tf = np.full(len(ia), np.inf)
            for sidx in range(sub):                                     # подшаги: цель проверяется на каждом (не проскочить квадрат цели внутри dt, hub-research-7)
                for k in (0, 1):
                    m_ = kk == k
                    if m_.any(): yy[m_] = rk4v(self.S, k, yy[m_], np.full(m_.sum(), dt / sub), n=2)
                g_ = self.goal(yy) & ~fin; tf[g_] = t + (sidx + 1) * dt / sub; fin |= g_
            Y[ia] = yy; t += dt; T[ia[fin]] = tf[fin]; act[ia[fin]] = False; act[ia[~ok]] = False
        return T, sw


def fill_gaps(S, cells, goal, Cell, lo=-1.0, hi=1.0, n=30000, rounds=3, seed=5, log=None):
    """Заполнение дыр покрытия: точки вне ядра+гало ВСЕХ клеток → новая клетка (слои по очереди); точки внутри неё выбывают.
    Без этого точка вне клеток получает V=BIG и агент обрывается (DI: 5% точек, дошли 83% → после заполнения 100%)."""
    rng = np.random.default_rng(seed)
    for rnd in range(rounds):
        Vt = SporeV(S, cells, goal, hs=0.05, ht=0.05); Qa = rng.uniform(lo, hi, (n, 2)); qi = Vt._pairs(Qa)[0]; unc = np.nonzero(np.bincount(qi, minlength=len(Qa)) == 0)[0]
        if log: log('раунд', rnd, 'клеток', len(cells), 'дыр', len(unc))
        if len(unc) < n // 1000: break
        add = 0
        while len(unc):
            c = Cell(S, add % 2, Qa[unc[0]]); cells.append(c); add += 1; unc = unc[~c.locate(Qa[unc], 1.1)[2]]
    return cells
