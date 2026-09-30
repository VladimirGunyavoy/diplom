"""H1 обобщённо (research/adaptive_vs_grid.md): дерево цепочек спор с отсечением по ядрам, без знания структуры системы.
Система — flow(P, s, t) (s — индекс слоя управления, t<0 назад), периодичность по θ через `period` (None или (per_x,)).
Спавн: от старта вперёд (выход слоя s за τ) и от цели назад; новая спора не создаётся, если в ядре (клетка rho в масштабированной метрике) уже есть спора ТОГО ЖЕ слоя (I1);
приоритет — A*: время от старта + оценка h(точка до цели). Бюджет N. V — Беллман по графу «спора → выход слоя за τ», Делоне-интерполяция, старт итераций K."""
import heapq
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class Sys:
    """flow(P, s, t): P (...,2) → (...,2); n_layers; scale (Lx, Lv) — масштаб ядра; per — период по 1-й координате (или None); goal_val(P)->V в цели или nan; in_goal(P)->bool; goal_pts — затравки обратного дерева."""
    def __init__(self, flow, n_layers, scale, goal_val, in_goal, goal_pts, per=None, lim=None):
        self.flow, self.L, self.scale, self.goal_val, self.in_goal, self.goal_pts, self.per, self.lim = flow, n_layers, np.asarray(scale, float), goal_val, in_goal, np.asarray(goal_pts, float), per, lim

    def wrap(self, P):
        P = np.array(P, float)
        if self.per: P[..., 0] = (P[..., 0] + self.per / 2) % self.per - self.per / 2
        return P

    def ok(self, p):
        return self.lim is None or (abs(p[1]) <= self.lim)


def build_tree(S, tau, x0s, N, rho, h=lambda p: 0.0, back=True, back_share=0.5, sw_w=0.0, T_max=1e9):
    """Возвращает P (N',2) споры. x0s — старты (список). h — оценка «до цели» для прямого приоритета. Ядра — по слою."""
    seen = set(); P = []
    key = lambda p, s: (s, int(np.floor(p[0] / (rho * S.scale[0]))), int(np.floor(p[1] / (rho * S.scale[1]))))
    def add(p, s):
        k = key(p, s)
        if k in seen: return False
        seen.add(k); P.append(p); return True
    pq = []; cnt = 0
    def push(p, t, fwd, ls=-1, ns=0):
        nonlocal cnt
        for s in range(S.L):
            e = S.wrap(S.flow(p, s, tau if fwd else -tau))
            if not S.ok(e): continue
            n2 = ns + (1 if (ls >= 0 and s != ls) else 0)
            if t + tau > T_max: continue
            pr = sw_w * n2 + t + tau + (h(e) if fwd else 0.0)                 # sw_w>0: цепочки без переключений раньше (обобщённые «веера»)
            cnt += 1; heapq.heappush(pq, (pr, cnt, s, fwd, e, t + tau, n2))
    for x0 in x0s:
        x0 = S.wrap(x0); add(x0, -1); push(x0, 0.0, True)
    if back:
        for g in S.goal_pts:
            g = S.wrap(g); add(g, -2); push(g, 0.0, False)
    nb = 0; nf = 0; capb = int(N * back_share) if back else 0
    while pq and len(P) < N:
        pr, _, s, fwd, e, t, n2 = heapq.heappop(pq)
        if not fwd and nb >= capb: continue
        if fwd and nf >= N - capb: continue
        if add(e, s):
            if fwd: nf += 1
            else: nb += 1
            push(e, t, fwd, s, n2)
    return np.array(P)


class Scattered:
    def __init__(self, S, P, tau, K=None, dmax=None):
        self.S, self.tau, self.dmax = S, tau, dmax; P = S.wrap(P); self.P = P
        self.Q = np.vstack([P, P + [S.per, 0], P - [S.per, 0]]) if S.per else P
        self.tri = Delaunay(self.Q); self.n = len(P)
        gv = S.goal_val(P); self.goal = ~np.isnan(gv); self.gv = gv
        self.K = K if K is not None else 3 * 60.0
        if dmax is not None:                                           # гало: интерполируем только в малых симплексах (диаметр ≤ dmax в масштабе), иначе — не оцениваем
            Z = self.Q / S.scale; T = Z[self.tri.simplices]; d = np.max([np.hypot(*(T[:, i] - T[:, j]).T) for i in range(3) for j in range(i + 1, 3)], axis=0); self.ok_s = d <= dmax
        self.ends = [S.wrap(S.flow(P, s, tau)) for s in range(S.L)]
        self.V = np.full(self.n, self.K); self.V[self.goal] = gv[self.goal]

    def _interp(self, V, E):
        Vq = np.tile(V, 3) if self.S.per else V
        lin = LinearNDInterpolator(self.tri, Vq)(E)
        if self.dmax is not None:
            si = self.tri.find_simplex(E); lin = np.where((si >= 0) & self.ok_s[np.maximum(si, 0)], lin, np.nan)
        if np.any(np.isnan(lin)):
            near = NearestNDInterpolator(self.Q, Vq)(E); lin = np.where(np.isnan(lin), near + 1.0 + (self.K if self.dmax is not None else 0.0), lin)
        return lin

    def solve(self, iters=2000, tol=1e-9):
        for it in range(iters):
            Vn = self.V.copy()
            for E in self.ends:
                Vn = np.minimum(Vn, self.tau + self._interp(self.V, E))
            Vn[self.goal] = self.gv[self.goal]; ch = float(np.max(self.V - Vn)); self.V = Vn
            if ch < tol: break
        return it + 1

    def value(self, x):
        return float(self._interp(self.V, self.S.wrap(np.atleast_2d(x)))[0])


def build_adaptive(S, tau, x0, N, rho, batch=8, wb=1.0, lam=1.0):
    """Без оракула: прямой A* по ТЕКУЩЕМУ V (решение по уже поставленным спорам пересчитывается каждый батч), обратное дерево — по близости к старту.
    Возвращает споры P. Ядра — по слою, как в build_tree."""
    seen = set(); P = []; gt = []; kind = []                     # gt — время от старта (вперёд) или до цели (назад)
    key = lambda p, s: (s, int(np.floor(p[0] / (rho * S.scale[0]))), int(np.floor(p[1] / (rho * S.scale[1]))))
    def add(p, s, g, k):
        kk = key(p, s)
        if kk in seen: return False
        seen.add(kk); P.append(p); gt.append(g); kind.append(k); return True
    x0 = S.wrap(x0); add(x0, -1, 0.0, 1)
    for g in S.goal_pts: add(S.wrap(g), -2, 0.0, 0)
    done = 0; sc = None
    while len(P) < N:
        Pa = np.array(P)
        if len(P) >= 8:
            sc = Scattered(S, Pa, tau); sc.solve()
            fin = sc.V < 0.9 * sc.K; Bp = Pa[fin]; Bv = sc.V[fin]
            def val(e, Bp=Bp, Bv=Bv):
                d = Bp - e
                if S.per: d[:, 0] = (d[:, 0] + S.per / 2) % S.per - S.per / 2
                return float(np.min(Bv + lam * np.hypot(d[:, 0] / S.scale[0], d[:, 1] / S.scale[1])))
        else:
            val = lambda e: 0.0
        cand = []
        for i in range(done, len(P)):
            fwd = kind[i] == 1 or (kind[i] == 2)
            for s in range(S.L):
                if kind[i] == 1:
                    e = S.wrap(S.flow(Pa[i], s, tau))
                    if not S.ok(e) or key(e, s) in seen: continue
                    cand.append((gt[i] + tau + val(e), 1, e, s, gt[i] + tau))
                else:
                    e = S.wrap(S.flow(Pa[i], s, -tau))
                    if not S.ok(e) or key(e, s) in seen: continue
                    d = np.hypot(*((e - x0) / S.scale)) if not S.per else np.hypot(*((np.array([(e[0] - x0[0] + S.per / 2) % S.per - S.per / 2, e[1] - x0[1]])) / S.scale))
                    cand.append((wb * d + gt[i] + tau, 0, e, s, gt[i] + tau))
        # frontier: незаконченные споры остаются кандидатами — не помечаем done, чтобы ядра отсекали повторы
        if not cand: break
        cf = sorted([c for c in cand if c[1] == 1], key=lambda c: c[0])[: batch // 2 + batch % 2]
        cb = sorted([c for c in cand if c[1] == 0], key=lambda c: c[0])[: batch // 2]
        n0 = len(P)
        for c in cf + cb:
            if len(P) >= N: break
            add(c[2], c[3], c[4], c[1])
        if len(P) == n0: break
    return np.array(P)


def build_until_junction(S, tau, x0, rho, dmax=None, sw_w=10.0, N0=40, Nmax=3000, extra=0.3, back_share=0.5):
    """Без горизонта-оракула (предложение research №3): удваиваем N, пока V(старт) не станет конечной (стыковка прямой и обратной цепочек), затем +extra спор."""
    N = N0
    while N <= Nmax:
        P = build_tree(S, tau, [x0], N, rho, sw_w=sw_w, back_share=back_share); sc = Scattered(S, P, tau, dmax=dmax); sc.solve()
        if sc.V[1 if False else 0] < 0.9 * sc.K: break        # P[0] — старт
        N *= 2
    P0, sc0 = P, sc                                                 # набор на момент стыковки; при расширении back_share-лимиты меняются — берём расширенный, только если стык сохранился
    N1 = int(len(P) * (1 + extra)); P = build_tree(S, tau, [x0], N1, rho, sw_w=sw_w, back_share=back_share)
    sc = Scattered(S, P, tau, dmax=dmax); sc.solve()
    return (P, sc) if sc.V[0] < 0.9 * sc.K else (P0, sc0)


def bidir_value(S, tau, x0, NB, NF, rho_b, rho_f, lam=1.0, halo=None):
    """Стыковка (research №1/№2): обратное дерево от цели — Дейкстра, V(b) точные по построению (V(пред) = τ + V(след), без интерполяции);
    прямое дерево от старта — A* с приоритетом g + оценка до ближайшей обратной споры, оценка(p) = min_b [V(b) + lam·|p−b|] (|·| в масштабе, гало ≤ halo).
    V(старт) = min по прямым спорам [g(p) + оценка(p)] — чистая минимизация по графу (верхняя оценка при lam ≥ реальной «цены расстояния»), монотонна по NB, NF.
    Возвращает (V, число обратных, число прямых)."""
    from scipy.spatial import cKDTree
    def core_keys(rho):
        return lambda p, s: (s, int(np.floor(p[0] / (rho * S.scale[0]))), int(np.floor(p[1] / (rho * S.scale[1]))))
    # обратное дерево
    kb = core_keys(rho_b); seen = set(); B = []; VB = []; pq = []; cnt = 0
    for g in S.goal_pts:
        g = S.wrap(g); seen.add(kb(g, -2)); B.append(g); VB.append(0.0)
        for s in range(S.L):
            e = S.wrap(S.flow(g, s, -tau))
            if S.ok(e): cnt += 1; heapq.heappush(pq, (tau, cnt, s, e))
    while pq and len(B) < NB:
        v, _, s, e = heapq.heappop(pq); k = kb(e, s)
        if k in seen: continue
        seen.add(k); B.append(e); VB.append(v)
        for s2 in range(S.L):
            e2 = S.wrap(S.flow(e, s2, -tau))
            if S.ok(e2): cnt += 1; heapq.heappush(pq, (v + tau, cnt, s2, e2))
    B = np.array(B); VB = np.array(VB)
    if S.per:
        Bc = np.vstack([B, B + [S.per, 0], B - [S.per, 0]]); Vc = np.tile(VB, 3)
    else:
        Bc, Vc = B, VB
    tree = cKDTree(Bc / S.scale)
    def est(p):
        # ближайшие обратные споры: минимум V(b)+lam·d по k=6 соседям (V растёт вдоль цепочки; ближайший не всегда лучший)
        d, i = tree.query(p / S.scale, k=min(8, len(Bc)))
        d = np.atleast_1d(d); i = np.atleast_1d(i)
        m = np.inf if halo is None else halo
        val = Vc[i] + lam * d; val = np.where(d <= m, val, np.inf)
        return float(np.min(val))
    # прямое дерево A*
    kf = core_keys(rho_f); seenf = set(); x0 = S.wrap(x0); seenf.add(kf(x0, -1)); best = est(x0); nf = 1
    pq = []; cnt = 0
    def push(p, g):
        nonlocal cnt
        for s in range(S.L):
            e = S.wrap(S.flow(p, s, tau))
            if S.ok(e):
                cnt += 1; heapq.heappush(pq, (g + tau + min(est(e), 1e6 if np.isinf(est(e)) else est(e)), cnt, s, e, g + tau))
    push(x0, 0.0)
    while pq and nf < NF:
        f, _, s, e, g = heapq.heappop(pq); k = kf(e, s)
        if k in seenf: continue
        seenf.add(k); nf += 1; best = min(best, g + est(e)); push(e, g)
    return best, len(B), nf


def _seg_x(A, B, C, D):
    """Пересечение отрезков AB и CD (2D). Возвращает (s∈[0,1] по AB, r∈[0,1] по CD) или None."""
    d1 = B - A; d2 = D - C; den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-14: return None
    w = C - A; s = (w[0] * d2[1] - w[1] * d2[0]) / den; r = (w[0] * d1[1] - w[1] * d1[0]) / den
    return (s, r) if (0 <= s <= 1 and 0 <= r <= 1) else None


def cross_value(S, tau, x0, NB, NF, rho_b, rho_f, dt=None, sw_w=10.0, cell=0.5, sw_wb=None):
    """Точная стыковка (research №4): обратное дерево хранит дуги (подотрезки dt) с ТОЧНЫМИ V (V(родитель)+время назад);
    из каждой прямой споры каждым слоем интегрируем шаг τ подотрезками и ищем ПЕРЕСЕЧЕНИЕ с обратной дугой — стык = t + V(×) — реальная траектория ⇒ верхняя оценка.
    Также: вход в клетку цели. Прямые споры — build_tree-порядок (мало переключений раньше). Возвращает (V, nb, nf)."""
    dt = tau / 5 if dt is None else dt; m = int(round(tau / dt))
    kb = lambda rho: (lambda p, s: (s, int(np.floor(p[0] / (rho * S.scale[0]))), int(np.floor(p[1] / (rho * S.scale[1])))))
    wrapd = (lambda dx: (dx + S.per / 2) % S.per - S.per / 2) if S.per else (lambda dx: dx)
    hsh = {}; segs = []
    def cellk(p): return (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)))
    def add_arc(p, s, v):                                           # обратная дуга от p (V=v) слоем s назад на τ
        pts = S.flow(p, s, -dt * np.arange(m + 1)[:, None] if False else 0)  # заглушка
    # обратное дерево (Дейкстра) с дугами
    k_ = kb(rho_b); seen = set(); B = []; pq = []; cnt = 0
    def arcs(p, s, v):
        t = -dt * np.arange(m + 1); P = np.array([S.flow(p, s, tt) for tt in t])      # P[k]: назад на k·dt, V = v + k·dt
        for k in range(m):
            i = len(segs); segs.append((P[k], P[k + 1], v + k * dt, v + (k + 1) * dt))
            for c in {cellk(S.wrap(P[k])), cellk(S.wrap(P[k + 1]))}:
                hsh.setdefault(c, []).append(i)
    for g in S.goal_pts:
        g = S.wrap(g); seen.add(k_(g, -2)); B.append(g)
        for s in range(S.L):
            cnt += 1; heapq.heappush(pq, (tau, cnt, s, g, 0.0, -1, 0))
    swb = sw_w if sw_wb is None else sw_wb
    while pq and len(B) < NB:
        _, _, s, p, vp, ls, ns = heapq.heappop(pq); v = vp + tau
        e = S.wrap(S.flow(p, s, -tau))
        if not S.ok(e): continue
        kk = k_(e, s)
        if kk in seen: continue
        seen.add(kk); B.append(e); arcs(p, s, vp)
        for s2 in range(S.L):
            n2 = ns + (1 if (ls >= 0 and s2 != s) else 0)               # переключение в обратной цепочке (слой меняется)
            cnt += 1; heapq.heappush(pq, (swb * n2 + v + tau, cnt, s2, e, v, s, n2))
    best = np.inf
    SC = np.array([q[0] for q in segs]); SD = np.array([q[1] for q in segs]); SVC = np.array([q[2] for q in segs]); SVD = np.array([q[3] for q in segs])
    def near(A):
        c = cellk(S.wrap(A)); out = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cc = (c[0] + di, c[1] + dj)
                if cc in hsh: out.update(hsh[cc])
        if S.per:
            per_c = int(round(S.per / cell))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    cc = ((c[0] + di + per_c // 2) % per_c - per_c // 2, c[1] + dj)
                    if cc in hsh: out.update(hsh[cc])
        return out
    def scan(p, s, g):
        nonlocal best
        Pf = np.array([S.flow(p, s, dt * j) for j in range(m + 1)])
        for j in range(m + 1):
            if S.in_goal(S.wrap(Pf[j])): best = min(best, g + j * dt); return
        idx = None
        for j in range(m):
            A, Bp = Pf[j], Pf[j + 1]
            ids = list(near(A))
            if not ids: continue
            C = SC[ids]; D = SD[ids]; VC = SVC[ids]; VD = SVD[ids]
            if S.per:
                dx = A[0] - C[:, 0]; sh = (dx + S.per / 2) % S.per - S.per / 2 - dx
                C = C + np.stack([sh, 0 * sh], 1); D = D + np.stack([sh, 0 * sh], 1)
            d1 = Bp - A; d2 = D - C; den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]; w = C - A
            with np.errstate(divide='ignore', invalid='ignore'):
                sp = (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / den; rp = (w[:, 0] * d1[1] - w[:, 1] * d1[0]) / den
            ok = (np.abs(den) > 1e-14) & (sp >= 0) & (sp <= 1) & (rp >= 0) & (rp <= 1)
            if ok.any():
                best = min(best, float(np.min(g + (j + sp[ok]) * dt + VC[ok] + rp[ok] * (VD[ok] - VC[ok]))))
    kf = kb(rho_f); seenf = set(); x0 = S.wrap(x0); seenf.add(kf(x0, -1)); nf = 1; pq = []; cnt = 0
    def push(p, g, ls, ns):
        nonlocal cnt
        for s in range(S.L):
            e = S.wrap(S.flow(p, s, tau)); n2 = ns + (1 if (ls >= 0 and s != ls) else 0)
            if S.ok(e): cnt += 1; heapq.heappush(pq, (sw_w * n2 + g + tau, cnt, s, e, g + tau, n2, p))
    for s in range(S.L): scan(x0, s, 0.0)
    push(x0, 0.0, -1, 0)
    while pq and nf < NF:
        _, _, s, e, g, n2, par = heapq.heappop(pq); k = kf(e, s)
        if k in seenf: continue
        seenf.add(k); nf += 1
        for s2 in range(S.L): scan(e, s2, g)
        push(e, g, s, n2)
    return best, len(B), nf
