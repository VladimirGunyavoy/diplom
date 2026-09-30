"""H1 в nD (research/docking_nd.md, упрощение «стык с гало + проигрыш»): дерево цепочек с отсечением по ядрам от старта (вперёд) и от цели (назад),
стык — ПРОИГРЫШ: из прямой споры p повторяем последовательность слоёв ближайших обратных спор b (их путь до цели) и проверяем ВХОД В ОКНО ЦЕЛИ.
Время g(p)+t_вход — реальная траектория ⇒ верхняя оценка без интерполяционной ошибки (честность даёт проигрыш, не геометрия стыка). Размерность любая.
SysN: flow(P, s, t) (P (...,d), t скаляр, t<0 назад), L слоёв, scale (d,), per (d,) периоды (0 — нет), in_goal(P), goal_pts (затравки обратного дерева, обычно центр окна)."""
import heapq
import numpy as np
from scipy.spatial import cKDTree


class SysN:
    def __init__(self, flow, L, scale, in_goal, goal_pts, per=None, ok=None, blocked=None):
        self.flow, self.L, self.scale = flow, L, np.asarray(scale, float); self.d = len(self.scale)
        self.per = np.zeros(self.d) if per is None else np.asarray(per, float)
        self.blocked = blocked or (lambda P: np.zeros(np.asarray(P).shape[:-1], bool))          # препятствия: P (...,d) → bool
        self.in_goal, self.goal_pts, self.ok = in_goal, np.asarray(goal_pts, float), (ok or (lambda p: True))

    def wrap(self, P):
        P = np.array(P, float)
        for k in np.nonzero(self.per)[0]:
            P[..., k] = (P[..., k] + self.per[k] / 2) % self.per[k] - self.per[k] / 2
        return P


def _tree(S, tau, starts, N, rho, sw_w, fwd, hist_on=True, hfun=None):
    """Дерево цепочек, switch-first. fwd: вперёд (выход слоя за τ) / назад. Возвращает списки: точки, время g, родитель, слой (слой, которым родитель→точка при fwd; точка→родитель при назад)."""
    key = lambda p, s: (s,) + tuple(int(np.floor(p[k] / (rho * S.scale[k]))) for k in range(S.d))
    seen = set(); P = []; G = []; par = []; lay = []; pq = []; cnt = 0
    def add(p, s, g, pa, ls):
        k = key(p, s)
        if k in seen: return -1
        seen.add(k); P.append(p); G.append(g); par.append(pa); lay.append(ls); return len(P) - 1
    def push(i, ns, ls):
        nonlocal cnt
        ch = []
        for s in range(S.L):
            e = S.wrap(S.flow(P[i], s, tau if fwd else -tau))
            if not S.ok(e) or S.blocked(e) or S.blocked(S.flow(P[i], s, (tau if fwd else -tau) / 2)): continue      # ребро и середина дуги свободны
            ch.append((s, e))
        hh = hfun(np.array([e for _, e in ch])) if (hfun is not None and ch) else [0.0] * len(ch)      # A*: эвристика «до обратного дерева» (hub-research-4)
        for (s, e), hv in zip(ch, hh):
            n2 = ns + (1 if (ls >= 0 and s != ls) else 0); cnt += 1
            heapq.heappush(pq, (sw_w * n2 + G[i] + tau + hv, cnt, s, e, G[i] + tau, n2, i))
    for st in starts:
        i = add(S.wrap(st), -1, 0.0, -1, -1); push(i, 0, -1)
    while pq and len(P) < N:
        _, _, s, e, g, n2, pa = heapq.heappop(pq)
        i = add(e, s, g, pa, s)
        if i >= 0: push(i, n2, s)
    return P, G, par, lay


def replay_value(S, tau, x0, NB, NF, rho_b, rho_f, sw_w=10.0, kn=8, dt=None, back=None):
    """V(x0) стыком-проигрышем. Возвращает (V, nb, nf, путь). back — готовое обратное дерево (общее для запросов к одной цели)."""
    dt = tau / 2 if dt is None else dt; m = int(round(tau / dt))
    if back is None:
        back = build_back(S, tau, NB, rho_b, sw_w)
    Pb, Gb, parb, layb, seqs, tree = back
    Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho_f, sw_w, True)
    best = np.inf; bpath = None
    for i in range(len(Pf)):
        if S.in_goal(Pf[i]) and Gf[i] < best:
            best = Gf[i]; bpath = _fpath(Pf, parf, layf, i, tau); continue
        # ближайшие обратные споры (с периодическими образами)
        q = Pf[i]; d, j = tree_query(S, tree, Pb, q, kn)
        X = np.array([q] * len(j)); alive = np.ones(len(j), bool); tin = np.full(len(j), np.inf)
        L = max(len(seqs[b]) for b in j)
        t = 0.0
        for step in range(L):
            for sub in range(m):
                act = alive & np.array([step < len(seqs[b]) for b in j])
                if not act.any(): break
                for s in range(S.L):
                    sel = act & np.array([step < len(seqs[b]) and seqs[b][step] == s for b in j])
                    if sel.any(): X[sel] = S.wrap(S.flow(X[sel], s, dt))
                alive &= ~S.blocked(X)                                    # проигрыш не проходит сквозь препятствие
                th = Gf[i] + t + (sub + 1) * dt if False else None
                ing = S.in_goal(X) & act
                tt = Gf[i] + step * tau + (sub + 1) * dt
                new = ing & np.isinf(tin); tin[new] = tt; alive &= ~ing
            if not alive.any(): break
        k = int(np.argmin(tin))
        if tin[k] < best:
            best = float(tin[k]); bpath = _fpath(Pf, parf, layf, i, tau) + [(s, tau) for s in seqs[j[k]]]
            # путь проигрыша с точностью до входа в окно (последний шаг может быть частичным — время уже учтено в best)
    return best, len(Pb), len(Pf), bpath


def _fpath(P, par, lay, i, tau):
    out = []
    while par[i] >= 0: out.append((lay[i], tau)); i = par[i]
    return out[::-1]


def build_back(S, tau, NB, rho_b, sw_w=10.0):
    """Обратное дерево цели: точки, время до цели, родитель, слой + последовательности слоёв «точка → цель» и KD-дерево (с периодическими образами)."""
    Pb, Gb, parb, layb = _tree(S, tau, S.goal_pts, NB, rho_b, sw_w, False)
    seqs = []
    for i in range(len(Pb)):
        seq = []; j = i
        while parb[j] >= 0: seq.append(layb[j]); j = parb[j]     # слой лежит на ребре точка→родитель (обратный шаг слоем s)
        seqs.append(seq)
    Z = _images(S, np.array(Pb)); tree = cKDTree(Z / np.tile(S.scale, (len(Z) // len(Pb),)) if False else _scaled(S, Z))
    return Pb, Gb, parb, layb, seqs, tree


def _images(S, P):
    """P и его периодические образы (3^k копий); первые len(P) строк — сами P."""
    out = [P]
    for k in np.nonzero(S.per)[0]:
        cur = []
        for base in out:
            for sh in (+1, -1):
                Q = base.copy(); Q[:, k] += sh * S.per[k]; cur.append(Q)
        out = out + cur
    return np.vstack(out)


def _scaled(S, Z): return Z / S.scale


def back_heuristic(S, back, wh=3.0, kap=1.0):
    """h(E) = wh·(Gb[ближайшая обратная спора] + kap·|E−b|_scaled) для A*-приоритета прямого дерева (hub-research-4: 6D NF800 2/4 → 3/4)."""
    Pb, Gb, tree = np.asarray(back[0]), np.asarray(back[1]), back[5]
    def h(E):
        d, j = tree.query(np.atleast_2d(E) / S.scale, k=1); return wh * (Gb[np.asarray(j) % len(Pb)] + kap * d)
    return h


def tree_query(S, tree, Pb, q, kn):
    d, j = tree.query(q / S.scale, k=min(kn * (1 + 2 * int((S.per > 0).sum())), len(tree.data)))
    j = np.asarray(j) % len(Pb); u, first = np.unique(j, return_index=True)
    order = np.sort(first)[:kn]; return d, j[order]


def replay_value_fast(S, tau, x0, NB, NF, rho_b, rho_f, sw_w=10.0, kn=8, dt=None, back=None):
    """То же, что replay_value, но проигрыш всех пар (прямая спора × ближайшая обратная) одним батчем по слоям (без Python-цикла по спорам)."""
    dt = tau / 2 if dt is None else dt; m = int(round(tau / dt))
    if back is None: back = build_back(S, tau, NB, rho_b, sw_w)
    Pb, Gb, parb, layb, seqs, tree = back
    Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho_f, sw_w, True)
    best = np.inf; bpath = None; bi = -1; pi_, pb_ = [], []
    for i in range(len(Pf)):
        if S.in_goal(Pf[i]):
            if Gf[i] < best: best = Gf[i]; bi = i; bpath = _fpath(Pf, parf, layf, i, tau)
            continue
        _, j = tree_query(S, tree, Pb, Pf[i], kn); pi_ += [i] * len(j); pb_ += list(j)
    if pi_:
        pi_ = np.array(pi_); pb_ = np.array(pb_); X = np.array([Pf[i] for i in pi_]); G0 = np.array([Gf[i] for i in pi_])
        ln = np.array([len(seqs[b]) for b in pb_]); alive = np.ones(len(pi_), bool); tin = np.full(len(pi_), np.inf)
        S_mat = np.full((len(pi_), max(ln.max(), 1)), -1, int)
        for r, b in enumerate(pb_): S_mat[r, :ln[r]] = seqs[b]
        for step in range(S_mat.shape[1]):
            act = alive & (step < ln)
            if not act.any(): break
            for sub in range(m):
                act = alive & (step < ln)
                if not act.any(): break
                for s in range(S.L):
                    sel = act & (S_mat[:, step] == s)
                    if sel.any(): X[sel] = S.wrap(S.flow(X[sel], s, dt))
                alive &= ~S.blocked(X)
                ing = S.in_goal(X) & act
                new = ing & np.isinf(tin); tin[new] = G0[new] + step * tau + (sub + 1) * dt; alive &= ~ing
        k = int(np.argmin(tin))
        if tin[k] < best:
            best = float(tin[k]); bpath = _fpath(Pf, parf, layf, pi_[k], tau) + [(s, tau) for s in seqs[pb_[k]]]
    return best, len(Pb), len(Pf), bpath
