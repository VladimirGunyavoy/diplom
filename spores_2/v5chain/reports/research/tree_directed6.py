"""research hub-research-4: прямое дерево с эвристикой из обратного (A*: приоритет sw_w·n + g + wh·h) против нынешнего switch-first Дейкстры.
Система — 6D манипулятор 3 звена + g=.3, цель вниз, DT=.02 (как tests/check_corridor_manip6.py G=.3 DOWN=1): там q1/q2 решались лишь при NB12000 NF3000.
Метрики без SLSQP: мин. расстояние прямое→обратное (как tests/diag_cover_6d.py), число попаданий candidates(), лучший промах, гистограмма переключений.
h(e) = min по ближайшей обратной споре b (KD, scaled): Gb[b] + kap·|e−b|_scaled. Запуск из v6: python3 tree_directed6.py NB NF [MODE...]
MODE: base | a<wh>_<kap> (например a1_1) — прямое дерево A*; CORR=1 — ещё и полный corridor_query (SLSQP) для проверки валидности."""
import sys, os, time, heapq; sys.path.insert(0, '.')
import numpy as np
import src.atlas6.corridor_nd as C
import src.atlas6.adaptive_nd as A
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.manip3dyn import flow
NB, NF = int(sys.argv[1]), int(sys.argv[2]); MODES = sys.argv[3:] or ['base', 'a1_1']
tau = 0.4; Rq = 0.3; Rw = 0.6; WM = 3.0; rho = 0.15; GG = float(os.environ.get('G', 0.3)); DT = float(os.environ.get('DT', 0.02))
C3 = np.array([-np.pi / 2 if os.environ.get('DOWN', '1') == '1' else 0.0, 0, 0, 0, 0, 0])
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
fl = lambda P, s, t: flow(P, s, t, dt_max=DT, g=GG)
ing = lambda P: (np.max(np.abs(wr(P[..., :3] - C3[:3])), -1) < Rq) & (np.max(np.abs(P[..., 3:]), -1) < Rw)
g = lambda x: np.array([Rq ** 2 - wr(x[0] - C3[0]) ** 2, Rq ** 2 - wr(x[1]) ** 2, Rq ** 2 - wr(x[2]) ** 2, Rw ** 2 - x[3] ** 2, Rw ** 2 - x[4] ** 2, Rw ** 2 - x[5] ** 2])
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :3] - C3[:3])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., 3:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C3 + v * np.array([Rq, Rq, Rq, Rw, Rw, Rw]))
S = SysN(fl, 8, (np.pi, np.pi, np.pi, WM, WM, WM), ing, seeds, per=(2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= WM)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)]) for _ in range(4)]
t0 = time.time(); back = build_back(S, tau, NB, rho); Pb = np.asarray(back[0]); Gb = np.asarray(back[1]); KD = back[5]
print('NB', len(Pb), 'build_back %.0f с' % (time.time() - t0), flush=True)


def nsw(par, lay):
    """число переключений слоя на пути от корня до каждого узла."""
    n = np.zeros(len(par), int)
    for i in range(len(par)):
        p = par[i]
        if p >= 0: n[i] = n[p] + (1 if (lay[p] >= 0 and lay[p] != lay[i]) else 0)
    return n


print('обратное: переключений', np.bincount(nsw(back[2], back[3])), flush=True)


def h_back(E, kap):
    d, j = KD.query(np.atleast_2d(E) / S.scale, k=1); return Gb[np.asarray(j) % len(Pb)] + kap * d


def dtree(S_, tau_, starts, N, rho_, sw_w, fwd, hist_on=True, wh=1.0, kap=1.0):
    """копия adaptive_nd._tree (вперёд) с A*-приоритетом: sw_w·n2 + g + wh·h(e); g в дереве — настоящее время."""
    key = lambda p, s: (s,) + tuple(int(np.floor(p[k] / (rho_ * S_.scale[k]))) for k in range(S_.d))
    seen = set(); P = []; G = []; par = []; lay = []; pq = []; cnt = 0
    def add(p, s, g_, pa, ls):
        k = key(p, s)
        if k in seen: return -1
        seen.add(k); P.append(p); G.append(g_); par.append(pa); lay.append(ls); return len(P) - 1
    def push(i, ns, ls):
        nonlocal cnt
        ch = []
        for s in range(S_.L):
            e = S_.wrap(S_.flow(P[i], s, tau_))
            if not S_.ok(e) or S_.blocked(e): continue
            ch.append((s, e))
        if not ch: return
        hh = wh * h_back(np.array([e for _, e in ch]), kap)
        for (s, e), hv in zip(ch, hh):
            n2 = ns + (1 if (ls >= 0 and s != ls) else 0); cnt += 1
            heapq.heappush(pq, (sw_w * n2 + G[i] + tau_ + hv, cnt, s, e, G[i] + tau_, n2, i))
    for st in starts:
        i = add(S_.wrap(st), -1, 0.0, -1, -1); push(i, 0, -1)
    while pq and len(P) < N:
        _, _, s, e, g_, n2, pa = heapq.heappop(pq)
        i = add(e, s, g_, pa, s)
        if i >= 0: push(i, n2, s)
    return P, G, par, lay


base_tree = A._tree
for mode in MODES:
    if mode == 'base': C._tree = base_tree
    else:
        wh, kap = [float(a) for a in mode[1:].split('_')]; sw = float(os.environ.get('SW', 10.0))
        C._tree = lambda S_, tau_, st, N, r, sw_w, fwd, wh=wh, kap=kap, sw=sw: dtree(S_, tau_, st, N, r, sw, fwd, wh=wh, kap=kap)
    for n, x in enumerate(Q):
        t1 = time.time(); Pf, Gf, parf, layf = C._tree(S, tau, [x], NF, rho, 10.0, True)
        d = KD.query(np.asarray(Pf) / S.scale, k=1)[0]; tb = time.time() - t1
        cand, nh = C.candidates(S, x, back, tau, NF, rho, miss)
        best_miss = min((c[1] for c in cand if c[0] == 1), default=np.nan); best_hit = min((c[1] for c in cand if c[0] == 0), default=np.nan)
        line = '%s q%d прямых %d (%.0f с) мин.расст %.3f медиана %.2f | попаданий %d лучшее t %.2f лучший промах-score %.2f | перекл. %s' % (
            mode, n + 1, len(Pf), tb, d.min(), np.median(d), nh, best_hit, best_miss, np.bincount(nsw(parf, layf)))
        if os.environ.get('CORR'):
            t2 = time.time(); b = C.corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=5, tries=3); xe = None
            if b is not None:
                xe = np.array(x, float)
                for s_, dd in zip(b[1], b[2]): xe = flow(xe, s_, dd, dt_max=0.005, g=GG)
            line += ' | T_corr %s валид %s (%.0f с)' % (None if b is None else round(b[0], 2), None if b is None else bool(np.all(g(xe) > -1e-4)), time.time() - t2)
        print(line, flush=True)
