"""solve8 (v8, ДИ 2D): граф на узлах клеток v8 → V → запрос V*(x) → агент → T/T_ref и доля дошедших. growN.py не меняется.
Узлы — все узлы G клеток всех слоёв (включая гало-строки); значение V(узел) = время до цели. Рёбра:
  E1 (свой u, точное): (c,k,j) → (c,k+1,j) вдоль потока, цена tau[k+1,j] − tau[k,j] > 0;
  E2 (цель): отрезок до следующего узла столбца входит в круг RHO → точное время входа (аналитика ДИ); узлы внутри круга V = 0;
  E3 (переключение/переход, цена 0): V(узел) ≤ V_u2(x узла), интерполяция по гиперячейке слоя u2 (как HexIdx, полилинейная); ячейки той же клетки исключены (петля на себя);
      при нескольких покрывающих ячейках берём самую «центральную» (max по ячейкам min по координатам min(s, 1−s)), не min по всем.
Проходы Гаусса–Зейделя до сходимости: E3 (Якоби по всем узлам) → E1 вдоль столбцов в обратном порядке строк.
INTERP: 'N' — узлы нормального скелета G (tau свой у узла), 'T' — узлы временного скелета Gt (равные шаги времени, те же траектории), интерполяция полилинейная.
Запуск:  python solve8.py run --seed 0 --layers 1,-1 --out out.json   (env ETOL и др. — как у growN)"""
import os, sys, time, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
ENV = dict(SYS='di', M='3', KF='21', NFAIL='400', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000', GS='0', GLIM='0', GSEED='1', GOALB='0', GOALSHAPE='ball', RHO='.2')     # как live.py (без MAXC: слой растёт до покрытия)
BIG = 1e9


def load_growN():
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
    for k, v in ENV.items(): os.environ.setdefault(k, v)
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from src.algo import growN as g
    return g


# ---------- эталон ----------
def t_point(x0, v0):
    """время быстродействия ДИ до ТОЧКИ (0,0): s = x + v|v|/2; s>0 → u=−1 сначала"""
    s = x0 + v0 * np.abs(v0) / 2; q = np.sqrt(np.maximum(np.where(s > 0, x0, -x0) + v0 * v0 / 2, 0)); return np.where(s > 0, v0, -v0) + 2 * q


def t_ref(P):
    """T_ref до круга: ref_di_disk (worker-1), если есть, иначе формула до точки (грубая замена, верхняя оценка)"""
    try: from src.algo import ref_di_disk as R; return R.T_ref(P, float(os.environ.get('RHO', .2))), 'ref_di_disk'
    except Exception: return t_point(P[:, 0], P[:, 1]), 't_point'


def entry_time(X, u, dt, rho, n=32):
    """первое t ∈ (0, dt] от x0 по u (ДИ аналитика), при котором |(x,v)| ≤ rho; inf, если нет. X (n,2), dt (n,)"""
    ts = np.linspace(0, 1, n + 1)[None, :] * dt[:, None]; x = X[:, :1] + X[:, 1:] * ts + u * ts ** 2 / 2; v = X[:, 1:] + u * ts; q = x ** 2 + v ** 2 - rho ** 2
    inn = q <= 0; first = np.where(inn.any(1), inn.argmax(1), -1); out = np.full(len(X), np.inf); ok = first > 0
    k = np.flatnonzero(ok); lo = ts[k, first[k] - 1].copy(); hi = ts[k, first[k]].copy(); xx, vv = X[k, 0], X[k, 1]
    for _ in range(40):
        mid = (lo + hi) / 2; qm = (xx + vv * mid + u * mid ** 2 / 2) ** 2 + (vv + u * mid) ** 2 - rho ** 2; c = qm <= 0; hi = np.where(c, mid, hi); lo = np.where(c, lo, mid)
    out[k] = hi; return out


# ---------- атлас и граф ----------
class Light:
    """клетка-обёртка для HexIdx: G (nt,M,N), tau (nt,M)"""
    def __init__(s, G, tau, r, hb, nb, nf): s.G, s.tau, s.r, s.hb, s.nb, s.nf, s.m = G, tau, r, hb, nb, nf, G.shape[1]


def _grow(args):
    u, seed = args; g = load_growN(); t = time.perf_counter(); cells, idx = g.build_layer(u, np.random.default_rng(seed)); return u, cells, time.perf_counter() - t


def build_atlas(layers, seed=0, par=True):
    """слои → {u: cells}, времена роста"""
    if par and len(layers) > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(len(layers)) as p: res = p.map(_grow, [(u, seed) for u in layers])
    else: res = [_grow((u, seed)) for u in layers]
    return {u: c for u, c, _ in res}, {u: t for u, _, t in res}


class Graph:
    def __init__(s, g, atlas, interp='N', K=3):
        s.g, s.interp, s.K = g, interp, K; s.us = list(atlas); s.L = len(s.us); N = g.N; assert N == 2 and g.m_ == 1; M = g.M; s.M = M; s.rho = float(g.RHOV[0])
        s.cells = []; s.idx = {}; s.base = {}; P, T, nxt, cost, nodecell, layer, rowid = [], [], [], [], [], [], []; n0 = 0; gc = 0
        for li, u in enumerate(s.us):
            idx = g.HexIdx(); base = []
            for c in atlas[u]:
                if interp == 'T': G = c.Gt; tau = np.repeat(np.asarray(c.tt)[:, None], M, 1); hb, nb, nf = 0, len(c.tt) - 1, 0                  # весь скелет — «ядро» (для query это не важно)
                else: G, tau, hb, nb, nf = c.G, c.tau, c.hb, c.nb, c.nf
                lc = Light(G, tau, c.r, hb, nb, nf); idx.add(lc); nt = len(G); base.append(n0); ids = n0 + np.arange(nt * M).reshape(nt, M)
                P.append(G.reshape(-1, N)); T.append(tau.reshape(-1)); nodecell.append(np.full(nt * M, gc)); layer.append(np.full(nt * M, li)); rowid.append(np.repeat(np.arange(nt), M))
                nx = np.full((nt, M), -1); nx[:-1] = ids[1:]; cs = np.zeros((nt, M)); cs[:-1] = tau[1:] - tau[:-1]; nxt.append(nx.reshape(-1)); cost.append(cs.reshape(-1)); n0 += nt * M; gc += 1
            s.idx[u] = idx; s.base[u] = np.array(base)
        s.P = np.concatenate(P); s.tau = np.concatenate(T); s.nxt = np.concatenate(nxt); s.cost = np.concatenate(cost); s.nodecell = np.concatenate(nodecell); s.layer = np.concatenate(layer); s.rowid = np.concatenate(rowid); s.n = n0
        off = 0; s.gcell = {}
        for u in s.us: s.gcell[u] = off + np.arange(len(atlas[u])); off += len(atlas[u])
        s.layer_counts = {u: (len(atlas[u]), int(sum(len(c.G) * M for c in atlas[u]))) for u in s.us}

    # --- интерполяция слоя u2 в точках Y: V-вершины и веса (лучшая по центральности ячейка) ---
    def locate(s, u2, Y, excl=None):
        """→ (индексы точек, вершины (k,4), веса (k,4)); excl[i] — глобальный номер клетки, ячейки которой для точки i запрещены"""
        N = 2; idx = s.idx[u2]; pi, hid, sc = idx.query(Y)
        if not len(pi): return np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0, int)
        cid = idx.I[hid, 0]; gcid = s.gcell[u2][cid]
        if excl is not None:
            k = gcid != excl[pi]; pi, hid, sc, cid = pi[k], hid[k], sc[k], cid[k]
            if not len(pi): return np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0, int)
        cen = np.minimum(sc, 1 - sc).min(1); o = np.lexsort((-cen, pi)); pi, hid, sc, cid = pi[o], hid[o], sc[o], cid[o]; st = np.r_[True, pi[1:] != pi[:-1]]; rank = np.arange(len(pi)) - np.maximum.accumulate(np.where(st, np.arange(len(pi)), 0))
        k = rank < s.K; pi, hid, sc, cid, rank = pi[k], hid[k], sc[k], cid[k], rank[k]
        row, col = idx.I[hid, 1], idx.I[hid, 2]; b = s.base[u2][cid]; vo = s.g.VOFF; M = s.M
        verts = np.stack([b + (row + o_[0]) * M + (col + o_[1]) for o_ in vo], 1)
        w = np.stack([np.prod(np.where(o_[None, :] == 1, sc, 1 - sc), 1) for o_ in vo], 1); return pi, verts, w, rank

    def evalpairs(s, V, n, pairs):
        """V в точках по лучшей (самой центральной) из ≤K ячеек, у которой все вершины определены; inf — нет такой"""
        out = np.full(n, np.inf); done = np.zeros(n, bool); pi, verts, w, rank = pairs
        for r in range(s.K):
            m = rank == r
            if not m.any(): continue
            p_, v_, w_ = pi[m], verts[m], w[m]; Vv = V[v_]; ok = (Vv < BIG / 2).all(1) & ~done[p_]; out[p_[ok]] = (w_ * Vv).sum(1)[ok]; done[p_[ok]] = True
        return out

    def prepare(s):
        """E2 и E3 заранее (геометрия не меняется между проходами)"""
        t0 = time.perf_counter(); rho = s.rho; V0 = np.full(s.n, BIG); inn = np.linalg.norm(s.P, axis=1) <= rho + 1e-12; V0[inn] = 0.
        for li, u in enumerate(s.us):                                                                               # E2: вход в круг на отрезке до следующего узла своего u
            k = np.flatnonzero((s.layer == li) & (s.nxt >= 0) & ~inn & (s.cost > 1e-12)); te = entry_time(s.P[k], u, s.cost[k], rho); ok = np.isfinite(te); V0[k[ok]] = np.minimum(V0[k[ok]], te[ok])
        s.V0 = V0; s.e3 = []
        for u2 in s.us:
            s.e3.append(s.locate(u2, s.P, excl=s.nodecell))
        s.t_prep = time.perf_counter() - t0
        has = np.flatnonzero((s.nxt >= 0) & (s.cost > 1e-12)); s.e1 = has

    def solve(s, maxit=2000, tol=1e-10, verbose=True):
        t0 = time.perf_counter(); V = s.V0.copy(); passes = 0
        e1 = s.e1; lev = s.rowid[e1]; order = np.argsort(-lev, kind='stable'); e1 = e1[order]; e1n = s.nxt[e1]; e1c = s.cost[e1]; lev = lev[order]
        cuts = np.flatnonzero(np.r_[True, lev[1:] != lev[:-1]]); cuts = np.r_[cuts, len(e1)]
        for passes in range(1, maxit + 1):
            Vo = V.copy()
            for pairs in s.e3: Vi = s.evalpairs(V, s.n, pairs); V[:] = np.minimum(V, np.where(np.isfinite(Vi), Vi, BIG))
            for a, b in zip(cuts[:-1], cuts[1:]):                                                                    # E1 по уровням строк сверху вниз (последняя строка → первая)
                nodes = e1[a:b]; V[nodes] = np.minimum(V[nodes], e1c[a:b] + V[e1n[a:b]])
            d = float(np.abs(np.minimum(V, BIG) - np.minimum(Vo, BIG)).max())
            if d < tol: break
        s.passes = passes; s.t_solve = time.perf_counter() - t0; s.V = V
        if verbose: print('solve8: узлов %d, проходов %d, %.1f с, достижимо %.1f%% узлов' % (s.n, passes, s.t_solve, 100 * (V < BIG / 2).mean()), flush=True)
        return V

    # --- запросы ---
    def Vlayers(s, X):
        """V_u(x) по слоям (n, L); inf — не покрыто / V не определено"""
        out = np.full((len(X), s.L), np.inf); inb = np.linalg.norm(X, axis=1) <= s.rho
        for li, u in enumerate(s.us):
            out[:, li] = s.evalpairs(s.V, len(X), s.locate(u, X))
        out[inb] = 0.; return out

    def Vstar(s, X):
        Vl = s.Vlayers(X); return Vl.min(1), Vl.argmin(1)

    def run_agent(s, X0, tmax, dt=.01, eps=1e-3):
        """агенты параллельно: точный поток ДИ шагом dt, смена слоя при выигрыше > eps. → время дохода (nan — провал), число переключений"""
        n = len(X0); X = X0.copy(); t = np.zeros(n); alive = np.ones(n, bool); T = np.full(n, np.nan); sw = np.zeros(n, int); uvec = np.array(s.us)
        Vl = s.Vlayers(X); cur = Vl.argmin(1); alive &= np.isfinite(Vl.min(1)); inn = np.linalg.norm(X, axis=1) <= s.rho; T[inn] = 0; alive &= ~inn
        step = 0
        while alive.any():
            a = np.flatnonzero(alive); u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; X[a, 0] = x + v * dt + u * dt * dt / 2; X[a, 1] = v + u * dt; t[a] += dt; step += 1
            hit = np.linalg.norm(X[a], axis=1) <= s.rho; T[a[hit]] = t[a[hit]]; alive[a[hit]] = False; alive[a[t[a] > tmax[a]]] = False; a = np.flatnonzero(alive)
            if not len(a): break
            Vl = s.Vlayers(X[a]); best = Vl.argmin(1); vc = Vl[np.arange(len(a)), cur[a]]; vb = Vl[np.arange(len(a)), best]; chg = (vb < vc - eps) | ~np.isfinite(vc) & np.isfinite(vb)
            sw[a[chg]] += 1; cur[a[chg]] = best[chg]; lost = ~np.isfinite(vb); alive[a[lost]] = False
        return T, sw


def make_graph(g, atlas, interp='N'):
    G = Graph(g, atlas, interp); G.prepare(); return G


# ---------- сценарий ----------
def starts(g, n, seed):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        p = rng.uniform(-g.XLV, g.XLV)
        if np.linalg.norm(p) > float(g.RHOV[0]): out.append(p)
    return np.array(out)


def scenario(a):
    g = load_growN(); layers = [float(x) for x in a.layers.split(',')]; t0 = time.perf_counter()
    atlas, tg = build_atlas(layers, a.seed, par=not a.serial); t_grow = time.perf_counter() - t0
    G = make_graph(g, atlas, a.interp); G.solve(); X0 = starts(g, a.nstarts, 1); Tr, src = t_ref(X0)
    t1 = time.perf_counter(); Vs, us = G.Vstar(X0); t_q = (time.perf_counter() - t1) / len(X0) * 1e3
    t1 = time.perf_counter(); T, sw = G.run_agent(X0, 3 * Tr + 2); t_a = (time.perf_counter() - t1) / len(X0) * 1e3
    ok = np.isfinite(T); ratio = T[ok] / Tr[ok]; vr = Vs[np.isfinite(Vs)] / Tr[np.isfinite(Vs)]; q = lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan')
    res = dict(layers=layers, seed=a.seed, interp=a.interp, etol=float(os.environ.get('ETOL', .01)), ref=src, n=len(X0), success=float(ok.mean()), T_over_ref=dict(mean=float(ratio.mean()) if ok.any() else None, median=q(ratio, 50), p90=q(ratio, 90), max=float(ratio.max()) if ok.any() else None),
               V_over_ref=dict(median=q(vr, 50), p90=q(vr, 90), max=float(vr.max()) if len(vr) else None, covered=float(np.isfinite(Vs).mean())), cells_nodes={str(u): G.layer_counts[u] for u in G.us}, t_grow={str(u): tg[u] for u in tg}, t_grow_wall=t_grow,
               t_prep=G.t_prep, t_solve=G.t_solve, passes=G.passes, ms_per_Vstar=t_q, ms_per_run=t_a, switches_mean=float(sw.mean()))
    worst = np.argsort(-np.where(ok, T / Tr, np.inf))[:5]; res['worst'] = [dict(x=X0[i].tolist(), T=None if not ok[i] else float(T[i]), Tref=float(Tr[i]), Vstar=float(Vs[i]) if np.isfinite(Vs[i]) else None, layer=float(G.us[us[i]]), switches=int(sw[i])) for i in worst]
    print(json.dumps(res, indent=1, ensure_ascii=False));
    if a.out: json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('run',)); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--layers', default='1,-1'); ap.add_argument('--interp', default='N')
    ap.add_argument('--nstarts', type=int, default=200); ap.add_argument('--out', default=''); ap.add_argument('--serial', action='store_true'); a = ap.parse_args(); scenario(a)
