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
ENV = dict(SYS='di', M='3', XL='3', VL='2.5', GMX='3', KF='21', NFAIL='400', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000', GS='0', GLIM='0', GSEED='1', GOALB='0', GOALSHAPE='ball', RHO='.2')     # как live.py (без MAXC: слой растёт до покрытия)
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
class Refined:
    """уточнённая клетка-полоса между соседними столбцами j, j+1 родителя: M столбцов (концы — столбцы родителя, внутри — выращенные как обычные столбцы потоком от интерполированной затравки), те же строки; prio выше родителя"""
    def __init__(s, G, tau, r, hb, nb, nf, prio, depth): s.G, s.tau, s.r, s.hb, s.nb, s.nf, s.prio, s.depth = G, tau, r, hb, nb, nf, prio, depth


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


def freeze_idx(g, s):
    """CSR-заморозка HexIdx (searcher-3, lit_v7_port п.2): бины → отсортированные ключи + searchsorted, без питон-цикла по бинам; пары те же, query подменяется на экземпляре. Индекс после этого не пополнять."""
    if not s.bins: return
    ks = list(s.bins); fk = np.array(ks, np.int64); o = np.argsort(fk); fk = fk[o]; L = [np.concatenate(s.bins[ks[i]]) for i in o]; ln = np.array([len(l) for l in L], np.int64)
    s.fk = fk; s.fst = np.r_[0, np.cumsum(ln)]; s.fv = np.concatenate(L); N = g.N
    def query(Y):
        z = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, N)))
        if s.n == 0 or not len(Y): return z
        kk = s._key(np.floor(Y / g.QB).astype(np.int64)); pos = np.minimum(np.searchsorted(s.fk, kk), len(s.fk) - 1); pts = np.flatnonzero(s.fk[pos] == kk)
        if not len(pts): return z
        st = s.fst[pos[pts]]; cn = s.fst[pos[pts] + 1] - st; tot = int(cn.sum()); pi = np.repeat(pts, cn); ix = np.repeat(st, cn) + np.arange(tot) - np.repeat(np.cumsum(cn) - cn, cn); out = []; CH = 1 << 20
        for a in range(0, tot, CH): out.append(s._test(Y, pi[a:a + CH], s.fv[ix[a:a + CH]]))
        return [np.concatenate(a) for a in zip(*out)]
    s.query = query


class Graph:
    def __init__(s, g, atlas, interp='N', K=3):
        s.mode, s.thr, s.pen = 'lin', .2, 0.; s.xentry = float(os.environ.get('XENTRY', 0)); s.extin = float(os.environ.get('EXTIN', 0)); s.subk = int(os.environ.get('SUBK', 0)); s.spsel = int(os.environ.get('SPSEL', 0)); pl = int(os.environ.get('SOLVE8_PLUS', 0)); s.bis = float(os.environ.get('BISECT', .005 if pl else 0)); s.bisr = int(os.environ.get('BISECT_ROUNDS', 3)); s.bnd = int(os.environ.get('BISECT_BOUND', 1 if pl else 0)); s.refthr = float(os.environ.get('REFINE_THR', .15 if pl else 0)); s.rdepth = int(os.environ.get('REFINE_DEPTH', 2)); s.warm = int(os.environ.get('WARM', 0)); s.wtol = float(os.environ.get('WARM_TOL', 1e-6 if pl else 0)); s._refined = False; s.refined = 0; s._subc = None; s.calc, s.quad, s.Vn = 'v', False, None; s.g, s.interp, s.K = g, interp, K; s.us = list(atlas); s.L = len(s.us); N = g.N; assert N == 2 and g.m_ == 1; M = g.M; s.M = M; s.rho = float(g.RHOV[0])
        s.cells = []; s.idx = {}; s.base = {}; P, T, nxt, cost, nodecell, layer, rowid = [], [], [], [], [], [], []; n0 = 0; gc = 0
        prio = []; hidA = g.HexIdx(); cb = []; cl = []; hid = {u: g.HexIdx() for u in s.us}; bs = {u: [] for u in s.us}; gl = {u: [] for u in s.us}
        order = [(li, u, c) for li, u in enumerate(s.us) for c in atlas[u] if not isinstance(c, Refined)] + [(li, u, c) for li, u in enumerate(s.us) for c in atlas[u] if isinstance(c, Refined)]      # узлы: сначала клетки атласа слой за слоем (как раньше — окно собирает сетку по этому порядку), затем уточнённые полосы
        for li, u, c in order:
            if interp == 'T': G = c.Gt; tau = np.repeat(np.asarray(c.tt)[:, None], M, 1); hb, nb, nf = 0, len(c.tt) - 1, 0                  # весь скелет — «ядро» (для query это не важно)
            else: G, tau, hb, nb, nf = c.G, c.tau, c.hb, c.nb, c.nf
            prio.append(float(getattr(c, 'prio', 0))); lc = Light(G, tau, c.r, hb, nb, nf); hid[u].add(lc); hidA.add(lc); cb.append(n0); cl.append(li); nt = len(G); bs[u].append(n0); gl[u].append(gc); ids = n0 + np.arange(nt * M).reshape(nt, M)
            P.append(G.reshape(-1, N)); T.append(tau.reshape(-1)); nodecell.append(np.full(nt * M, gc)); layer.append(np.full(nt * M, li)); rowid.append(np.repeat(np.arange(nt), M))
            nx = np.full((nt, M), -1); nx[:-1] = ids[1:]; cs = np.zeros((nt, M)); cs[:-1] = tau[1:] - tau[:-1]; nxt.append(nx.reshape(-1)); cost.append(cs.reshape(-1)); n0 += nt * M; gc += 1
        for u in s.us: s.idx[u] = hid[u]; s.base[u] = np.array(bs[u])
        s.idxA = hidA; s.cbase = np.array(cb); s.clayer = np.array(cl)
        if int(os.environ.get('CSR', 1)):
            for ix in list(hid.values()) + [hidA]: freeze_idx(g, ix)
        s.prio = np.array(prio); s.P = np.concatenate(P); s.tau = np.concatenate(T); s.nxt = np.concatenate(nxt); s.cost = np.concatenate(cost); s.nodecell = np.concatenate(nodecell); s.layer = np.concatenate(layer); s.rowid = np.concatenate(rowid); s.n = n0
        s.gcell = {u: np.array(gl[u]) for u in s.us}; s._atlas, s._interp = atlas, interp
        s.layer_counts = {u: (len(atlas[u]), int(sum(len(c.G) * M for c in atlas[u]))) for u in s.us}

    # --- интерполяция слоя u2 в точках Y: V-вершины и веса (лучшая по центральности ячейка) ---
    def locate(s, u2, Y, excl=None):
        """→ (индексы точек, вершины (k,4), веса (k,4)); excl[i] — глобальный номер клетки, ячейки которой для точки i запрещены"""
        if s.quad: return s._locate_quad(u2, Y, excl)
        N = 2; idx = s.idx[u2]; pi, hid, sc = idx.query(Y)
        if not len(pi): return np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0, int)
        cid = idx.I[hid, 0]; gcid = s.gcell[u2][cid]
        if excl is not None:
            k = gcid != excl[pi]; pi, hid, sc, cid = pi[k], hid[k], sc[k], cid[k]
            if not len(pi): return np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0, int)
        cen = np.minimum(sc, 1 - sc).min(1) + s.prio[s.gcell[u2][cid]]; o = np.lexsort((-cen, pi)); pi, hid, sc, cid = pi[o], hid[o], sc[o], cid[o]; st = np.r_[True, pi[1:] != pi[:-1]]; rank = np.arange(len(pi)) - np.maximum.accumulate(np.where(st, np.arange(len(pi)), 0))
        k = rank < s.K; pi, hid, sc, cid, rank = pi[k], hid[k], sc[k], cid[k], rank[k]
        row, col = idx.I[hid, 1], idx.I[hid, 2]; b = s.base[u2][cid]; vo = s.g.VOFF; M = s.M
        verts = np.stack([b + (row + o_[0]) * M + (col + o_[1]) for o_ in vo], 1)
        w = np.stack([np.prod(np.where(o_[None, :] == 1, sc, 1 - sc), 1) for o_ in vo], 1); return pi, verts, w, rank

    def locate_all(s, Y):
        """locate сразу для всех слоёв одним запросом к общему индексу (агент: в L раз меньше вызовов) → список по слоям (pi, вершины, веса, ранг), как locate"""
        idx = s.idxA; pi, hid, sc = idx.query(Y); L = s.L; z = (np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0, int))
        if not len(pi): return [z] * L
        cid = idx.I[hid, 0]; lay = s.clayer[cid]; cen = np.minimum(sc, 1 - sc).min(1) + s.prio[cid]; grp = pi * L + lay; o = np.lexsort((-cen, grp)); pi, hid, sc, cid, lay, grp = pi[o], hid[o], sc[o], cid[o], lay[o], grp[o]
        st = np.r_[True, grp[1:] != grp[:-1]]; rank = np.arange(len(pi)) - np.maximum.accumulate(np.where(st, np.arange(len(pi)), 0)); k = rank < s.K; pi, hid, sc, cid, lay, rank = pi[k], hid[k], sc[k], cid[k], lay[k], rank[k]
        row, col = idx.I[hid, 1], idx.I[hid, 2]; b = s.cbase[cid]; M = s.M; verts = np.stack([b + (row + o_[0]) * M + (col + o_[1]) for o_ in s.g.VOFF], 1); w = np.stack([np.prod(np.where(o_[None, :] == 1, sc, 1 - sc), 1) for o_ in s.g.VOFF], 1); res = []
        for li in range(L): m = lay == li; res.append((pi[m], verts[m], w[m], rank[m]))
        return res

    def _locate_quad(s, u2, Y, excl=None):
        """calc='qquad' (только M=3): вершины 2 строки × 3 столбца (вся ширина клетки), веса — линейно по времени × Лагранж 2-й степени поперёк потока (прототип lr3/qsolve2.patch_quad)"""
        idx = s.idx[u2]; pi, hid, sc = idx.query(Y); z = (np.zeros(0, int), np.zeros((0, 6), int), np.zeros((0, 6)), np.zeros(0, int))
        if not len(pi): return z
        cid = idx.I[hid, 0]; gcid = s.gcell[u2][cid]
        if excl is not None:
            k = gcid != excl[pi]; pi, hid, sc, cid = pi[k], hid[k], sc[k], cid[k]
            if not len(pi): return z
        cen = np.minimum(sc, 1 - sc).min(1); o = np.lexsort((-cen, pi)); pi, hid, sc, cid = pi[o], hid[o], sc[o], cid[o]; st = np.r_[True, pi[1:] != pi[:-1]]; rank = np.arange(len(pi)) - np.maximum.accumulate(np.where(st, np.arange(len(pi)), 0))
        k = rank < s.K; pi, hid, sc, cid, rank = pi[k], hid[k], sc[k], cid[k], rank[k]; row, col = idx.I[hid, 1], idx.I[hid, 2]; b = s.base[u2][cid]
        xi = col + sc[:, 1]; L = np.stack([(xi - 1) * (xi - 2) / 2, -xi * (xi - 2), xi * (xi - 1) / 2], 1); wt = np.stack([1 - sc[:, 0], sc[:, 0]], 1)
        verts = np.stack([b + (row + r) * 3 + c for r in (0, 1) for c in (0, 1, 2)], 1); w = np.stack([wt[:, r] * L[:, c] for r in (0, 1) for c in (0, 1, 2)], 1); return pi, verts, w, rank

    def evalpairs(s, V, n, pairs):
        """V в точках по лучшей (самой центральной) из ≤K ячеек, у которой все вершины определены; inf — нет такой"""
        out = np.full(n, np.inf); done = np.zeros(n, bool); pi, verts, w, rank = pairs
        if s.spsel and s.mode == 'lin':                                                                      # SPSEL: из перекрывающихся ячеек берём ту, у которой меньше разброс значений по вершинам (разброс = ячейка режет излом/дыру), а не самую центральную
            Vv = V[verts]; fin = (Vv < BIG / 2).all(1); sp = np.where(fin, Vv.max(1) - Vv.min(1), np.inf); val = (w * Vv).sum(1)
            o = np.lexsort((rank, sp, pi)); pi_, val_, sp_ = pi[o], val[o], sp[o]; first = np.r_[True, pi_[1:] != pi_[:-1]]; ok = first & np.isfinite(sp_); out[pi_[ok]] = val_[ok]; return out
        for r in range(s.K):
            m = rank == r
            if not m.any(): continue
            p_, v_, w_ = pi[m], verts[m], w[m]; Vv = V[v_]; fin = Vv < BIG / 2
            if s.mode == 'flow':                                                                             # интерполяция только вдоль потока: линейно по строкам, ближайший столбец (вершины: порядок (0,0),(0,1),(1,0),(1,1) = (строка, столбец))
                wr1 = w_[:, 2] + w_[:, 3]; c1 = (w_[:, 1] + w_[:, 3]) > .5; w2 = np.stack([(1 - wr1) * ~c1, (1 - wr1) * c1, wr1 * ~c1, wr1 * c1], 1); ok = (fin | (w2 <= 0)).all(1) & ~done[p_]; val = (w2 * np.where(w2 > 0, Vv, 0)).sum(1)
            elif s.mode == 'maxv':                                                                           # большой разброс V по вершинам (излом) → оценка сверху: max по вершинам с весом > .05
                ok = fin.all(1) & ~done[p_]; spread = Vv.max(1) - Vv.min(1); val = np.where(spread > s.thr, np.where(w_ > .05, Vv, -np.inf).max(1), (w_ * Vv).sum(1))
            else: ok = fin.all(1) & ~done[p_]; val = (w_ * Vv).sum(1)
            out[p_[ok]] = val[ok]; done[p_[ok]] = True
        return out

    def prepare(s):
        """E2 и E3 заранее (геометрия не меняется между проходами)"""
        t0 = time.perf_counter(); rho = s.rho; V0 = np.full(s.n, BIG); inn = np.linalg.norm(s.P, axis=1) <= rho + 1e-12; V0[inn] = 0.
        for li, u in enumerate(s.us):                                                                               # E2: вход в круг на отрезке до следующего узла своего u
            k = np.flatnonzero((s.layer == li) & (s.nxt >= 0) & ~inn & (s.cost > 1e-12)); te = entry_time(s.P[k], u, s.cost[k], rho); ok = np.isfinite(te); V0[k[ok]] = np.minimum(V0[k[ok]], te[ok])
        if s.extin:                                                                                                 # W2: внутри круга V продолжается отрицательным (−время после входа по потоку столбца) — интерполяция через излом на границе круга не завышает
            has = np.flatnonzero((s.nxt >= 0) & (s.cost > 1e-12)); has = has[np.argsort(s.rowid[has], kind='stable')]
            for k in has:
                n_ = s.nxt[k]
                if inn[n_] and V0[k] < BIG / 2: V0[n_] = max(V0[k] - s.cost[k], -s.extin)
        s.V0 = V0; s.e3 = []; s._subc = None
        for u2 in s.us:
            s.e3.append(s.locate(u2, s.P, excl=s.nodecell))
        s.t_prep = time.perf_counter() - t0
        has = np.flatnonzero((s.nxt >= 0) & (s.cost > 1e-12)); s.e1 = has

    def set_calc(s, calc):
        """расчёт V: 'v' (по умолчанию, как было) | 'q' — переходы по Q (цена «продолжать своё u», минимум ПОСЛЕ интерполяции) | 'qquad' — то же + квадратичная интерполяция поперёк потока (только M=3). Обоснование: knowledge/research/v_kink_q.md (v5chain)"""
        assert calc in ('v', 'q', 'qquad'), calc
        if calc == 'qquad': assert s.M == 3, 'qquad только для M=3'
        s.calc = calc
        if (calc == 'qquad') != s.quad: s.quad = calc == 'qquad'; s.prepare()                                    # геометрия переходов (e3) другая

    def _xentry_init(s):
        """W2 (laptop-v5chain-worker-6): точный вход в цель по прямому потоку на горизонте xentry — верхняя оценка, не зависящая от грубой сетки у цели. Q0: свой слой (продолжать своё u до входа); V0: любой слой (переключиться сразу и идти прямо)"""
        H = np.full(s.n, s.xentry); te = np.stack([entry_time(s.P, u, H, s.rho) for u in s.us], 1); Q0 = np.minimum(s.V0, np.where(np.isfinite(te[np.arange(s.n), s.layer]), te[np.arange(s.n), s.layer], BIG)); m = te.min(1)
        return Q0, np.minimum(Q0, np.where(np.isfinite(m), m, BIG))

    def solve(s, maxit=2000, tol=1e-10, verbose=True, progress=None):
        """V (calc='v') или Q (calc='q'/'qquad': s.V = Q — запросы Vlayers/Vstar/агент идут по Q; s.Vn — V узлов = min(Q, Q чужих слоёв))"""
        if s.calc != 'v' or s.quad: s.set_calc(s.calc)
        if s.calc == 'q' and s.refthr > 0 and not s._refined:                                                # SOLVE8_PLUS / REFINE_THR: адаптивные полосы по скачку Q (refine_atlas), затем бисекция переключений; граф усыновляет уточнённый
            _, G2, ins = refine_atlas(s.g, s._atlas, s.refthr, s.rdepth, 'q', verbose=False); G2._refined = True; G2.refined = ins; keep = (s.bis, s.bisr, s.bnd, s.refthr, s.rdepth); s.__dict__.update(G2.__dict__); s.bis, s.bisr, s.bnd, s.refthr, s.rdepth = keep; s._refined = True; s.refined = ins
        if s.calc != 'v':
            V = s._solve_q(maxit, max(tol, s.wtol), verbose, progress=progress)
            if s.bis > 0 and not s.subk:                                                                      # BISECT: подточки — моменты переключения, найденные бисекцией по времени на рёбрах, где «продолжать» и «переключиться» меняются местами
                for _ in range(s.bisr): s._bisect_pass(); V = s._solve_q(maxit, max(tol, s.wtol), False, progress=progress, warm=bool(s.warm))
                s.subk = 0; s._subc = None
            return V
        t0 = time.perf_counter(); V = s.V0.copy(); passes = 0
        e1 = s.e1; lev = s.rowid[e1]; order = np.argsort(-lev, kind='stable'); e1 = e1[order]; e1n = s.nxt[e1]; e1c = s.cost[e1]; lev = lev[order]
        cuts = np.flatnonzero(np.r_[True, lev[1:] != lev[:-1]]); cuts = np.r_[cuts, len(e1)]
        for passes in range(1, maxit + 1):
            Vo = V.copy()
            for pairs in s.e3: Vi = s.evalpairs(V, s.n, pairs); V[:] = np.minimum(V, np.where(np.isfinite(Vi), Vi + s.pen, BIG))
            for a, b in zip(cuts[:-1], cuts[1:]):                                                                    # E1 по уровням строк сверху вниз (последняя строка → первая)
                nodes = e1[a:b]; V[nodes] = np.minimum(V[nodes], e1c[a:b] + V[e1n[a:b]])
            d = float(np.abs(np.minimum(V, BIG) - np.minimum(Vo, BIG)).max())
            if progress is not None: progress(passes, d, V)      # progress(проход, невязка, V) — для строки прогресса
            if d < tol: break
        s.passes = passes; s.converged = d < tol; s.t_solve = time.perf_counter() - t0; s.V = V; s.Vn = V
        if verbose: print('solve8: узлов %d, проходов %d, %.1f с, достижимо %.1f%% узлов' % (s.n, passes, s.t_solve, 100 * (V < BIG / 2).mean()), flush=True)
        return V

    def _bisect_pass(s):
        """по текущему Q: на ребре k→k+1 своего слоя и чужом u2 стоимость переключения в момент t: g(t) = t + Q_u2(x(t)), точный поток; минимум по t ∈ [0, h] — бисекция по знаку g(t+δ) − g(t) до bis (δ = bis); подточка в найденном t, если g(t) < min(g(0), g(h)) − 1e-4. Подточки → кэш _sub, s.subk = −1"""
        Q = s.V; n = s.e1; h = s.cost[n]; us = np.array(s.us); u = us[s.layer[n]]
        def X(idx, t, uu): x, v = s.P[idx, 0], s.P[idx, 1]; return np.stack([x + v * t + uu * t * t / 2, v + uu * t], 1)
        def Q2(li2, Y): return np.where(np.isfinite(o := s.evalpairs(Q, len(Y), s.locate(s.us[li2], Y))), o, BIG)
        own, tt, PP = [], [], []; d = s.bis
        for li2 in range(s.L):
            m = np.flatnonzero((s.layer[n] != li2) & (s.nxt[n] >= 0))
            if not len(m): continue
            idx = n[m]; hh = h[m]; uu = u[m]; g0 = Q2(li2, s.P[idx]); gh = hh + Q2(li2, X(idx, hh, uu)); ok = (g0 < BIG / 2) & (gh < BIG / 2) & (hh > 2 * d)
            if not ok.any(): continue
            idx, hh, uu, g0, gh = idx[ok], hh[ok], uu[ok], g0[ok], gh[ok]; lo = np.zeros(len(idx)); hi = hh.copy()
            for _ in range(int(np.ceil(np.log2(max(float(hh.max()), 1e-9) / d))) + 1):
                mid = (lo + hi) / 2; a_ = mid + Q2(li2, X(idx, mid, uu)); b_ = mid + d + Q2(li2, X(idx, np.minimum(mid + d, hh), uu)); right = b_ < a_; lo = np.where(right, mid, lo); hi = np.where(right, hi, mid)
            t = (lo + hi) / 2; gt = t + Q2(li2, X(idx, t, uu)); good = gt < np.minimum(g0, gh) - 1e-4
            if good.any(): own.append(idx[good]); tt.append(t[good]); PP.append(X(idx[good], t[good], uu[good]))
        if s.bnd:                                                                                            # граница клетки чужого слоя: узел-клон ровно по обе стороны границы (t_b−ε и t_b+ε), t_b — бисекция по времени вдоль столбца (слово пользователя)
            def cid(li2, Y):
                pi, verts, w, rank = s.locate(s.us[li2], Y); out = np.full(len(Y), -1); r0 = rank == 0; out[pi[r0]] = s.nodecell[verts[r0, 0]]; return out
            for li2 in range(s.L):
                m = np.flatnonzero((s.layer[n] != li2) & (s.nxt[n] >= 0)); idn = cid(li2, s.P); idx = n[m]; c0 = idn[idx]; c1 = idn[s.nxt[idx]]; sel = c0 != c1
                if not sel.any(): continue
                idx = idx[sel]; hh = h[m][sel]; uu = u[m][sel]; c0 = c0[sel]; lo = np.zeros(len(idx)); hi = hh.copy()
                for _ in range(int(np.ceil(np.log2(max(float(hh.max()), 1e-9) / (s.bis / 8)))) + 1):
                    mid = (lo + hi) / 2; same = cid(li2, X(idx, mid, uu)) == c0; lo = np.where(same, mid, lo); hi = np.where(same, hi, mid)
                eps = s.bis / 8
                for tb in (np.maximum(lo - 0., 0.), np.minimum(hi + eps, hh)): own.append(idx); tt.append(tb); PP.append(X(idx, tb, uu))
        if not own: s.subk = -1; s._subc = (-1, (np.zeros(0, int), np.zeros(0), [(np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0, int)) for _ in s.us], 0)); return 0
        P = np.concatenate(PP); r = (np.concatenate(own), np.concatenate(tt), [s.locate(u2, P) for u2 in s.us], len(P)); s.subk = -1; s._subc = (-1, r); return len(P)

    def _sub(s):
        """подточки рёбер E1 (K на ребро, равные доли шага по времени; точный поток ДИ): (узел-владелец, время t_j, пары locate по слоям, число точек); кэш по K"""
        if s._subc is not None and s._subc[0] == s.subk: return s._subc[1]
        n = s.e1; c = s.cost[n]; u = np.array(s.us)[s.layer[n]]; own, tt, PP = [], [], []
        for j in range(1, s.subk + 1):
            t = c * j / (s.subk + 1); x, v = s.P[n, 0], s.P[n, 1]; PP.append(np.stack([x + v * t + u * t * t / 2, v + u * t], 1)); own.append(n); tt.append(t)
        P = np.concatenate(PP); r = (np.concatenate(own), np.concatenate(tt), [s.locate(u2, P) for u2 in s.us], len(P)); s._subc = (s.subk, r); return r

    def _solve_q(s, maxit=3000, tol=1e-10, verbose=True, pen=None, progress=None, warm=False):
        """Q_u(узел) = вход в цель / шаг вдоль потока + V(следующего) / стык с клеткой СВОЕГО слоя (интерполяция Q); V(узел) = min(Q, интерполяция Q других слоёв) (прототип lr3/qsolve.solve_q)"""
        pen = s.pen if pen is None else pen; t0 = time.perf_counter(); Q0, V0 = s._xentry_init() if s.xentry > 0 else (s.V0, s.V0); V = V0.copy(); Q = Q0.copy()
        if warm and getattr(s, 'V', None) is not None and getattr(s, 'Vn', None) is not None and len(s.V) == s.n: Q = np.minimum(Q, s.V); V = np.minimum(V, s.Vn)                  # тёплый старт: Q/V прошлого решения — верхняя оценка (подточки бисекции только добавляют опции)
        e1 = s.e1; lev = s.rowid[e1]; o = np.argsort(-lev, kind='stable'); e1 = e1[o]; e1n = s.nxt[e1]; e1c = s.cost[e1]; lev = lev[o]
        cuts = np.flatnonzero(np.r_[True, lev[1:] != lev[:-1]]); cuts = np.r_[cuts, len(e1)]
        for passes in range(1, maxit + 1):
            Vo = V.copy(); Qo = Q.copy()
            for li2, pairs in enumerate(s.e3):
                Ei = s.evalpairs(Qo, s.n, pairs); Ei = np.where(np.isfinite(Ei), Ei, BIG); same = s.layer == li2
                Q[same] = np.minimum(Q[same], Ei[same]); V[~same] = np.minimum(V[~same], Ei[~same] + pen)
            if s.subk:                                                                                          # SUBK: переключение внутри ребра столбца — K подточек точного потока, Q(k) = min(…, t_j + Q чужого слоя в точке j)
                own, tt, prs, ns = s._sub()
                for li2, pairs in enumerate(prs):
                    Ei = s.evalpairs(Qo, ns, pairs); m = np.isfinite(Ei) & (s.layer[own] != li2); np.minimum.at(Q, own[m], tt[m] + Ei[m] + pen)
            V = np.minimum(V, Q)
            for a, b in zip(cuts[:-1], cuts[1:]):
                nd = e1[a:b]; Q[nd] = np.minimum(Q[nd], e1c[a:b] + V[e1n[a:b]]); V[nd] = np.minimum(V[nd], Q[nd])
            dd = max(float(np.abs(np.minimum(V, BIG) - np.minimum(Vo, BIG)).max()), float(np.abs(np.minimum(Q, BIG) - np.minimum(Qo, BIG)).max()))
            if progress is not None: progress(passes, dd, V)
            if dd < tol: break
        s.converged = dd < tol; s.passes = passes; s.t_solve = time.perf_counter() - t0; s.V = Q; s.Vn = V
        if verbose: print('solve8[%s]: узлов %d, проходов %d, %.1f с, достижимо %.1f%% узлов' % (s.calc, s.n, passes, s.t_solve, 100 * (V < BIG / 2).mean()), flush=True)
        return V

    # --- запросы ---
    def Vlayers(s, X):
        """V_u(x) по слоям (n, L); inf — не покрыто / V не определено"""
        out = np.full((len(X), s.L), np.inf); inb = np.linalg.norm(X, axis=1) <= s.rho
        PA = s.locate_all(X) if (not s.quad and int(os.environ.get('LOC_ALL', 1))) else None
        for li, u in enumerate(s.us):
            out[:, li] = s.evalpairs(s.V, len(X), PA[li] if PA is not None else s.locate(u, X))
            if s.xentry > 0 and s.calc != 'v': out[:, li] = np.minimum(out[:, li], entry_time(X, u, np.full(len(X), s.xentry), s.rho))       # точный вход по своему потоку — верхняя оценка Q_u (W2)
        out[inb] = 0.; return out

    def Vstar(s, X):
        Vl = s.Vlayers(X); return Vl.min(1), Vl.argmin(1)

    COMMIT_JS = np.array([0, 1, 2, 5, 10, 20, 40])

    def run_commit(s, X0, tmax, dt=.01, eps=1e-3, path=None, rec=5, progress=None, every=None):
        """агент «commit» (lr3/qagent.run_commit, v5chain): идёт своим управлением и переключается, только когда лучший момент переключения — сейчас. На шаге: вдоль СВОЕГО потока точки t_j ∈ {0,1,2,5,10,20,40}·dt, g_j = t_j + min по ДРУГИМ слоям Q_u2(x(t_j)); смена слоя на argmin, если минимум g достигается при j=0 и g_0 < Q своего − eps (или Q своего не определено). Нужен расчёт по Q (set_calc('q'|'qquad')). Выход и path — как у run_agent; progress(step, живых) — вызывается каждый шаг (для строки прогресса в консоли)"""
        n = len(X0); X = X0.copy(); t = np.zeros(n); T = np.full(n, np.nan); sw = np.zeros(n, int); uvec = np.array(s.us); Vl = s.Vlayers(X); cur = Vl.argmin(1); alive = np.isfinite(Vl.min(1)); inn = np.linalg.norm(X, axis=1) <= s.rho; T[inn] = 0; alive &= ~inn; step = 0
        if path is not None: path.append((np.arange(n), X.copy(), cur.copy()))
        every = every or COMMIT_EVERY
        while alive.any():
            a = np.flatnonzero(alive); u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; ar = np.arange(len(a))
            if step % every:                                                                               # решение о переключении — раз в every шагов (счёт ×every быстрее), между ними держим управление
                u = uvec[cur[a]]; X[a, 0] = x + v * dt + u * dt * dt / 2; X[a, 1] = v + u * dt; t[a] += dt; step += 1
                hit = np.linalg.norm(X[a], axis=1) <= s.rho; T[a[hit]] = t[a[hit]]; alive[a[hit]] = False; alive[a[t[a] > tmax[a]]] = False
                if path is not None and (step % rec == 0 or hit.any()): path.append((a, X[a].copy(), cur[a].copy()) if step % rec == 0 else (a[hit], X[a[hit]].copy(), cur[a[hit]].copy()))
                if progress is not None: progress(step, int(alive.sum()))
                continue
            tj = (s.COMMIT_JS * dt)[:, None]; nj = len(tj); xj = x[None] + v[None] * tj + u[None] * tj * tj / 2; vj = v[None] + u[None] * tj                       # все точки t_j одним запросом (быстрее: накладные расходы Vlayers на вызов, ×2.7 на ноуте)
            Q = s.Vlayers(np.stack([xj.ravel(), vj.ravel()], 1)).reshape(nj, len(a), s.L); qcur = Q[0, ar, cur[a]].copy(); Q[:, ar, cur[a]] = np.inf; gg = tj + Q.min(2); bj = gg.argmin(0); best = gg[bj, ar]; bl = Q[bj, ar].argmin(1)
            chg = (bj == 0) & np.isfinite(best) & ((best < qcur - eps) | ~np.isfinite(qcur)); lost = ~np.isfinite(best) & ~np.isfinite(qcur)
            cur[a[chg]] = bl[chg]; sw[a[chg]] += 1; alive[a[lost]] = False; a = np.flatnonzero(alive)
            if not len(a): break
            u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; X[a, 0] = x + v * dt + u * dt * dt / 2; X[a, 1] = v + u * dt; t[a] += dt; step += 1
            hit = np.linalg.norm(X[a], axis=1) <= s.rho; T[a[hit]] = t[a[hit]]; alive[a[hit]] = False; alive[a[t[a] > tmax[a]]] = False
            if path is not None and (step % rec == 0 or hit.any()): path.append((a, X[a].copy(), cur[a].copy()) if step % rec == 0 else (a[hit], X[a[hit]].copy(), cur[a[hit]].copy()))
            if progress is not None: progress(step, int(alive.sum()))
        return T, sw

    def run_agent(s, X0, tmax, dt=.01, eps=1e-3, kappa=0., vdz=.5, vhold=0., look=False, path=None, rec=5, commit=False, progress=None):
        """агенты параллельно: точный поток ДИ шагом dt, смена слоя при выигрыше > порога eps + kappa·max(0, vdz − V*) (мёртвая зона растёт у цели); при V* < vhold слой держится (финиш), пока текущий слой имеет конечное V. → время дохода (nan — провал), число переключений. path — список (по умолчанию None, на счёт не влияет): дописывается (номера агентов, их положения, номер слоя = управление, действовавшее на последнем шаге) начальным состоянием, каждым rec-м шагом и финальным шагом попавших в цель"""
        if commit: return s.run_commit(X0, tmax, dt=dt, eps=eps, path=path, rec=rec, progress=progress)
        n = len(X0); X = X0.copy(); t = np.zeros(n); alive = np.ones(n, bool); T = np.full(n, np.nan); sw = np.zeros(n, int); uvec = np.array(s.us)
        Vl = s.Vlayers(X); cur = Vl.argmin(1); alive &= np.isfinite(Vl.min(1)); inn = np.linalg.norm(X, axis=1) <= s.rho; T[inn] = 0; alive &= ~inn
        step = 0
        if path is not None: path.append((np.arange(n), X.copy(), cur.copy()))
        while alive.any():
            a = np.flatnonzero(alive); u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; X[a, 0] = x + v * dt + u * dt * dt / 2; X[a, 1] = v + u * dt; t[a] += dt; step += 1
            hit = np.linalg.norm(X[a], axis=1) <= s.rho; T[a[hit]] = t[a[hit]]; alive[a[hit]] = False; alive[a[t[a] > tmax[a]]] = False
            if path is not None and (step % rec == 0 or hit.any()): path.append((a, X[a].copy(), cur[a].copy()) if step % rec == 0 else (a[hit], X[a[hit]].copy(), cur[a[hit]].copy()))
            a = np.flatnonzero(alive)
            if not len(a): break
            if look:                                                                                         # Беллман на один шаг: слой с min V*(x после шага под этим слоем); смена слоя при выигрыше > eps
                sc_ = np.empty((len(a), s.L))
                for li, u_ in enumerate(uvec): xn = np.stack([X[a, 0] + X[a, 1] * dt + u_ * dt * dt / 2, X[a, 1] + u_ * dt], 1); sc_[:, li] = s.Vlayers(xn).min(1)
                best = sc_.argmin(1); vc = sc_[np.arange(len(a)), cur[a]]; vb = sc_[np.arange(len(a)), best]; chg = (vb < vc - eps * dt) | (~np.isfinite(vc) & np.isfinite(vb)); sw[a[chg]] += 1; cur[a[chg]] = best[chg]; alive[a[~np.isfinite(vb)]] = False
                if progress is not None: progress(step, int(alive.sum()))
                continue
            Vl = s.Vlayers(X[a]); best = Vl.argmin(1); vc = Vl[np.arange(len(a)), cur[a]]; vb = Vl[np.arange(len(a)), best]; thr = eps + kappa * np.maximum(0., vdz - vb); chg = ((vb < vc - thr) & ~((vb < vhold) & np.isfinite(vc))) | ~np.isfinite(vc) & np.isfinite(vb)
            sw[a[chg]] += 1; cur[a[chg]] = best[chg]; lost = ~np.isfinite(vb); alive[a[lost]] = False
            if progress is not None: progress(step, int(alive.sum()))
        return T, sw


def make_graph(g, atlas, interp='N'):
    G = Graph(g, atlas, interp); G.prepare(); return G


def refine_atlas(g, atlas, thr=.15, maxdepth=2, calc='q', verbose=True):
    """адаптивные столбцы по скачку V (слово пользователя «бисекция»): после solve у соседних столбцов клетки в строке |ΔQ| > thr ⇒ полоса между ними пересеивается M столбцами (новые — потоком от затравки на опорной строке, время строки — интерполяция времён соседей); повтор, пока скачок < thr или глубина maxdepth (ширина < r/16 при M=5, maxdepth 2). Запрос — ячейки полосы (prio). → (новый атлас, граф, число вставленных клеток-полос)"""
    from copy import copy
    M = g.M; US = {u: u for u in atlas}; cur = {u: list(atlas[u]) for u in atlas}; ins = 0
    for d in range(maxdepth + 1):
        G = make_graph(g, cur, 'N'); G.set_calc(calc); b0 = G.bis; G.bis = 0.; G._refined = True; G.solve(verbose=False)
        if d == maxdepth: break
        new = {u: [] for u in cur}; added = 0
        for u in cur:
            for ci, c in enumerate(cur[u]):
                if getattr(c, 'depth', 0) != d: continue
                nt = len(c.G); b = G.base[u][ci]; V = G.V[b:b + nt * M].reshape(nt, M); Gc = np.asarray(c.G); tau = np.asarray(c.tau); k0 = int(min(max(c.nb, 0), nt - 1))
                for j in range(M - 1):
                    fin = (V[:, j] < BIG / 2) & (V[:, j + 1] < BIG / 2)
                    if not (fin & (np.abs(V[:, j] - V[:, j + 1]) > thr)).any(): continue
                    cols = [Gc[:, j]]; ts = [tau[:, j]]
                    for f in np.linspace(0, 1, M)[1:-1]:
                        tt = (1 - f) * tau[:, j] + f * tau[:, j + 1]; y = np.zeros((nt, 2)); y[k0] = (1 - f) * Gc[k0, j] + f * Gc[k0, j + 1]
                        for k in range(k0 + 1, nt): y[k] = g.adv(y[k - 1], u, tt[k] - tt[k - 1])
                        for k in range(k0 - 1, -1, -1): y[k] = g.adv(y[k + 1], u, tt[k] - tt[k + 1])
                        cols.append(y); ts.append(tt)
                    cols.append(Gc[:, j + 1]); ts.append(tau[:, j + 1]); r = float(np.asarray(c.r).ravel()[0]) / (M - 1)
                    new[u].append(Refined(np.stack(cols, 1), np.stack(ts, 1), np.array([r]), c.hb, c.nb, c.nf, getattr(c, 'prio', 0) + 1, d + 1)); added += 1
        if verbose: print('refine: глубина %d → вставлено клеток-полос %d' % (d + 1, added), flush=True)
        if not added: break
        for u in cur: cur[u] += new[u]
        ins += added
    G.bis = b0
    return cur, G, ins


# ---------- сценарий ----------
def starts(g, n, seed):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        p = rng.uniform(-g.XLV, g.XLV)
        if np.linalg.norm(p) > float(g.RHOV[0]): out.append(p)
    return np.array(out)


COMMIT_EVERY = int(os.environ.get('COMMIT_EVERY', 3 if int(os.environ.get('SOLVE8_PLUS', 0)) else 1))          # решение агента commit раз в N шагов dt (1 — каждый шаг, как в lr3/qagent)
POLICIES = {'base': dict(), 'dead+hold': dict(kappa=.05, vdz=.5, vhold=.3), 'look': dict(look=True), 'commit': dict(commit=True)}


def path_extent(X, rho):
    """max |x| вдоль оптимальной траектории (ref_di_disk) для каждого старта"""
    from src.algo import ref_di_disk as R
    _, info = R.T_ref(X, rho, return_details=True); mx = []
    for x, (s, t1, tot) in zip(X, info):
        t1_ = tot if t1 is None else t1; tt = np.linspace(0, tot, 300); a = np.minimum(tt, t1_); b = tt - a; x1 = x[0] + x[1] * a + s * a * a / 2; v1 = x[1] + s * a; mx.append(np.abs(x1 + v1 * b - s * b * b / 2).max())
    return np.array(mx)


def spore_check(G, atlas):
    """V в узлах центральных столбцов спор цели (первая клетка слоёв ±1) против T_ref этих узлов"""
    from src.algo import ref_di_disk as R; out = {}; M = G.M
    for u in (1., -1.):
        if u not in atlas: continue
        li = G.us.index(u); c = atlas[u][0]; gcell = G.gcell[u][0]; ids = np.flatnonzero(G.nodecell == gcell); ids = ids[(ids - ids[0]) % M == M // 2]; P = G.P[ids]; V = G.V[ids]; out_ = np.linalg.norm(P, axis=1) > G.rho
        Tr = R.T_ref(P[out_], G.rho); fin = V[out_] < BIG / 2; d = V[out_][fin] - Tr[fin]
        out[str(u)] = dict(nodes=int(out_.sum()), finite=float(fin.mean()), err_max=float(np.abs(d).max()) if len(d) else None, err_median=float(np.median(np.abs(d))) if len(d) else None, over=float((d > 1e-3).mean()) if len(d) else None, under=float((d < -1e-3).mean()) if len(d) else None, cell_rows=len(c.G))
    return out


def trace(G, x0, tmax, spore_cells, dt=.01, eps=1e-3, **pk):
    """один агент с трассой: время первого попадания в клетки спор цели, число переключений, конец"""
    x = np.array(x0, float)[None]; Vl = G.Vlayers(x); cur = int(Vl.argmin()); t = 0.; first = None; sw = 0; uvec = np.array(G.us)
    while t < tmax:
        if np.linalg.norm(x[0]) <= G.rho: return dict(reached=True, T=t, first_spore=first, switches=sw)
        if first is None:
            for u in (1., -1.):
                if u in G.us:
                    idx = G.idx[u]; pi, hid, sc = idx.query(x)
                    if len(pi) and (G.gcell[u][idx.I[hid, 0]] == spore_cells[u]).any(): first = t; break
        Vl = G.Vlayers(x)[0]; best = int(Vl.argmin()); vc = Vl[cur]; vb = Vl[best]
        if not np.isfinite(vb): return dict(reached=False, T=None, first_spore=first, switches=sw, end=x[0].tolist(), why='V* не определено')
        thr = eps + pk.get('kappa', 0.) * max(0., pk.get('vdz', .5) - vb)
        if (vb < vc - thr and not (vb < pk.get('vhold', 0.) and np.isfinite(vc))) or not np.isfinite(vc): cur = best; sw += 1
        u = uvec[cur]; x[0, 0] += x[0, 1] * dt + u * dt * dt / 2; x[0, 1] += u * dt; t += dt
    return dict(reached=False, T=None, first_spore=first, switches=sw, end=x[0].tolist(), why='время вышло')


def scenario(a):
    g = load_growN(); layers = [float(x) for x in a.layers.split(',')]; t0 = time.perf_counter()
    atlas, tg = build_atlas(layers, a.seed, par=not a.serial); t_grow = time.perf_counter() - t0
    G = make_graph(g, atlas, a.interp); X0 = starts(g, a.nstarts, 1); Tr, src = t_ref(X0); mx = path_extent(X0, G.rho); q = lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan'); allres = {}
    for mname in a.modes.split(','):
        mode, _, pen = mname.partition('+'); G.mode = mode; G.pen = float(pen) if pen else 0.; G.solve(verbose=False); allres[mname] = one_mode(a, g, G, atlas, X0, Tr, mx, src, tg, t_grow, q)
    if a.out: json.dump(allres, open(a.out, 'w'), ensure_ascii=False, indent=1)
    return allres


def one_mode(a, g, G, atlas, X0, Tr, mx, src, tg, t_grow, q):
    layers = [float(x) for x in a.layers.split(',')]
    t1 = time.perf_counter(); Vs, us = G.Vstar(X0); t_q = (time.perf_counter() - t1) / len(X0) * 1e3
    inside = {'xl': mx <= float(g.XLV[0]), 'xl+gm': mx <= float(g.XLV[0]) + float(g.GMV[0])}; vm = np.isfinite(Vs); vr = Vs[vm] / Tr[vm]
    pols = {}; pol_names = list(POLICIES)
    for pn in pol_names:
        t1 = time.perf_counter(); T, sw = G.run_agent(X0, 3 * Tr + 2, **POLICIES[pn]); ms = (time.perf_counter() - t1) / len(X0) * 1e3; ok = np.isfinite(T); r = T[ok] / Tr[ok]
        ri = ok & inside['xl+gm']; r2 = T[ri] / Tr[ri]; pols[pn] = dict(success=float(ok.mean()), in35=dict(success=float(ok[inside['xl+gm']].mean()), n=int(inside['xl+gm'].sum()), T_over_ref_median=q(r2, 50), p90=q(r2, 90), switches_mean=float(sw[inside['xl+gm']].mean())), success_path_in_xl=float(ok[inside['xl']].mean()), success_path_in_xlgm=float(ok[inside['xl+gm']].mean()), success_given_V=float(ok[vm].mean()) if vm.any() else None,
                        T_over_ref=dict(mean=float(r.mean()) if ok.any() else None, median=q(r, 50), p90=q(r, 90), max=float(r.max()) if ok.any() else None), switches_mean=float(sw.mean()), switches_median=float(np.median(sw)), ms_per_run=ms, _T=T, _sw=sw)
    T, sw = pols['base']['_T'], pols['base']['_sw']; ok = np.isfinite(T)
    res = dict(mode=G.mode, pen=G.pen, layers=layers, seed=a.seed, interp=a.interp, M=int(g.M), etol=float(os.environ.get('ETOL', .01)), ref=src, n=len(X0), V_over_ref=dict(median=q(vr, 50), p90=q(vr, 90), max=float(vr.max()) if len(vr) else None, covered=float(vm.mean())),
               path_leaves=dict(xl=float((~inside['xl']).mean()), xlgm=float((~inside['xl+gm']).mean()), no_V_and_leaves_xlgm=float((~vm & ~inside['xl+gm']).sum() / max((~vm).sum(), 1))), cells_nodes={str(u): G.layer_counts[u] for u in G.us},
               t_grow={str(u): tg[u] for u in tg}, t_grow_wall=t_grow, t_prep=G.t_prep, t_solve=G.t_solve, passes=G.passes, ms_per_Vstar=t_q, spore_check=spore_check(G, atlas))
    res['policies'] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in pols.items()}
    sc = {u: G.gcell[u][0] for u in (1., -1.) if u in G.us}; worst = np.argsort(-np.where(ok, T / Tr, np.inf))[:5]
    res['worst'] = [dict(x=X0[i].tolist(), Tref=float(Tr[i]), Vstar=float(Vs[i]) if np.isfinite(Vs[i]) else None, path_max_abs_x=float(mx[i]), trace=trace(G, X0[i], 3 * Tr[i] + 2, sc)) for i in worst]
    print(G.mode, G.pen, 'V/ref p90 %.3f' % res['V_over_ref']['p90'], 'spore under', {u: v['under'] for u, v in res['spore_check'].items()}, {k: (round(v['success'], 3), round(v['in35']['success'], 3), round(v['in35']['T_over_ref_median'], 2), round(v['in35']['p90'], 2), round(v['switches_mean'])) for k, v in res['policies'].items()}, flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('run',)); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--layers', default='1,-1'); ap.add_argument('--interp', default='N')
    ap.add_argument('--nstarts', type=int, default=200); ap.add_argument('--out', default=''); ap.add_argument('--serial', action='store_true'); ap.add_argument('--policies', action='store_true'); ap.add_argument('--modes', default='lin'); a = ap.parse_args(); scenario(a)
