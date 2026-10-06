"""w23 (PLAN п.26, research-18): атлас под запрос — двунаправленный рост (BIDIR) + обрезка по эллипсу T̂_s + V ≤ (1+ε)C.
growN импортируется как модуль, growN.py не правится (владелец — линия B).
Запуск (маятник): SYS=pend UM=.3 GS=60 GLIM=0 FRAC=1 python3 src/cells7/growNq.py
Дерево на слой u: общий HexIdx, две FIFO-очереди — от цели (yb + SIDE) и от старта (yf + SIDE); раунд = поколение каждой очереди.
Стоп: два раунда подряд без новых клеток с V ≤ ST·C (обратное дерево) и T̂_s ≤ ST·C (прямое); затем PRUNE: оставить клетки с min(T̂_s + V) ≤ (1+ε)·C."""
import os, sys, json, time, io, contextlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import growN as G
E = os.environ.get; CTR = (G.M // 2,) * G.m_; ST = float(E('ST', .55)); EPS = float(E('ELLEPS', .2)); SIDE = int(E('SIDE', 1)); NQ = int(E('NQ', 5)); MAXR = int(E('MAXR', 80))
def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a)
class Tree:
    """слой u: клетки, общий индекс покрытия, очереди back (от цели) и fwd (от старта)."""
    def __init__(s, u, rng, back=True):
        s.u = u; s.cells = []; s.idx = G.HexIdx(); s.q = {'b': list(G.goal_seeds(rng)) if back else [], 'f': []}
    def seed_fwd(s, y): s.q['f'].append(np.array(y, float))
    def round(s, which):
        """одно поколение очереди which ('b' | 'f'); вернуть добавленные клетки."""
        items = s.q[which]; s.q[which] = []; added = []; u = s.u; sg = -1. if which == 'b' else 1.
        for p in items:
            if len(s.cells) >= G.MAXC or not G.inbox(p): continue
            p = G.wrapy(p)
            if G.ingoal(p) or s.idx.covered(p[None])[0]: continue
            rm, tm = G.LIMITS(p, u) if G.LIMITS else (G.RMAX, G.TMAX); c = G.growN(p, u, s.idx, rm, tm)
            if c is None: continue
            c.build(); s.cells.append(c); s.idx.add(c); added.append(c); g0, g1 = c.G[-1], c.G[0]; end = g1 if which == 'b' else g0; y = end[CTR]
            for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2))): y = G.step(y, u, sg)
            new = [y]
            if SIDE:
                for e0 in (c.c, end[CTR]):
                    for k in range(G.m_): new += [e0 + 1.9 * c.r[k] * c.e[k], e0 - 1.9 * c.r[k] * c.e[k]]
            s.q[which] += new
        return added
def atlas_of(trees):
    A = G.Atlas.__new__(G.Atlas); A.layers = [t.cells for t in trees]; A.finish(); return A
def solve_fwd(A, s, it=20000, r0=None):
    """T̂_s(узел) — время из s: Якоби по стенсилам step(P, u, −1); источник — узлы в коробке r0 вокруг s (T̂ = 0). Прототип research-18 (r18/ell_prune.py)."""
    I_, IDX_, W_ = [], [], []
    for u in G.US:
        a, b, c_ = A.stencils(G.step(A.P, u, -1.)); I_.append(a); IDX_.append(b); W_.append(c_)
    I = np.concatenate(I_); IDX = np.concatenate(IDX_); W = np.concatenate(W_); o = np.argsort(I, kind='stable'); I, IDX, W = I[o], IDX[o], W[o]
    st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]
    r0 = G.RHOV * 1.5 if r0 is None else r0; src = (np.abs(G.wrapy(A.P - s)) <= r0).all(1); T = np.full(A.N, G.BIG); T[src] = 0.
    for n in range(it):
        val = G.DTN + A.interp(W, T[IDX]); new = T.copy(); new[nd] = np.minimum(T[nd], np.minimum.reduceat(val, st)); new[src] = 0.
        d = np.max(np.abs(new - T)); T = new
        if d < 1e-9: break
    return T, int(src.sum())
def cell_min(A, X):
    return np.array([X[c.o:c.o + c.G.shape[0] * G.M ** G.m_].min() for c in A.cells])
def solve_both(A, s):
    quiet(A.solve); Ts, _ = solve_fwd(A, s); fin = (A.V < G.BIG / 2) & (Ts < G.BIG / 2); C = float((A.V + Ts)[fin].min()) if fin.any() else np.inf
    return Ts, C
def bidir(s, log=print):
    """двунаправленный рост под старт s; вернуть (A, Ts, C, info)."""
    rng = np.random.default_rng(0); trees = [Tree(u, np.random.default_rng(k)) for k, u in enumerate(G.US)]
    for t in trees: t.seed_fwd(s)
    stall = 0; t0 = time.time(); info = dict(rounds=0)
    for r in range(MAXR):
        addb = [c for t in trees for c in t.round('b')]; addf = [c for t in trees for c in t.round('f')]; nb, nf = len(addb), len(addf)
        A = atlas_of(trees); Ts, C = solve_both(A, s); info['rounds'] = r + 1
        if np.isfinite(C):
            vm, tm = cell_min(A, A.V), cell_min(A, Ts); idm = {id(c): i for i, c in enumerate(A.cells)}
            hit = any(vm[idm[id(c)]] <= ST * C for c in addb) or any(tm[idm[id(c)]] <= ST * C for c in addf)
            stall = 0 if hit else stall + 1
        log('r%d +b %d +f %d всего %d C %.3f stall %d' % (r + 1, nb, nf, len(A.cells), C, stall))
        if stall >= 2 or (nb == 0 and nf == 0): break
    info['cells'] = len(A.cells); info['sec'] = round(time.time() - t0, 1); return A, Ts, C, info
class Inc:
    """w23 (тёплый старт, research-18/19): атлас, растущий по раундам. Узлы клеток нумеруются в порядке создания (номера не сдвигаются), HexIdx и стенсилы
    считаются только для новых клеток; у старых узлов пересчитываются стенсилы лишь тех, чья точка попала в новые клетки (малый индекс новых клеток).
    V и T̂_s прошлого раунда — верхние оценки (новые клетки только добавляют варианты), Якоби вниз сходится к тому же, что и холодный старт."""
    def __init__(s, s0):
        A = G.Atlas.__new__(G.Atlas); A.cells = []; A.qx = G.HexIdx(); A.O = np.zeros(0, int); A.P = np.zeros((0, G.N)); A.N = 0; A.V = np.zeros(0); A.goal = np.zeros(0, bool)
        s.A = A; s.s0 = s0; s.Ts = np.zeros(0); s.Vg = np.zeros(0); s.st = {(k, sg): [np.zeros(0, int), np.zeros((0, 2 ** G.N), np.int64), np.zeros((0, 2 ** G.N), np.float32)] for k in range(len(G.US)) for sg in (1, -1)}
    def add(s, cells):
        A = s.A; n0 = A.N; newidx = G.HexIdx(); offs = []; Ps = []
        for c in cells:
            offs.append(A.N); A.N += c.G.shape[0] * G.M ** G.m_; A.qx.add(c); newidx.add(c); Ps.append(c.G.reshape(-1, G.N)); A.cells.append(c)
        if not cells: return
        Pn = np.concatenate(Ps); A.O = np.concatenate([A.O, offs]).astype(int); A.P = np.concatenate([A.P, Pn]); gn = G.ingoal(Pn); A.goal = np.concatenate([A.goal, gn])
        A.V = np.concatenate([A.V, np.where(gn, 0., G.BIG)]); s.Ts = np.concatenate([s.Ts, np.full(len(Pn), G.BIG)])
        vg = np.full(len(Pn), np.inf)
        for u in G.US: vg = np.minimum(vg, A.tgoal(Pn, u))
        s.Vg = np.concatenate([s.Vg, vg])
        for k, u in enumerate(G.US):
            for sg in (1, -1):
                Y = G.wrapy(G.step(A.P, u, float(sg))); hit = np.unique(newidx.query(Y[:n0])[0]) if n0 else np.zeros(0, int); R = np.concatenate([hit, np.arange(n0, A.N)]).astype(int)
                pi, IDX, W = A.stencils(Y[R]); I, IDX0, W0 = s.st[(k, sg)]; keep = ~np.isin(I, hit)
                s.st[(k, sg)] = [np.concatenate([I[keep], R[pi]]), np.concatenate([IDX0[keep], IDX.astype(np.int64)]), np.concatenate([W0[keep], W])]
    def _rows(s, sg):
        I = np.concatenate([s.st[(k, sg)][0] for k in range(len(G.US))]); IDX = np.concatenate([s.st[(k, sg)][1] for k in range(len(G.US))]); W = np.concatenate([s.st[(k, sg)][2] for k in range(len(G.US))])
        o = np.argsort(I, kind='stable'); I, IDX, W = I[o], IDX[o], W[o]; st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); return IDX, W, st, I[st]
    def _jac(s, X, IDX, W, st, nd, fix, fixval, it=20000):
        for n in range(it):
            val = G.DTN + s.A.interp(W, X[IDX]); new = X.copy(); new[nd] = np.minimum(X[nd], np.minimum.reduceat(val, st)); new[fix] = fixval
            d = np.max(np.abs(new - X)); X = new
            if d < 1e-9: break
        return X, n
    def solve(s):
        """тёплый V и T̂_s; вернуть (Ts, C, итераций V, итераций T)."""
        A = s.A; IDX, W, st, nd = s._rows(1); V0 = np.minimum(A.V, s.Vg); A.V, nv = s._jac(V0, IDX, W, st, nd, A.goal, 0.)
        IDX, W, st, nd = s._rows(-1); src = (np.abs(G.wrapy(A.P - s.s0)) <= G.RHOV * 1.5).all(1); T0 = s.Ts.copy(); T0[src] = 0.; s.Ts, nt = s._jac(T0, IDX, W, st, nd, src, 0.)
        fin = (A.V < G.BIG / 2) & (s.Ts < G.BIG / 2); C = float((A.V + s.Ts)[fin].min()) if fin.any() else np.inf; return C, nv, nt
    def cell_min(s, X): return np.array([X[o:o + c.G.shape[0] * G.M ** G.m_].min() for c, o in zip(s.A.cells, s.A.O)])
def bidir_inc(s, log=print):
    """BIDIR с тёплым стартом (WARM=1): те же деревья/стоп, что bidir; solve — Inc. В конце один холодный solve (точные C, V, T̂_s)."""
    trees = [Tree(u, np.random.default_rng(k)) for k, u in enumerate(G.US)]
    for t in trees: t.seed_fwd(s)
    inc = Inc(s); stall = 0; t0 = time.time(); info = dict(rounds=0, sec_grow=0., sec_solve=0.)
    for r in range(MAXR):
        ta = time.time(); addb = [c for t in trees for c in t.round('b')]; addf = [c for t in trees for c in t.round('f')]; tb = time.time(); inc.add(addb + addf); C, nv, nt = inc.solve(); tc = time.time()
        info['sec_grow'] += tb - ta; info['sec_solve'] += tc - tb; info['rounds'] = r + 1
        if np.isfinite(C):
            vm, tm = inc.cell_min(inc.A.V), inc.cell_min(inc.Ts); idm = {id(c): i for i, c in enumerate(inc.A.cells)}
            hit = any(vm[idm[id(c)]] <= ST * C for c in addb) or any(tm[idm[id(c)]] <= ST * C for c in addf); stall = 0 if hit else stall + 1
        log('r%d +b %d +f %d всего %d C %.3f stall %d it V %d T %d solve %.1fс' % (r + 1, len(addb), len(addf), len(inc.A.cells), C, stall, nv, nt, tc - tb))
        if stall >= 2 or (not addb and not addf): break
    Cinc = C; A = atlas_of(trees); tc0 = time.time(); Ts, C = solve_both(A, s); info['sec_cold_final'] = round(time.time() - tc0, 1); info['C_inc'] = round(Cinc, 4); info['cells'] = len(A.cells); info['sec'] = round(time.time() - t0, 1)
    info['sec_grow'] = round(info['sec_grow'], 1); info['sec_solve'] = round(info['sec_solve'], 1); return A, Ts, C, info
def pool_run(Qs, log=print):
    """26в: общий пул клеток на несколько стартов. Обратные деревья и клетки остаются между стартами; для нового старта — прямая очередь от него + раунды до стопа (тот же критерий), T̂_s считается заново, V — тёплый."""
    trees = [Tree(u, np.random.default_rng(k)) for k, u in enumerate(G.US)]; inc = Inc(G.wrapy(Qs[0])); out = []
    for qi, s in enumerate(Qs):
        s = G.wrapy(s); inc.s0 = s; inc.Ts[:] = G.BIG
        for t in trees: t.seed_fwd(s)
        stall = 0; n_before = len(inc.A.cells); t0 = time.time()
        for r in range(MAXR):
            addb = [c for t in trees for c in t.round('b')]; addf = [c for t in trees for c in t.round('f')]; inc.add(addb + addf); C, nv, nt = inc.solve()
            if np.isfinite(C):
                vm, tm = inc.cell_min(inc.A.V), inc.cell_min(inc.Ts); idm = {id(c): i for i, c in enumerate(inc.A.cells)}
                hit = any(vm[idm[id(c)]] <= ST * C for c in addb) or any(tm[idm[id(c)]] <= ST * C for c in addf); stall = 0 if hit else stall + 1
            log('s%d r%d +b %d +f %d пул %d C %.3f stall %d' % (qi, r + 1, len(addb), len(addf), len(inc.A.cells), C, stall))
            if stall >= 2 or (not addb and not addf and np.isfinite(C)): break
        cm = inc.cell_min(inc.Ts + inc.A.V); keep = {id(c): cm[i] <= (1 + EPS) * C for i, c in enumerate(inc.A.cells)}
        out.append(dict(s=s, C=C, pool=len(inc.A.cells), new=len(inc.A.cells) - n_before, keep_ids={k for k, v in keep.items() if v}, sec=round(time.time() - t0, 1)))
    return trees, inc, out
def prune(A, Ts, C, eps, s_=None):
    cm = cell_min(A, Ts + A.V); keep = cm <= (1 + eps) * C; A2 = G.Atlas.__new__(G.Atlas); A2.layers = [[c for c in l if keep[A.cells.index(c)]] for l in A.layers]; A2.finish(); quiet(A2.solve); T2 = A2.rollout(s_[None])[0][0] if s_ is not None else None
    n_ = 0
    for c in A.cells: c.o = n_; n_ += c.G.shape[0] * G.M ** G.m_          # клетки общие с A: finish() A2 переписал c.o — вернуть смещения A
    return A2, keep, T2
def full_atlas():
    """полный атлас: рост только от цели (FIFO + SIDE) до исчерпания очереди — эталон сравнения."""
    trees = [Tree(u, np.random.default_rng(k)) for k, u in enumerate(G.US)]
    while any(t.q['b'] for t in trees) and all(len(t.cells) < G.MAXC for t in trees):
        for t in trees: t.round('b')
    A = atlas_of(trees); quiet(A.solve); return A
if __name__ == '__main__':
    t0 = time.time(); Af = full_atlas(); print('полный атлас: клеток', len(Af.cells), 'узлов', Af.N, round(time.time() - t0), 'с', flush=True)
    Q, ref = G.starts_ref(); Cs = Af.vstar(G.wrapy(Q)); ok = np.flatnonzero(Cs < G.BIG / 2); print('стартов с конечной V', len(ok), 'из', len(Q), flush=True)

    if int(E('POOL', 0)):
        Q0 = int(E('Q0', 0)); idq = ok[Q0:Q0 + NQ]; Qs = [Q[i] for i in idq]; trees, inc, out = pool_run(Qs, log=lambda m: print('  ', m, flush=True)); Ap = atlas_of(trees); quiet(Ap.solve); allc = {id(c) for t in trees for c in t.cells}
        for i, o in zip(idq, out):
            s = o['s']; A2 = G.Atlas.__new__(G.Atlas); A2.layers = [[c for c in t.cells if id(c) in o['keep_ids']] for t in trees]; A2.finish(); quiet(A2.solve); Tfull = float(Af.rollout(s[None])[0][0]); Tpool = float(Ap.rollout(s[None])[0][0]); Tp = float(A2.rollout(s[None])[0][0])
            print(json.dumps(dict(q=int(i), ref=round(float(ref[i]), 3), T_full=round(Tfull, 3), T_pool_all=round(Tpool, 3), cells_full=len(Af.cells), pool_after=o['pool'], new_cells=o['new'], C=round(o['C'], 3), pruned_cells=len(o['keep_ids']), T_pruned=round(Tp, 3), sec=o['sec'])), flush=True)
        print('ПУЛ: клеток', len(allc), 'против полного', len(Af.cells), flush=True); sys.exit(0)
    for i in ok[int(E('Q0', 0)):int(E('Q0', 0)) + NQ]:
        s = G.wrapy(Q[i]); Tfull = float(Af.rollout(s[None])[0][0]); Cf = float(Af.vstar(s[None])[0]); A, Ts, C, info = (bidir_inc if int(E('WARM', 0)) else bidir)(s, log=lambda m: print('  ', m, flush=True))
        T_b = float(A.rollout(s[None])[0][0]); res = dict(q=int(i), ref=round(float(ref[i]), 3), C_full=round(Cf, 3), T_full=round(Tfull, 3), cells_full=len(Af.cells), C_bidir=round(C, 3), T_bidir=round(T_b, 3), cells_bidir=info['cells'], rounds=info['rounds'], sec=info['sec'], **{k: v for k, v in info.items() if k.startswith('sec_') or k == 'C_inc'})
        for eps in (EPS, .5):
            A2, keep, T2 = prune(A, Ts, C, eps, s); res['eps%.1f' % eps] = dict(cells=int(keep.sum()), frac_of_full=round(float(keep.sum() / len(Af.cells)), 3), T=round(float(T2), 3))
        print(json.dumps(res), flush=True)
