# research-18: BFS-рост от цели (только yb + SIDE) — идут ли поколения клеток по уровням V? (допущение PLAN 26б)
# Запуск: SYS=pend UM=.3 GS=60 python3 bfs_gen.py  (growN импортируется как модуль, src не трогаем)
import os, sys, numpy as np, json, time
sys.path.insert(0, os.path.expanduser('~/claude-work/projects/spore/spores_2/v7/src/cells7'))
import growN as G, heapq
HEAP = int(os.environ.get('HEAP', 0))
def build_back(u, rng):
    """HEAP=0: FIFO по поколениям; HEAP=1: куча по tch — время до цели по цепочке клеток (верхняя оценка V)."""
    cells = []; idx = G.HexIdx(); q = [(0., i, p, 0) for i, p in enumerate(G.goal_seeds(rng))]; cnt = len(q); ctr = (G.M // 2,) * G.m_; side = int(os.environ.get('SIDE', 1))
    if HEAP: heapq.heapify(q)
    while q and len(cells) < G.MAXC:
        tch, _, p, g = heapq.heappop(q) if HEAP else q.pop(0)
        if not G.inbox(p): continue
        p = G.wrapy(p)
        if G.ingoal(p) or idx.covered(p[None])[0]: continue
        rm, tm = G.LIMITS(p, u) if G.LIMITS else (G.RMAX, G.TMAX); c = G.growN(p, u, idx, rm, tm)
        if c is None: continue
        c.build(); c.gen = g; c.tch = tch; c.order = len(cells); cells.append(c); idx.add(c); g1 = c.G[0]; yb = g1[ctr]
        nst = max(2, int(.9 * (c.nf + c.nb) / 2)); tb = tch + c.nb * G.DTN
        for _ in range(nst): yb = G.step(yb, u, -1.)
        new = [(tb + nst * G.DTN, yb)]
        if side:
            for e0, t0 in ((c.c, tch), (g1[ctr], tb)):
                for k in range(G.m_): new += [(t0, e0 + 1.9 * c.r[k] * c.e[k]), (t0, e0 - 1.9 * c.r[k] * c.e[k])]
        for t_, y_ in new:
            cnt += 1; it = (t_, cnt, y_, g + 1)
            if HEAP: heapq.heappush(q, it)
            else: q.append(it)
    return cells
def _l(a): return build_back(a[0], np.random.default_rng(a[1]))
if __name__ == '__main__':
    t0 = time.time(); from multiprocessing import Pool
    A = G.Atlas.__new__(G.Atlas)
    with Pool(len(G.US)) as pool: A.layers = pool.map(_l, [(u, k) for k, u in enumerate(G.US)])
    A.finish(); print('построено', [len(l) for l in A.layers], 'узлов', A.N, round(time.time() - t0), 'с', flush=True); A.solve()
    gen = np.array([c.gen for c in A.cells]); order = np.array([c.order for c in A.cells]); tch = np.array([c.tch for c in A.cells]); vmin = np.array([A.V[c.o:c.o + c.G.shape[0] * G.M ** G.m_].min() for c in A.cells])
    fin = vmin < G.BIG / 2; print('клеток', len(gen), 'с конечной V', fin.sum())
    rng = np.random.default_rng(9); pr = np.array([G.rand_seed(rng) for _ in range(3000)]); vs = A.vstar(G.wrapy(pr)); print('покрытие поля (V конечна у проб)', round(float((vs < G.BIG / 2).mean()), 3))
    out = []
    for v in np.quantile(vmin[fin], [.1, .25, .5, .75, .9]):
        sub = fin & (vmin <= v); n_sub = sub.sum(); n_bfs = 0
        for l in A.layers: o_ = np.array([c.order for c in l]); vv = np.array([A.V[c.o:c.o + c.G.shape[0] * G.M ** G.m_].min() for c in l]); n_bfs += (o_.max() + 1) if False else (o_ <= o_[vv <= v].max()).sum() if (vv <= v).any() else 0
        need = -1
        out.append(dict(v=round(float(v), 2), cells_V_le_v=int(n_sub), gen_needed=int(need), cells_up_to_gen=int(n_bfs), overhead=round(n_bfs / n_sub, 2)))
        print(out[-1], flush=True)
    print('корр. tch–V', round(float(np.corrcoef(tch[fin], vmin[fin])[0, 1]), 3), 'tch/V мед', round(float(np.median(tch[fin & (vmin > .3)] / vmin[fin & (vmin > .3)])), 3)); print('корр. gen–V', round(float(np.corrcoef(gen[fin], vmin[fin])[0, 1]), 3))
    np.savez(os.environ.get('OUT', 'bfs_gen_%s_h%d.npz' % (G.SYS, HEAP)), gen=gen, vmin=vmin)
    if os.environ.get('ROLL'):
        Q, ref = G.starts_ref(); T, sw, _ = A.rollout(Q); fz = np.isfinite(T) & np.isfinite(ref); r = T[fz] / ref[fz]
        print('ROLL дошли', int(np.isfinite(T).sum()), '/', len(T), 'T/ref mean', round(float(r.mean()), 4), 'med', round(float(np.median(r)), 4), 'max', round(float(r.max()), 3), flush=True)
