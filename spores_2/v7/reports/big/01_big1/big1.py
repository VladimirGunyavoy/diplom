"""BIG1: запрет малых клеток КАК ПРАВИЛО ПОСТРОЕНИЯ (маятник u .3). Ленивая жадная вставка: кандидаты-семена → клетка растёт пробно (grow2 против текущего покрытия),
принимается самая большая; семя принимается только если объём (2r·(nb+nf)·DTN) ≥ VMIN. Объём кандидата при росте покрытия только убывает ⇒ ленивая куча корректна.
Запуск из bench/: VMIN=.2 NCAND=600 python3 ../v7/reports/big/01_big1/big1.py"""
import os, sys, json, time, heapq
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big')); import big_lib as B, numpy as np
G = B.G; VMIN = float(os.environ.get('VMIN', .2)); NCAND = int(os.environ.get('NCAND', 600)); SEED = int(os.environ.get('BSEED', 0)); GOALSEEDS = int(os.environ.get('GOALSEEDS', 1))
def vol(c): return 2 * c.r * (c.nf + c.nb) * G.DTN
STAT = dict(grown=0, accepted=0, rej_small=0)
def build_layer_big(u, rng, log=None):
    cells = []; idx = G.Index(); cnt = 0; heap = []
    def push(p, key=-1e9):
        nonlocal cnt; cnt += 1; heapq.heappush(heap, (key, cnt, np.asarray(p, float)))
    for _ in range(NCAND): push([rng.uniform(-G.XL, G.XL), rng.uniform(-G.WL, G.WL)])
    if GOALSEEDS:
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False): push([1.5 * G.RHO * np.cos(a), 1.5 * G.RHO * np.sin(a)])
    while heap:
        key, _, p = heapq.heappop(heap)
        if not G.inbox(p): continue
        p = np.array([G.wrap(p[0]), p[1]])
        if np.linalg.norm(G.f(p, u)) < G.FMIN or idx.covered(p[None])[0]: continue
        rm, tm = G.LIMITS(p, u) if G.LIMITS else (G.RMAX, G.TMAX)
        c = G.grow2(p, u, idx, rm, tm); STAT['grown'] += 1
        if c is None or c.nb + c.nf == 0: continue
        v = vol(c)
        if v < VMIN: STAT['rej_small'] += 1; continue
        if heap and -heap[0][0] > v + 1e-12 and key != -v: push(p, -v); continue    # устарела оценка: вернуть с новой, пусть вперёд идёт более крупный
        STAT['accepted'] += 1; c.build(); [cells.append(d) or idx.add(d) for d in G.split_rows(c)]
        g0, g1 = c.G[-1], c.G[0]; yf = g0[len(g0) // 2]; yb = g1[len(g1) // 2]
        for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2))): yf = G.step(yf, u); yb = G.step(yb, u, -1.)
        for q in [yf, yb, c.c + 1.9 * c.r * c.n, c.c - 1.9 * c.r * c.n]: push(q, -v)
    return cells, idx
if __name__ == '__main__':
    G.build_layer = build_layer_big
    t0 = time.perf_counter(); A = G.Atlas(seed=SEED); tg = time.perf_counter() - t0; ts = B.solve(A); s = B.evalA(A)
    r = dict(VMIN=VMIN, NCAND=NCAND, seed=SEED, cells=len(A.cells), per_layer=[len(l) for l in A.layers], nodes=int(A.N), t_grow=tg, t_solve=ts, reached=s['reached'], ratio_med=s['ratio_med'], ratio_p90=s['ratio_p90'], ratio_max=s['ratio_max'], stat=STAT, T=s['T'])
    print(json.dumps({k: v for k, v in r.items() if k != 'T'}), flush=True)
    json.dump(r, open(os.path.join(os.environ['BIGOUT'], 'big1_vmin%g_n%d_s%d.json' % (VMIN, NCAND, SEED)), 'w'), indent=1, default=float)
