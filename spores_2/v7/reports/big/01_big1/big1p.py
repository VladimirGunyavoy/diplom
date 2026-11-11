"""BIG1-P: то же правило (семя принимается при объёме ≥ VMIN, крупные первыми), но пробный рост кандидатов — раунд за раундом параллельно (Pool fork, NPROC ядер).
Раунд: все живые кандидаты растут против ТЕКУЩЕГО покрытия параллельно → кандидаты с объёмом < VMIN умирают (объём при росте покрытия только убывает) → принимаются по убыванию объёма клетки,
чьи рамки (bb, с периодом θ) не пересекают уже принятых в этом раунде; остальные идут в следующий раунд (растут заново против нового покрытия).
Запуск из bench/: VMIN=.8 NPROC=28 BIGOUT=… python3 ../v7/reports/big/01_big1/big1p.py"""
import os, sys, json, time, multiprocessing as mp
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big')); import big_lib as B, numpy as np
G = B.G; VMIN = float(os.environ.get('VMIN', .8)); NCAND = int(os.environ.get('NCAND', 600)); SEED = int(os.environ.get('BSEED', 0)); NPROC = int(os.environ.get('NPROC', 28)); TAG = os.environ.get('TAG', '')
def vol(c): return 2 * c.r * (c.nf + c.nb) * G.DTN
STAT = dict(grown=0, accepted=0, rejected=0, rounds=0); _CTX = {}
def _grow(p):
    idx, u = _CTX['idx'], _CTX['u']; p = np.array([G.wrap(p[0]), p[1]])
    if not G.inbox(p) or np.linalg.norm(G.f(p, u)) < G.FMIN or idx.covered(p[None])[0]: return None
    rm, tm = G.LIMITS(p, u) if G.LIMITS else (G.RMAX, G.TMAX); c = G.grow2(p, u, idx, rm, tm)
    if c is None or c.nb + c.nf == 0: return ('small', 0.)
    v = vol(c)
    if v < VMIN: return ('small', v)
    c.build(); return ('ok', v, c)
def _hit(bb1, bb2):
    for sh in (-2 * np.pi, 0., 2 * np.pi):
        if bb1[0] + sh <= bb2[1] and bb1[1] + sh >= bb2[0] and bb1[2] <= bb2[3] and bb1[3] >= bb2[2]: return True
    return False
def build_layer_big(u, rng, log=None):
    cells = []; idx = G.Index(); cand = [np.array([rng.uniform(-G.XL, G.XL), rng.uniform(-G.WL, G.WL)]) for _ in range(NCAND)]
    cand += [np.array([1.5 * G.RHO * np.cos(a), 1.5 * G.RHO * np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    while cand:
        STAT['rounds'] += 1; _CTX['idx'] = idx; _CTX['u'] = u
        with mp.get_context('fork').Pool(NPROC) as pool: out = pool.map(_grow, cand, chunksize=max(1, len(cand) // (4 * NPROC)))
        STAT['grown'] += len(cand); alive = [(o[1], i, o[2]) for i, o in enumerate(out) if o is not None and o[0] == 'ok']; STAT['rejected'] += sum(1 for o in out if o is not None and o[0] == 'small')
        alive.sort(key=lambda t: -t[0]); acc = []; nxt = []
        for v, i, c in alive:
            if any(_hit(c.bb, d.bb) for d in acc): nxt.append(cand[i]); continue
            acc.append(c)
        if not acc: break
        for c in acc:
            STAT['accepted'] += 1; [cells.append(d) or idx.add(d) for d in G.split_rows(c)]
            g0, g1 = c.G[-1], c.G[0]; yf = g0[len(g0) // 2]; yb = g1[len(g1) // 2]
            for _ in range(max(2, int(.9 * (c.nf + c.nb) / 2))): yf = G.step(yf, u); yb = G.step(yb, u, -1.)
            nxt += [yf, yb, c.c + 1.9 * c.r * c.n, c.c - 1.9 * c.r * c.n]
        cand = nxt
    return cells, idx
if __name__ == '__main__':
    if os.environ.get('NOZERO'): G.US = (-G.UM, G.UM)
    if not os.environ.get('ORIG'): G.build_layer = build_layer_big
    t0 = time.perf_counter(); A = G.Atlas(seed=SEED); tg = time.perf_counter() - t0; ts = B.solve(A); s100, _ = B.eval100(A); s = B.evalA(A)
    import pickle; pickle.dump(dict(layers=A.layers), open(os.path.expanduser('~/spore_v5/a31/big1p_vmin%g_s%d.pkl' % (VMIN, SEED)), 'wb'))
    r = dict(tag=TAG, nozero=bool(os.environ.get('NOZERO')), orig=bool(os.environ.get('ORIG')), VMIN=VMIN, NCAND=NCAND, seed=SEED, nproc=NPROC, cells=len(A.cells), per_layer=[len(l) for l in A.layers], nodes=int(A.N), t_grow=tg, t_solve=ts, s100=s100, reached20=s['reached'], ratio20_med=s['ratio_med'], stat=STAT)
    print(json.dumps(r), flush=True); json.dump(r, open(os.path.join(os.environ['BIGOUT'], 'big1p_vmin%g_n%d_s%d%s.json' % (VMIN, NCAND, SEED, TAG)), 'w'), indent=1, default=float)
