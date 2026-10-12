"""b3: гипотеза шва θ=±π: (1) доля стартов с V*<T_ref−.1 по |θ|; (2) V* вдоль пути агента для старта I (V* против остатка T−t)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); PI = np.pi; rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-PI, PI, 100), rq.uniform(-2, 2, 100)], 1)
ref = np.load('pend_ref_best_u0.3.npy'); atlas, tg = S.build_atlas(list(g.US), 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
Vs = G.Vlayers(Q).min(1); low = Vs < ref - .1; high = Vs > ref + .3; a = np.abs(Q[:, 0])
for nm, m in (('|θ|<1', a < 1), ('1≤|θ|<2', (a >= 1) & (a < 2)), ('2≤|θ|<2.5', (a >= 2) & (a < 2.5)), ('|θ|≥2.5 (шов)', a >= 2.5)): print('RES %-14s стартов %2d: занижено V*<T_ref−.1 %2d (%.0f%%), завышено >+.3 %2d' % (nm, m.sum(), (low & m).sum(), 100 * (low & m).sum() / max(m.sum(), 1), (high & m).sum()))
path = []; T, sw = G.run_agent(Q, 3 * np.where(np.isfinite(ref), ref, 5) + 2, path=path, rec=5, **S.POLICIES['commit'])
for I in map(int, sys.argv[1:]):
    pts = [(st, P[list(ix).index(I)]) for st, (ix, P, c) in enumerate(path) if I in ix]
    print('RES старт %d T_agent %.2f T_ref %.2f V*(старт) %.2f; путь (θ, ω, V*, остаток T−t, V*−остаток):' % (I, T[I], ref[I], Vs[I]))
    X = np.array([p for _, p in pts]); V = G.Vlayers(X).min(1); t = np.arange(len(X)) * .05
    for k in range(0, len(X), max(1, len(X) // 14)): print('RES   t=%.2f θ=%6.2f ω=%6.2f V*=%6.2f ост=%6.2f Δ=%+.2f' % (t[k], *X[k], V[k], T[I] - t[k], V[k] - (T[I] - t[k])))
