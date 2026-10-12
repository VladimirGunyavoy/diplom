"""b2: маятник на механике v8 (solve8, SOLVE8_PLUS): старты/эталон как в хабовом бенче (bench/common.py: rng 0, θ∈±π, ω∈±2, 100 стартов, ref = pend_ref_best_u0.3.npy, keep = finite & >.05). Цель — ящик ±RHO (GOALSHAPE=box).
SYS=pend GOALSHAPE=box GSEED=0 GOALB=1 DELTA=1e9 TMAX=8 RMAX=.3 RHO=.1 M=5 DTN=.06 SOLVE8_PLUS=1 [MAXC=5] python3 reports/laptop_b/bench_pend.py   (PEND_REF — папка с pend_ref_best_u0.3.npy)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); PI = np.pi; rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-PI, PI, 100), rq.uniform(-2, 2, 100)], 1)
ref = np.load(os.path.join(os.environ.get('PEND_REF', '.'), 'pend_ref_best_u0.3.npy')); keep = np.isfinite(ref) & (ref > .05); tag = os.environ.get('TAG', '-')
t0 = time.perf_counter(); atlas, tg = S.build_atlas(list(g.US), 0, par=False); tb = time.perf_counter() - t0; G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); t0 = time.perf_counter(); G.solve(verbose=False); ts = time.perf_counter() - t0
print('RES[%s] клеток %s узлов %d | рост %.1f с, solve %.1f с (таблица %.1f с) проходов %d' % (tag, {u: len(atlas[u]) for u in atlas}, G.n, tb, ts, getattr(G, 't_table', 0), G.passes), flush=True)
for pol in ('commit', 'look'):
    t0 = time.perf_counter(); T, sw = G.run_agent(Q, 3 * np.where(np.isfinite(ref), ref, 5) + 2, **S.POLICIES[pol]); ta = time.perf_counter() - t0; ok = np.isfinite(T) & (T <= 3 * np.where(np.isfinite(ref), ref, 0)); r = T[keep & ok] / ref[keep & ok]
    print('RES[%s]   %-6s агент 100 стартов %.2f с | keep %d дошли %.3f T/T_ref мед %.3f p90 %.3f max %.2f | переключений мед %.1f' % (tag, pol, ta, keep.sum(), ok[keep].mean(), np.median(r), np.percentile(r, 90), r.max(), np.median(sw[keep])), flush=True)
