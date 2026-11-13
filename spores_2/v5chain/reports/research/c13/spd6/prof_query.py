"""Блок 3 (3): cProfile одного запроса коридора ДИ6D (NB50k NF3000 K3), старт 0..: топ функций по tottime и cumulative; без оптимизации."""
import sys, runpy, cProfile, pstats, time, numpy as np
sys.argv = ['spores_di.py', 'di6', '50000', '3000', '3', 'prof', '/home/random/spore_v5/r37/di6_ref_200_rho35_s8.npy']
g = runpy.run_path('spores_di.py', run_name='lib'); C = g['corridor_query']
S, back, e, tau, rho, gg, miss, flow = [g[k] for k in ('S', 'back', 'e', 'tau', 'rho', 'g', 'miss', 'flow')]
i = 0; C(S, flow, e.X0[i], back, tau, 3000, rho, gg, miss, K=3, tries=3, kn=8, neigh=0, nbest=1)   # прогрев (numpy/кеши)
pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable(); b = C(S, flow, e.X0[i], back, tau, 3000, rho, gg, miss, K=3, tries=3, kn=8, neigh=0, nbest=1); pr.disable()
print('запрос старта %d: %.3f с (с профилировщиком), T=%s' % (i, time.perf_counter() - t0, None if b is None else round(b[0], 3)))
st = pstats.Stats(pr); st.sort_stats('tottime').print_stats(10); st.sort_stats('cumulative').print_stats(10)
