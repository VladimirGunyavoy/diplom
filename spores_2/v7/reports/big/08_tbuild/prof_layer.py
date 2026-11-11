"""(3) разбивка t_build кара carfree: cProfile одного слоя (build_layer, MAXC=400) — доля growN (ряды потока/изгиб/грани), HexIdx.add/covered/query, Cell.build.
Запуск (aida, v7/src/cells7): env как tp.sh (SYS=car NODISC=1 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 PESS=1 GS=300 MAXC=400 DTN=.2 VF=1.5) python prof_layer.py"""
import os, sys, cProfile, pstats, time
sys.path.insert(0, os.path.expanduser('~/spore_v5/a31/spores_2/v7/src/cells7'))
import numpy as np, growCar as G
rng = np.random.default_rng(0); u = G.US[0]; t0 = time.time(); pr = cProfile.Profile(); pr.enable(); cells, idx = G.build_layer(u, rng, None); pr.disable()
print('клеток', len(cells), 'всего %.1f с' % (time.time() - t0)); st = pstats.Stats(pr); st.sort_stats('cumulative').print_stats(25)
