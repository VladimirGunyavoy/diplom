"""b3: профиль solve pend: тот же env, что bench_pend.py"""
import os, sys, time, cProfile, pstats
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S
g = S.load_growN(); atlas, tg = S.build_atlas(list(g.US), 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q')
pr = cProfile.Profile(); pr.enable(); t0 = time.perf_counter(); G.solve(verbose=False); ts = time.perf_counter() - t0; pr.disable()
print('RES solve %.1f с, узлов %d, проходов %d' % (ts, G.n, G.passes)); pstats.Stats(pr).sort_stats('cumtime').print_stats(18)
