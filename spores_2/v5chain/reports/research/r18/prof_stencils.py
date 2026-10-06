# research-18: cProfile стенсилов growN (manip 4D, малый атлас MAXC на слой) — что дорого: HexIdx.query, Ньютон обратной полилинейной карты, прочее
import os, sys, cProfile, pstats, time, numpy as np
sys.path.insert(0, os.path.expanduser('~/claude-work/projects/spore/spores_2/v7/src/cells7'))
import growN as G
A = G.Atlas(); print('клеток', [len(l) for l in A.layers], 'узлов', A.N, flush=True)
Y = G.step(A.P, G.US[0]); t0 = time.time()
pr = cProfile.Profile(); pr.enable(); r = A.stencils(Y); pr.disable()
print('стенсилы', len(Y), 'точек', round(time.time() - t0, 1), 'с, найдено', len(r[0]), flush=True)
pstats.Stats(pr).sort_stats('cumulative').print_stats(18)
