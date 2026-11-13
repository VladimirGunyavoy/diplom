import cProfile, pstats, sys, os, time, numpy as np
import growNa as G
t=time.time(); pr=cProfile.Profile(); pr.enable()
cells = G.build_layer(G.US[0], np.random.default_rng(0), None)[0]
pr.disable(); print('cells', len(cells), 'wall %.1f' % (time.time()-t), flush=True)
pstats.Stats(pr).sort_stats('cumulative').print_stats(22)
