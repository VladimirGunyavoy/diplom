"""research-22, эксп. 08: профиль роста слоя (build) di4 — следующее узкое место после GPU-стенсилов и GPU-solve. Один слой, MAXC клеток, cProfile."""
import sys, os, time, cProfile, pstats, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
pr = cProfile.Profile(); t0 = time.time(); pr.enable(); L = G.build_layer(G.US[0], np.random.default_rng(0)); pr.disable(); cells = L[0] if isinstance(L, tuple) else L
print('build слоя: %.1f с, клеток %d, узлов %d' % (time.time() - t0, len(cells), sum(c.G.reshape(-1, G.N).shape[0] for c in cells)), flush=True)
pstats.Stats(pr).sort_stats('cumulative').print_stats(26); pstats.Stats(pr).sort_stats('tottime').print_stats(14)
