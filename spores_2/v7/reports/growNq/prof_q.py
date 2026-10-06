import os, sys, cProfile, pstats, numpy as np
sys.path.insert(0, 'src/cells7'); import growNq as Q; G = Q.G
Qs, ref = G.starts_ref(); s = G.wrapy(Qs[1]); Q.MAXR = 4
pr = cProfile.Profile(); pr.enable(); A, Ts, C, info = Q.bidir(s); pr.disable()
print(info); pstats.Stats(pr).sort_stats('cumulative').print_stats(14)
