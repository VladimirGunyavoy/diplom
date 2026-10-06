import os, sys, time, pickle, cProfile, pstats, numpy as np
import fast_test as FT
G = FT.G; A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open('atlas_manip60.pkl', 'rb')); A.finish()
cnt = dict(pairs=0, aabb=0, it0=0)
def _t(s, Y, pi, vv):
    hid = vv // G.KSH; y = Y[pi] - G.SH[vv % G.KSH]; ok = ((y >= s.LO[hid]) & (y <= s.HI[hid])).all(1); cnt['pairs'] += len(pi); cnt['aabb'] += int(ok.sum())
    return FT._test_fast(s, Y, pi, vv)
G.HexIdx._test = _t; Y = G.step(A.P, G.US[0])
pr = cProfile.Profile(); pr.enable(); r = A.stencils(Y); pr.disable()
print(cnt, 'найдено', len(r[0]), 'точек', len(Y), 'KSH', G.KSH, 'QB', G.QB, 'ячеек в индексе', A.qx.n)
pstats.Stats(pr).sort_stats('tottime').print_stats(10)
