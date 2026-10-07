"""research-22, эксп. 06: профиль построения стенсилов (после GPU-solve это узкое место 4D). di4 L400: finish + stencils для одного u, cProfile."""
import sys, os, time, pickle, cProfile, pstats, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; t0 = time.time(); A.finish(); print('finish %.1f с, узлов %d, клеток %d' % (time.time() - t0, A.N, len(A.cells)), flush=True)
Y = G.step(A.P, G.US[0]); pr = cProfile.Profile(); t0 = time.time(); pr.enable(); a, b, c = A.stencils(Y); pr.disable(); print('stencils одного u: %.1f с, пар %d (%.2f на узел)' % (time.time() - t0, len(a), len(a) / A.N), flush=True)
pstats.Stats(pr).sort_stats('cumulative').print_stats(22); pstats.Stats(pr).sort_stats('tottime').print_stats(14)
