"""профиль stencils() ДИ 4D: подвыборка NS узлов, один u. Запуск из ~/spore_v5/a31/spores_2/v7."""
import os, sys, time, pickle, cProfile, pstats, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); G = __import__('growNj')
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d = pickle.load(open(os.environ['LOADL'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); t = time.time(); A._init_qx(); print('init_qx %.1f' % (time.time() - t), flush=True)
NS = int(os.environ.get('NS', 100000)); Y = G.step(A.P[::A.N // NS][:NS], G.US[0]); t = time.time()
pr = cProfile.Profile(); pr.enable(); a, b, c = A.stencils(Y); pr.disable(); print('stencils %d узлов -> %d пар %.1f с' % (len(Y), len(a), time.time() - t), flush=True)
pstats.Stats(pr).sort_stats('cumulative').print_stats(14)
