"""research-23: покрытие слоёв manip c3000 (b4, исправленная динамика) — доля случайных точек поля и стартов (20, поток rng как growN.starts_ref) в ядре клеток каждого слоя, V в стартах."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.expanduser('~/spore_v5/wb4/spores_2/v7/src/cells7'))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open(os.path.expanduser('~/spore_v5/wb4/manip_c3000_cor.pkl'), 'rb')); L = d_['layers']
rng = np.random.default_rng(0)
for _ in range(32): rng.uniform(-1, 1, 4)
Q = np.array([[*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)] for _ in range(20)])
r2 = np.random.default_rng(5); Y = np.c_[r2.uniform(-np.pi, np.pi, (100000, 2)), r2.uniform(-3, 3, (100000, 2))]; Y1 = Y[np.abs(Y[:, 2:]).max(1) <= 1]
for j, l in enumerate(L):
    idx = G.HexIdx()
    for c in l: idx.add(c)
    print('слой', j, G.US[j], 'клеток', len(l), '| покрытие поля |w|≤3: %.3f, |w|≤1: %.3f' % (idx.covered(Y).mean(), idx.covered(Y1).mean()), '| старты в ядре:', ''.join('x' if v else '.' for v in idx.covered(Q)), flush=True)
