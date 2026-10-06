"""п.30: vstar по CSR-индексу == по старому (QFAST=0) на случайных точках. python3 neighbors_check.py A.pkl"""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/cells7'))
import grow_cells2d as G
A = pickle.load(open(sys.argv[1], 'rb')); rng = np.random.default_rng(5); Y = np.c_[rng.uniform(-G.XL, G.XL, 20000), rng.uniform(-G.WL, G.WL, 20000)]
G.QFAST = 1; v1 = A.vstar(Y); G.QFAST = 0; v0 = A.vstar(Y); print('vstar max |diff|', float(np.abs(v1 - v0).max()), 'inf-mismatch', int(((v1 >= G.BIG / 2) != (v0 >= G.BIG / 2)).sum()), 'n', len(Y), 'BIG pts', int((v0 >= G.BIG / 2).sum()))
