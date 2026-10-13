"""CA-MAZE 4D диагностика: LAYERS=pkl → solve (GPULOCK) → V в файл; доля конечных V, V у цели, V*(старты) и первый шаг агента. aida: SRC_DIR=~/spore_v5/c1/maze4_src SOLVEGPU=1 LAYERS=... python diag_maze4.py"""
import os, sys, pickle, numpy as np
sys.argv = ['x']
import spores_maze4 as SM
import growN as G, common
e = common.make('maze4'); A = G.Atlas.__new__(G.Atlas); LP = os.path.expanduser(os.environ['LAYERS']); VP = LP.replace('.pkl', '_V.pkl')
A.layers = pickle.load(open(LP, 'rb'))['layers']; A.finish()
if os.path.exists(VP): A.V = pickle.load(open(VP, 'rb')); A.n_it = 0; A.edges = 0
else: A.solve(); pickle.dump(A.V, open(VP, 'wb'))
fin = A.V < G.BIG / 2; P = G.wrapy(A.P); print('узлов', A.N, 'конечных V', fin.mean().round(4), flush=True)
r = np.linalg.norm(P[:, :2], axis=1); print('узлов в круге r<.3:', int((r < .3).sum()), 'конечных', int(fin[r < .3].sum()), '; min V', A.V.min(), flush=True)
for a, b in ((0, .3), (.3, 1), (1, 2), (2, 4), (4, 9)): m = (r >= a) & (r < b); print('r %.1f-%.1f узлов %d конечных %.3f' % (a, b, m.sum(), fin[m].mean() if m.any() else 0), flush=True)
vs = A.vstar(e.X0); print('V*(старты) конечных', int((vs < G.BIG / 2).sum()), 'из', len(vs), 'первые', np.round(vs[:8], 2), 'T_lb', np.round(e.Tref[:8], 2), flush=True)
Y = G.wrapy(e.X0[:3]); print('ingoal старт:', G.ingoal(Y), 'free:', G.free(Y), 'edge_ok:', [G.edge_ok(Y, u) for u in G.US], flush=True)
