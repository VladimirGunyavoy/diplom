"""Разбор аномалии: Mmax 17 при DTN .06/.24 — 0% дошли. Сравнение с Mmax 33/9: узлы цели, доля V<∞, клетки с узлами цели (m, слой), ребра."""
import os, sys, json; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
G = B.G; G.US = (-G.UM, G.UM); A = G.Atlas(); ts = B.solve(A); fin = A.V < G.BIG / 2
gc = [(ci, c.u, int(c.m), int(c.G.shape[0]), int(G.ingoal(c.G).sum())) for ci, c in enumerate(A.cells) if G.ingoal(c.G).any()]
out = dict(MCAP=os.environ.get('MCAP'), DTN=G.DTN, cells=len(A.cells), nodes=int(A.N), goal_nodes=int(A.goal.sum()), goal_cells=gc, V_finite_frac=float(fin.mean()), n_it=int(A.n_it), edges=int(A.edges))
Qs, ref = G.starts_ref(); V0 = A.vstar(Qs); out['starts_with_V_finite'] = int((V0 < G.BIG / 2).sum())
for l, lay in enumerate(A.layers): out['layer%d_cells' % l] = len(lay); out['layer%d_V_finite' % l] = float(np.mean(np.concatenate([(A.V[c.o:c.o + c.G.shape[0] * c.m] < G.BIG / 2) for c in lay])))
print(json.dumps(out), flush=True); json.dump(out, open(os.environ['BIGOUT'] + '/anomaly_mcap%s_dtn%s.json' % (os.environ.get('MCAP'), os.environ.get('DTN')), 'w'), indent=1)
