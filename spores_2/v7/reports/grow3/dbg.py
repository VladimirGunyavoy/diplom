"""debug: build atlas once, pickle cells; then diagnose why V does not propagate"""
import sys, os, pickle, time, numpy as np
sys.path.insert(0, 'src/cells7'); import grow3 as G
pk = 'reports/grow3/layers.pkl'
if os.path.exists(pk): L = pickle.load(open(pk, 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = L; A.idx = []; A.finish()
else: A = G.Atlas(); pickle.dump(A.layers, open(pk, 'wb'))
print('nodes', A.N, 'in goal', int(A.goal.sum()), flush=True)
Vg = np.full(A.N, np.inf)
for u in G.US: Vg = np.minimum(Vg, A.tgoal(A.P, u))
print('nodes reaching goal within DTN', int(np.isfinite(Vg).sum()), flush=True)
sub = np.random.default_rng(0).choice(A.N, 20000, replace=False)
for u in G.US:
    I, IDX, W = A.stencils(G.step(A.P[sub], u)); print(u, 'stencil hit frac', len(np.unique(I)) / len(sub), flush=True)
nl = [len(l) for l in A.layers]; print('cells', nl)
# distance of nearest node to goal center
d = np.linalg.norm(np.c_[A.P[:, :2], G.wrap(A.P[:, 2])], axis=1); print('min dist node to goal center', d.min(), 'nodes within .15:', int((d < .15).sum()), flush=True)
