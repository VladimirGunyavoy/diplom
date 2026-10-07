"""research-23: где конечна V в атласе manip (A_*.pkl): по слоям, max/квантили V, расстояние конечных узлов до цели."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
for p in sys.argv[1:]:
    d_ = pickle.load(open(os.path.expanduser(p), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); V = d_['V']; fin = V < G.BIG / 2
    lay = np.concatenate([np.full(c.G.shape[0] * G.M ** G.m_, j) for j, l in enumerate(A.layers) for c in l]); gd = G.goal_dist(A.P)
    print(os.path.basename(p), 'узлов', A.N, 'конечных', int(fin.sum()), 'в цели', int(A.goal.sum()), '| по слоям', [int((fin & (lay == j)).sum()) for j in range(4)],
          '| V кв. 50/90/max', np.round(np.percentile(V[fin], [50, 90, 100]), 2).tolist(), '| goal_dist конечных кв. 50/90/max', np.round(np.percentile(gd[fin], [50, 90, 100]), 2).tolist(), flush=True)
