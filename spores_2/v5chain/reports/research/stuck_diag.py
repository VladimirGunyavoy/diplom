"""research-14: где застревает агент (T = inf) — последние точки пути и V* вокруг."""
import numpy as np, os
import grow_cells2d as G
A = G.Atlas(seed=int(os.environ.get('SEED', 0))); A.solve(); Q, ref = G.starts_ref(); ok = ref > .05; Q = Q[ok]
T, sw, path = A.rollout(Q); bad = np.flatnonzero(~np.isfinite(T)); print('не дошли', len(bad), flush=True)
ends = np.array([path[-1, k] for k in bad]); ends[:, 0] = G.wrap(ends[:, 0])
print('концы путей (округл. .05):', sorted(set(map(tuple, (np.round(ends / .05) * .05).round(2).tolist())))[:20])
for k in bad[:4]:
    y = path[-1, k:k + 1]; print('  старт', Q[k].round(2), 'конец', y[0].round(3), 'V* тут %.2f' % A.vstar(y)[0], '| шаги:', [round(float(A.vstar(G.step(y, u))[0]), 2) for u in G.US],
          '| покрыт ядрами атласов:', [bool(ix.covered(y)[0]) for ix in A.idx])
