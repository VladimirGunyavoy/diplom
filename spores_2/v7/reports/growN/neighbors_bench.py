"""PLAN п.30: стенд q_ms. build: строит жирный атлас (как в п.30), решает V, пикл A.pkl. run: время ОДНОГО запроса (rollout 1 старта), мед./p90, + T.
Запуск: [env жирных клеток] python3 neighbors_bench.py build OUT.pkl | python3 neighbors_bench.py run OUT.pkl [N] [LOOK]"""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/cells7'))
import grow_cells2d as G
if sys.argv[1] == 'build':
    t0 = time.time(); A = G.Atlas(); tb = time.time() - t0; A.solve(); print('cells', [len(l) for l in A.layers], 'nodes', A.N, 'build %.0f solve %.0f s' % (tb, time.time() - t0 - tb), flush=True)
    for a in ('qx',):
        if hasattr(A, a): delattr(A, a)
    pickle.dump(A, open(sys.argv[2], 'wb'), protocol=4)
else:
    A = pickle.load(open(sys.argv[2], 'rb')); n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    if len(sys.argv) > 4: G.LOOK = int(sys.argv[4])
    Q, ref = G.starts_ref(); ok = ref > .05; Q = Q[ok][:n]; ts = []; Tq = []
    for q in Q: t1 = time.time(); T, sw, _ = A.rollout(q[None]); ts.append(time.time() - t1); Tq.append(float(T[0]))
    ts = np.array(ts); print('LOOK', G.LOOK, 'q_ms med %.3f p90 %.3f s' % (np.median(ts), np.quantile(ts, .9)), 'T', np.round(Tq, 4).tolist(), flush=True)
