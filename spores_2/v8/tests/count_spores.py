"""клеток/узлов и время build/solve/агент для N спор на слой (env окна). python3 tests/count_spores.py N"""
import os, sys, time
os.environ['V8_MAXCELLS'] = sys.argv[1]; V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, V8)
from src.stepper import live
import numpy as np
from src.algo import growN as g
from src.stepper import vsurf as VS
cl = {}; t = time.perf_counter()
for k, u in enumerate((1, -1)): cl[u] = [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]]
tb = time.perf_counter() - t; t = time.perf_counter(); Vs, info = VS.solve_layers(cl, calc='q'); ts = time.perf_counter() - t; G = info.pop('G'); t = time.perf_counter(); res = VS.agent_paths(G, cl=cl); ta = time.perf_counter() - t
st = np.asarray(res['status'], object); w = (st != 'out_of_area') & ~res['left_work']; ok = st == 'reached'
print('N=%s cells %s nodes %d build %.1fs solve %.1fs (passes %d) agent %.1fs | work-zone reached %d/%d, T median %.3f' % (sys.argv[1], {u: len(c) for u, c in cl.items()}, G.n, tb, ts, info['passes'], ta, int((ok & w).sum()), int(w.sum()), float(np.median(res['T'][ok & w]))))
