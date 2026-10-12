"""LivePlay без GUI: время step() на кадр при 300 агентах + O: один агент от точки (таблица Q). python3 tests/check_liveplay.py"""
import os, sys, time
os.environ.setdefault('V8_MAXCELLS', '0'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.stepper import live
import numpy as np
from src.algo import growN as g
from src.stepper import vsurf as VS
from src.stepper.liveplay import LivePlay
cl = {u: [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]] for k, u in enumerate(live.LAYERS)}
Vs, info = VS.solve_layers(cl, calc='q'); G = info.pop('G')
t = time.perf_counter(); r1 = VS.agent_paths(G, cl=cl, starts=np.array([[1.2, -0.7]])); print('O single agent, no table %.3fs' % (time.perf_counter() - t))
G.build_table(); t = time.perf_counter(); r1 = VS.agent_paths(G, cl=cl, starts=np.array([[1.2, -0.7]])); print('O single agent, table %.3fs, points %d, reached %s' % (time.perf_counter() - t, len(r1['paths'][0]), r1['status'][0]))
res = VS.agent_paths(G, cl=cl); L = LivePlay(); L.add(res); ts = []; nv = 0
while L.active and len(ts) < 3000:
    t = time.perf_counter(); o = L.step(1 / 30.); ts.append(time.perf_counter() - t)
    if o['tail']['floor'] is not None: nv = max(nv, len(o['tail']['floor']))
ts = np.array(ts) * 1e3; print('frames %d, step ms: mean %.2f p95 %.2f max %.2f, max tail verts %d, heads %d' % (len(ts), ts.mean(), np.percentile(ts, 95), ts.max(), nv, len(o['heads'][0])))
L.add(r1); o = L.step(1 / 30.); print('late agent joined, active', L.active)
