"""таблица + журнал решений: время агента до/после, decisions в res, склейка пачек. python3 tests/check_dlog.py"""
import os, sys, time
os.environ.setdefault('V8_MAXCELLS', '0'); V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, V8)
from src.stepper import live
import numpy as np
from src.algo import growN as g
from src.stepper import vsurf as VS
cl = {u: [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]] for k, u in enumerate(live.LAYERS)}
Vs, info = VS.solve_layers(cl, calc='q'); G = info.pop('G'); print('tab before:', G.tab is None)
t = time.perf_counter(); r0 = VS.agent_paths(G, cl=cl); t0 = time.perf_counter() - t
t = time.perf_counter(); G.build_table(); tt = time.perf_counter() - t
t = time.perf_counter(); r1 = VS.agent_paths(G, cl=cl); t1 = time.perf_counter() - t
t = time.perf_counter(); r2 = VS.agent_paths(G, cl=cl, batch=25); t2 = time.perf_counter() - t
d = r1['decisions']; print('agent no table %.2fs | table %.2fs | agent with table %.2fs batched %.2fs' % (t0, tt, t1, t2))
print('decisions', len(d), 'cols', r1['decisions_cols'], 'rules', {k: sum(1 for r in d if r[-1] == k) for k in ('commit', 'hold', 'look')}, 'traj idx range', min(r[0] for r in d), max(r[0] for r in d), 'batched same count', len(r2['decisions']) == len(d))
rt = lambda r: np.nanmedian(r['T'][r['status'] == 'reached'] / r['Tref'][r['status'] == 'reached'])
print('reached', int((r0['status'] == 'reached').sum()), '->', int((r1['status'] == 'reached').sum()), 'T/Tref med', round(rt(r0), 3), '->', round(rt(r1), 3))
