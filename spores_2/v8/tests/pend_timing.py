"""Тайминг pend по шагам (первый запуск: 1 спора от цели на слой u=±UM): рост по слоям, solve (проходы/раунды), таблица Q, траектории.
Лог — строка на шаг с секундами + картинка reports/pend_timing.png (горизонтальные полосы). Запуск: V8_MAXCELLS=1 python3 tests/pend_timing.py [N]"""
import os, sys, time, json
V8 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, V8)
for k, v in dict(SYS='pend', M='5', KF='11', NFAIL='400', DELTA='.2', RMAX='.3', TMAX='8', TQDM_MI='1000', GS='0', GLIM='0', GOALB='0', GOALSHAPE='ball', GSEED='1', RHO='.1', UM='.3', COVTOL='.003', OVL='.05', GROW='2', OVH='.5', FRAC='.5', DTN='.06', NORM='gram', SEL='stop', STOPS='0', SOLVE8_PLUS='1', V8_MAXCELLS='1', V8_ONLY_GOAL='1', REFINE_THR='0', BISECT_BOUND='0').items(): os.environ.setdefault(k, v)
os.environ['MAXC'] = os.environ['V8_MAXCELLS']
import numpy as np
from tqdm import tqdm
from src.algo import growN as g
from src.stepper import vsurf as VS
rows = []
def step(name, f):
    t0 = time.perf_counter(); r = f(); dt = time.perf_counter() - t0; rows.append((name, dt)); print('[time] %-28s %8.2f s' % (name, dt), flush=True); return r
layers = [u for u in g.US if u != 0]; cl = {}
for k, u in enumerate(layers):
    res = step('growth u=%+g' % u, lambda: g.build_layer(u, np.random.default_rng(k), seeds=None, only_queue=False)[0]); cl[u] = [g.cell_dict(c) for c in res]
    print('   cells', len(cl[u]), 'rows', [len(c['G']) for c in cl[u]], flush=True)
bar = tqdm(desc='solve passes', mininterval=5)
def prog(p, d, V): bar.n = p; bar.set_postfix(res='%.1e' % d); bar.refresh()
Vs, info = step('solve (q lin, PLUS)', lambda: VS.solve_layers(cl, calc='q', progress=prog)); bar.close(); G = info.pop('G')
print('   passes', info.get('passes'), 'rounds', getattr(G, 'nrounds', '?'), 'npts', getattr(G, 'npts', '?'), 'nodes', getattr(G, 'n', '?'), flush=True)
step('table Q (build_table)', lambda: G.build_table())
res = step('trajectories %d' % VS.TRAJ_N, lambda: VS.agent_paths(G, cl=cl, batch=VS.TRAJ_BATCH))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, .5 * len(rows) + 1)); names = [r[0] for r in rows][::-1]; ts = [r[1] for r in rows][::-1]
ax.barh(names, ts, color='#4c78a8'); [ax.text(t, i, ' %.1f s' % t, va='center') for i, t in enumerate(ts)]; ax.set_xlabel('s'); ax.set_title('pend, 1 спора на слой: время по шагам (всего %.1f s)' % sum(ts)); fig.tight_layout()
out = os.path.join(V8, 'reports', 'pend_timing.png'); fig.savefig(out, dpi=110); json.dump(rows, open(os.path.join(V8, 'reports', 'pend_timing.json'), 'w')); print('TOTAL %.1f s -> %s' % (sum(ts), out), flush=True)
