"""Профиль картинок окна (aida, env окна): рост 3 спор -> V -> traj -> cProfile каждой картинки pics.*; md5 PNG для сверки до/после. python3 tests/prof_pics.py outdir"""
import os, sys, time, cProfile, pstats, hashlib, io
V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, V8)
os.environ.setdefault('V8_MAXCELLS', '3')
from src.stepper import live
import numpy as np
from src.algo import growN as g
from src.stepper import vsurf as VS
from src import pics as PX
T0 = time.perf_counter(); out = sys.argv[1]; os.makedirs(out, exist_ok=True); cl = {}
for k, u in enumerate(live.LAYERS): cl[u] = [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]]
t = time.perf_counter(); print({u: len(c) for u, c in cl.items()}, 'build %.1fs' % (t - T0)); Vs, info = VS.solve_layers(cl, calc='q'); G = info.pop('G'); print('solve %.1fs' % (time.perf_counter() - t), flush=True)
t = time.perf_counter(); res = VS.agent_paths(G, cl=cl); print('agents %.1fs' % (time.perf_counter() - t), flush=True)
for name, f in (('cells', lambda: PX.cells_png(G, cl, [out], title='(di)')), ('error', lambda: PX.error_map(G, [out], 'q')), ('traj', lambda: PX.traj_png(G, res, [out], 'q')), ('hist', lambda: PX.hist_png(G, res, [out], 'q'))):
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); f(); pr.disable(); print('== %s %.2fs' % (name, time.perf_counter() - t), flush=True)
    if time.perf_counter() - t > 1 or True:
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(7); print('\n'.join(l[:150] for l in s.getvalue().splitlines() if l.strip())[-900:], flush=True)
for fn in sorted(os.listdir(out)):
    if fn.endswith('.png'): print(fn, hashlib.md5(open(os.path.join(out, fn), 'rb').read()).hexdigest())
