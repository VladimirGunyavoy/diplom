"""calc_worker шлёт 'Tp' по пачкам, затем 'T' с тем же итогом; склейка partial == final. python3 tests/check_tp_stream.py"""
import os, sys, subprocess, pickle
os.environ.setdefault('V8_MAXCELLS', '0'); V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, V8)
from src.stepper import live
import numpy as np
from src.algo import growN as g
from src.stepper import vsurf as VS
cl = {u: [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]] for k, u in enumerate(live.LAYERS)}
p = subprocess.Popen([sys.executable, os.path.join(V8, 'src', 'stepper', 'calc_worker.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=os.environ.copy())
pickle.dump(dict(items=[('T', 'q', cl)], pics=[]), p.stdin, protocol=4); p.stdin.close(); parts = []; fin = None; prog = []
while True:
    try: m = pickle.load(p.stdout)
    except EOFError: break
    if m[0] == 'Tp': parts.append(m[2])
    elif m[0] == 'T': fin = m[2]
    elif m[0] == 'prog': prog.append(m[2][:40])
    elif m[0] == 'err': print('ERR', m[2]); sys.exit(1)
mg = VS.merge_paths(parts); ok = all(np.array_equal(np.asarray(mg['T']), np.asarray(fin['T']), equal_nan=True) for _ in [0]) and all(np.array_equal(a, b) for a, b in zip(mg['paths'], fin['paths']))
g1 = VS.traj_geom(mg); g2 = VS.traj_geom(fin); g3 = None
cat = {k: np.concatenate([VS.traj_geom(q)[k] for q in parts]) for k in g2}
print('batches', len(parts), 'partial==final T/paths', ok, 'geom concat == final', all(np.array_equal(cat[k], g2[k]) for k in g2), '| prog sample', prog[:2], prog[-1])
