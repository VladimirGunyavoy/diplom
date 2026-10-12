"""Путь окна без GUI (aida): рост слоёв ±1 на 3 спорах (как live.py) -> calc_worker (V, T, картинки) -> vsurf.build_mesh. Env как окно: SOLVE8_PLUS=1, V8_ONLY_GOAL=1, MAXC=3. python3 tests/check_window_path.py [picsdir]"""
import os, sys, subprocess, pickle, time
V8 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, V8)
from src.stepper import live     # те же env-умолчания, что у окна (ursina не нужен)
import numpy as np
from src.algo import growN as g
print('env', {k: os.environ.get(k) for k in ('SOLVE8_PLUS', 'V8_ONLY_GOAL', 'MAXC', 'M')}, flush=True)
cl = {}
for k, u in enumerate((1, -1)): cl[u] = [g.cell_dict(c) for c in g.build_layer(u, np.random.default_rng(k))[0]]; print('layer', u, len(cl[u]), 'cells', flush=True)
pics = [sys.argv[1]] if len(sys.argv) > 1 else []
p = subprocess.Popen([sys.executable, os.path.join(V8, 'src', 'stepper', 'calc_worker.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=os.environ.copy())
pickle.dump(dict(items=[('V', 'q', cl), ('T', 'q', cl)], pics=pics), p.stdin, protocol=4); p.stdin.close(); Vres = None
while True:
    try: m = pickle.load(p.stdout)
    except EOFError: break
    if m[0] == 'prog': print('prog', m[1], m[2][:90], '%.1fs' % m[3], flush=True)
    elif m[0] == 'V': Vres = m[2]; info = m[2][1]; print('V done: passes %s converged %s finite %.0f%% t %.2fs total %.1fs' % (info['passes'], info.get('converged'), 100 * info['finite'], info['t'], m[3]), flush=True)
    elif m[0] == 'T': r = m[2]; print('T done: reached', int((np.asarray(r['status'], object) == 'reached').sum()), '/', len(r['status']), '%.1fs' % m[3], flush=True)
    elif m[0] == 'pics': print('pics', m[2], '%.1fs' % m[3], flush=True)
    elif m[0] == 'err': print('ERR', m[2]); sys.exit(1)
print('worker rc', p.wait())
from src.stepper import vsurf as VS
Vs, info = Vres; vmax = VS.vscale(Vs)
for u in cl: m = VS.build_mesh(cl[u], Vs[u], (0, 1), vmax, win=info['wins'][u], goalcols=info['goalcols'][u]); print('mesh', u, 'verts', len(m['verts']), 'tris', len(m['tris']))
print('OK')
