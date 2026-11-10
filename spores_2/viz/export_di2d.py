"""export_di2d.py — данные viz для двойного интегратора (фазовая плоскость (x, v)): эпизоды «споры» (v8, commit-агент по полному атласу M=5, SOLVE8_PLUS) и «эталон (bang-bang)» (ref_di_disk.T_ref + разбор переключения).
Запуск из папки v8 (aida, numpy/scipy; MAXC не задавать): `python3 ../viz/export_di2d.py [--seed 0] [--n 200] [--target 1.003] [--out ../viz/data/di2d.json]`. Код v8 не правится; выбирается старт с T/T_ref ближе всего к --target (T_ref ≥ 2.5)."""
import os, sys, json, argparse
import numpy as np
os.environ.setdefault('SOLVE8_PLUS', '1'); os.environ.setdefault('COMMIT_EVERY', '3')
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, default=0); p.add_argument('--n', type=int, default=200); p.add_argument('--target', type=float, default=1.003); p.add_argument('--out', default='../viz/data/di2d.json'); p.add_argument('--tmin', type=float, default=2.5); a = p.parse_args()
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], a.seed, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False); rho = float(G.rho)
X2 = S.starts(g, a.n, 1); Tr = R.T_ref(X2, rho); T, sw = G.run_agent(X2, 3 * Tr + 2, **S.POLICIES['commit']); r = T / Tr
ok = np.isfinite(r) & (Tr >= a.tmin); i = int(np.argmin(np.where(ok, np.abs(r - a.target), np.inf))); x0 = X2[i]
print('старт %d: x0 = (%.3f, %.3f), T %.3f, T_ref %.3f, T/T_ref %.4f; медиана по стартам %.4f' % (i, x0[0], x0[1], T[i], Tr[i], r[i], np.nanmedian(r)))
path = []; T1, _ = G.run_agent(x0[None], np.array([3 * Tr[i] + 2]), path=path, rec=2, **S.POLICIES['commit']); us = np.array(G.us, float)
Xs = [x0.copy()]; Us = []; cur0 = None
for aa, Xa, ca in path[1:]:
    if len(aa): Xs.append(np.array(Xa[0], float)); Us.append([float(us[ca[0]])])
if len(Us) < len(Xs): Us.append(Us[-1] if Us else [0.])
Xs = np.array(Xs); Us = np.array(Us)
TT, (s0, t1, _) = R.T_ref(x0, rho, return_details=True); dt = .02; ts = np.append(np.arange(0, TT, dt), TT); Xr = []; Ur = []
for t in ts:
    if t1 is None or t <= t1: u = s0; tt = t; x, v = x0
    else: u = -s0; tt = t - t1; x, v = R._flow(x0[0], x0[1], s0, t1)
    xx, vv = R._flow(x, v, u, tt); Xr.append([float(xx), float(vv)]); Ur.append([float(u)])
rd = lambda A: np.round(np.asarray(A, float), 5).tolist()
d = {'sys': 'di2d', 'title': 'Двойной интегратор, |u| ≤ 1: фазовая плоскость (x, v)', 'params': {'rho': rho, 'umax': [1.], 'unames': ['u'], 'view': [-4.7, -2.7, 3.7, 2.7], 'switch': True},
     'goal': {'c': [0., 0.], 'r': rho}, 'obstacles': [], 'episodes': [
    {'method': 'споры', 'start': i, 'dt': dt, 'X': rd(Xs), 'U': rd(Us), 'T': float(T[i]), 'T_ref': float(Tr[i]), 'ok': True, 'src': 'v8 solve8 commit, полный атлас M=5, seed %d' % a.seed},
    {'method': 'эталон (bang-bang)', 'start': i, 'dt': dt, 'X': rd(Xr), 'U': rd(Ur), 'T': float(TT), 'T_ref': float(TT), 'ok': True, 'src': 'ref_di_disk.T_ref'}]}
json.dump(d, open(a.out, 'w'), ensure_ascii=False, separators=(',', ':')); print('записано', a.out, len(Xs), len(Xr))
