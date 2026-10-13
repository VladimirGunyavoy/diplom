"""viz (линия B): выгрузка машины (x, y, θ, v; u = (a, ω)) в viz/data/car.json (3 диска) и carfree.json (без препятствий).
Запуск на aida: cd ~/spore_v5/r21/v6 && CARL=6 CM=.002 python3 <путь>/viz/export_car.py car|carfree [--ids 0,3,7,...]   (нужны src.atlas6, car.py v6; ~1 мин на систему)
Метод «споры» — коридор v6 как в главной строке бенча (bench/spores_car.py, src = car_spores.json); траектория пересчитана общим проверяльщиком (common.Env.step) и снята на сетку .02 с.
carfree: те же старты, диски убраны (HZ — далеко за полем; препятствий нет, T_ref нет — берётся T споров)."""
import os, sys, json, time, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.abspath(os.path.join(HERE, '..', 'bench')); CWD = os.getcwd(); sys.path.insert(0, BENCH); sys.path.insert(0, HERE)
os.environ.setdefault('CARL', '6'); os.environ.setdefault('CM', '.002')
ap = argparse.ArgumentParser(); ap.add_argument('system'); ap.add_argument('--ids', default=''); ap.add_argument('--nb', type=int, default=30000); a = ap.parse_args()
import common, export_mobile as EM
sys.path.insert(0, CWD)
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
import car as C
FREE = a.system == 'carfree'
if FREE: C.HZ = np.array([[1e3, 1e3]])                     # диски убраны
NB, NF, K = a.nb, 7500, 5; tau, rho = .4, .15
g = lambda x: np.array([C.GR ** 2 - ((x[0] - C.GOAL[0]) ** 2 + (x[1] - C.GOAL[1]) ** 2)])
miss = lambda X: np.maximum(np.linalg.norm(np.asarray(X)[..., :2] - C.GOAL, axis=-1) - C.GR, 0)
ing = lambda P: np.linalg.norm(np.asarray(P)[..., :2] - C.GOAL, axis=-1) < C.GR
clear = lambda P: C.clearance(P) - float(os.environ['CM'])
rng = np.random.default_rng(0); seeds = []
for _ in range(32):
    r, an = C.GR * .95 * np.sqrt(rng.uniform()), rng.uniform(-np.pi, np.pi); seeds.append(np.array([C.GOAL[0] + r * np.cos(an), C.GOAL[1] + r * np.sin(an), rng.uniform(-np.pi, np.pi), rng.uniform(-.5, 2.)]))
fl = lambda P, s, t: C.flow(P, s, t, dt_max=.05)
S = SysN(fl, len(C.US), (C.XL, C.XL, np.pi, 1.5), ing, seeds, per=(0, 0, 2 * np.pi, 0), ok=lambda p: C.clearance(p) >= 0, blocked=lambda P: C.clearance(P) < 0)
e = common.make('car')
if FREE: e.HZ = np.array([[1e3, 1e3]])                    # проверяльщик тоже без дисков
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); print('обратное дерево %.1f с' % (time.perf_counter() - t0), flush=True)
ids = [int(k) for k in a.ids.split(',')] if a.ids else list(range(len(e.X0))); eps = []; DT = EM.DT
for i in ids:
    x = e.X0[i]; b = corridor_query(S, fl, x, back, tau, NF, rho, g, miss, K=K, clear=clear, tries=3, kn=8)
    if b is None: print(i, 'нет плана', flush=True); continue
    segs = [(C.US[s], float(d)) for s, d in zip(b[1], b[2])]; ep = EM.episode(e, i, 'споры', segs, float(e.Tref[i]) if not FREE else float('nan'), 'car_spores.json' if not FREE else 'car_spores.json (без дисков, тот же метод)')
    if FREE and ep['T'] is not None: ep['T_ref'] = ep['T']
    eps.append(ep); print(i, ep['T'], ep['T_ref'], ep['ok'], len(ep['X']), flush=True)
d = dict(sys=a.system, title='Машина: ускорение и руль' + (' — без препятствий (эталона нет: T/T_ref ≡ 1)' if FREE else ', три препятствия'), params=dict(car_len=.8, car_wid=.44, fov=[-5, 5, -5, 5], goal_r=float(C.GR)), goal=dict(type='disc', c=[float(C.GOAL[0]), float(C.GOAL[1])], r=float(C.GR)),
         obstacles=[] if FREE else [dict(type='disc', c=[float(h[0]), float(h[1])], r=float(C.HR)) for h in C.HZ], episodes=eps); d['params']['obstacles'] = d['obstacles']; d['params']['goal'] = d['goal']
out = os.path.join(HERE, 'data', a.system + '.json'); json.dump(d, open(out, 'w'), ensure_ascii=False, separators=(',', ':')); print(out, os.path.getsize(out) // 1024, 'КБ')
