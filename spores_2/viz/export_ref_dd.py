"""viz (линия B): эталон-«призрак» дифдрайва — как эталон бенча (r17 refbox.py / refbox_obst.py): стрельба ≤ 5 дуг (вершины ромба u ∈ {(±1,0),(0,±1)}), SLSQP, min T в коробку цели; для ddobst — с зазором .05 до дисков.
Бенч хранит только T (dd_refbox_60.npy, dd_obst_refbox.npy); здесь пересчитан сам путь на тех же стартах (поиск в T ≤ 1.1·T_ref, лучшее найденное; T эталона в эпизоде — после общего проверяльщика, T_ref — из бенча).
Запуск: python3 viz/export_ref_dd.py dd|ddobst [--procs 5]; дописывает эпизоды «эталон (дуги)» в viz/data/<система>.json."""
import os, sys, json, itertools, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'bench')); sys.path.insert(0, HERE)
os.environ.setdefault('NA', '5'); os.environ.setdefault('BENCH_ENV', 'dd')
if len(sys.argv) > 1 and sys.argv[1].startswith('ddobst'): os.environ['OBST'] = '1'
sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
from scipy.optimize import minimize
from multiprocessing import Pool
import grow3 as G, common, export_mobile as EM
DISKS = np.array([[1.0, 0.3, .45], [-0.8, -0.9, .4]]); CL = .05; OBST = False
def arc(y, u, t):
    x, yy, th = y; v, w = u
    return np.array([x + v * t * np.cos(th), yy + v * t * np.sin(th), th]) if w == 0 else np.array([x, yy, th + w * t])
TOPS = [tp for n in range(1, 6) for tp in itertools.product(range(4), repeat=n) if all(tp[i] != tp[i + 1] for i in range(n - 1)) and all((G.US[tp[i]][1] == 0) != (G.US[tp[i + 1]][1] == 0) for i in range(n - 1))]
def one(a):
    y, tmax = a; best = (tmax, None, None); rng = np.random.default_rng(0)
    for tp in TOPS:
        def end(d):
            z = y.copy()
            for k, dt in zip(tp, d): z = arc(z, G.US[k], dt)
            return np.r_[z[:2], G.wrap(z[2])]
        def clear(d):
            if not OBST: return np.ones(1)
            z = y.copy(); out = []
            for k, dt in zip(tp, d):
                if G.US[k][1] == 0:
                    s_ = np.linspace(0, dt, 16); P = z[:2] + G.US[k][0] * s_[:, None] * np.array([np.cos(z[2]), np.sin(z[2])])
                    out.append((np.hypot(P[:, None, 0] - DISKS[:, 0], P[:, None, 1] - DISKS[:, 1]) - DISKS[:, 2] - CL).ravel())
                z = arc(z, G.US[k], dt)
            return np.concatenate(out) if out else np.ones(1)
        cons = [{"type": "ineq", "fun": lambda d: G.RHO * .999 - np.abs(end(d))}, {"type": "ineq", "fun": clear}]
        for _ in range(6):
            d0 = rng.uniform(0, 1, len(tp)); d0 *= rng.uniform(.5, 1.) * tmax / d0.sum()
            r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * len(tp), constraints=cons, options=dict(maxiter=200, ftol=1e-10))
            if r.success and np.all(np.abs(end(r.x)) <= G.RHO + 1e-9) and clear(r.x).min() >= -1e-9 and r.x.sum() < best[0]: best = (float(r.x.sum()), tp, r.x.copy())
    return best
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('system'); ap.add_argument('--procs', type=int, default=5); a = ap.parse_args(); OBST = a.system == 'ddobst'
    p = os.path.join(HERE, 'data', a.system + '.json'); d = json.load(open(p)); e = common.make({'dd': 'dd', 'ddobst': 'ddobst'}[a.system]); starts = sorted({ep['start'] for ep in d['episodes']})
    jobs = [(G.wrapy(e.X0[i]), 1.1 * float(e.Tref[i])) for i in starts]
    with Pool(min(a.procs, len(jobs))) as pool: R = pool.map(one, jobs)
    d['episodes'] = [ep for ep in d['episodes'] if 'эталон' not in ep['method']]
    for i, (T, tp, dd) in zip(starts, R):
        if tp is None: print(a.system, i, 'эталон не найден', flush=True); continue
        ep = EM.episode(e, i, 'эталон (дуги)', [(G.US[k], float(t)) for k, t in zip(tp, dd)], e.Tref[i], 'dd_refbox_60.npy' if a.system == 'dd' else 'dd_obst_refbox.npy'); d['episodes'].append(ep); print(a.system, i, 'T', ep['T'], 'T_ref', ep['T_ref'], 'дуг', len(tp), flush=True)
    EM.finalize(d); json.dump(d, open(p, 'w'), ensure_ascii=False, separators=(',', ':')); print(p, os.path.getsize(p) // 1024, 'КБ')
