"""c7 (для viz/ANIM): траектории конкурентов бенча с выгрузкой в эпизоды viz/data/<система>.json.
python viz_comp.py ENV ALGO BUDGET START[,START..] LABEL OUTFILE [--procs 6]   ALGO: sst | rrt | krrt (kRRT*, дд/ДИ)
Повторяет run_sst.work (тот же seed 1000+i, тот же код sst_own/rrtstar) → путь (u_k, dt_k) → единый проверяльщик (RK4 h≤.01, как common.verify) → X через .02.
Недошедший (нет решения/проверка no_goal/box): траектория до точки остановки (путь к узлу, ближайшему к цели, либо до выхода из коробки), ok=false, T=None.
T может отличаться от таблицы (anytime по wall-часам, загрузка aida) — в эпизод пишется t_table (T из бенча на этом старте) и T_run."""
import os, sys, json, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import common, sst_own, rrtstar_di, rrtstar_dd
DT = .02; H = .01

def replay(e, y0, us, dts):
    y = np.asarray(y0, float)[None].copy(); X = [y[0].copy()]; U = []; t = 0.; ok = False; why = 'no_goal'; nxt = DT
    if e.in_goal(y)[0]: return X, U, 0., True
    for u, dt in zip(us, dts):
        u = e.clip_u(np.asarray(u, float).reshape(1, -1)); k = max(1, int(np.ceil(dt / H - 1e-9))); hh = dt / k
        for _ in range(k):
            y2 = e.step(y, u, hh)
            if e.out_of_box(y2)[0]: return X, U, None, False
            y = y2; t += hh
            while t >= nxt - 1e-9: X.append(y[0].copy()); U.append(u[0].copy()); nxt += DT
            if e.in_goal(y)[0]:
                if abs(X[-1] - y[0]).max() > 0 or len(X) == 0: pass
                X.append(y[0].copy()); U.append(u[0].copy()); return X, U, t, True
    return X, U, None, False

def work(a):
    env, algo, budget, i, seed_off = a; e = common.make(env); seed = 1000 + i + seed_off; y0 = e.X0[i]
    if algo == 'sst': snaps = sst_own.sst(e, y0, [budget], sel=.2, prune=.1, seed=seed)
    elif algo == 'rrt': snaps = sst_own.rrt(e, y0, [budget], seed=seed)
    else: snaps = (rrtstar_dd if env in ('dd', 'ddobst', 'ddobst7') else rrtstar_di).rrtstar(e, y0, [budget], seed=seed)
    s = snaps[-1]
    if s['has']: us, dts = s['us'], s['dts']
    elif s.get('near'): us, dts = s['near']
    else: return dict(i=i, empty=True)
    X, U, T, ok = replay(e, y0, us, dts)
    return dict(i=i, X=np.array(X).tolist(), U=np.array(U).tolist(), T=T, ok=ok, has=bool(s['has']), T_plan=s.get('T_plan'), tref=float(e.Tref[i]), t_wall=s['t'])

if __name__ == '__main__':
    env, algo, budget, starts, label, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), [int(x) for x in sys.argv[4].split(',')], sys.argv[5], sys.argv[6]
    procs = int(sys.argv[sys.argv.index('--procs') + 1]) if '--procs' in sys.argv else 6
    with Pool(procs) as p: R = p.map(work, [(env, algo, budget, i, 0) for i in starts])
    src = {'sst': f'{env}_sst.json', 'rrt': f'{env}_rrt.json', 'krrt': f'{env}_rrtstar.json'}[algo]
    tab = json.load(open(os.path.join(common.DATA, src))); eps = []
    for r in R:
        if r.get('empty'): print('нет пути', r['i'], flush=True); continue
        tt = [q for x in tab['raw'] if x['i'] == r['i'] for q in x['runs'] if abs(q['budget'] - budget) < 1e-9]
        eps.append(dict(method=label, start=r['i'], dt=DT, X=r['X'], U=r['U'], T=r['T'], T_ref=r['tref'], ok=bool(r['ok']), src=src, T_table=(tt[0].get('T') if tt else None), has_path=r['has'], t_wall=r['t_wall']))
        print(r['i'], 'ok' if r['ok'] else 'НЕ ДОШЁЛ', 'T', r['T'], 'T/ref', (r['T'] / r['tref'] if r['T'] else None), 'в таблице', eps[-1]['T_table'], 'точек', len(r['X']), flush=True)
    json.dump(eps, open(out, 'w'))
