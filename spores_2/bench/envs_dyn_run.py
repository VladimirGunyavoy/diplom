"""ENV2 базовая строка (b14): атлас carfree (V₀ без дисков) + агент (открытый план envs2026.plan, их клип v) → непрерывный проигрыш в динамической среде (envs_dyn.replay). Режимы: fixed (их φ_i) и random (5 зёрен φ/A). Метрики: доля эпизодов с касанием (pen > 1e-3), успех, T/T_ref(статик). Запуск: LOAD=~/spore_v5/car2/free.pkl VSAT=0 SYS=car python envs_dyn_run.py → data/env_dyn_base.json"""
import os, sys, json, pickle, multiprocessing as mp
os.environ.setdefault('SYS', 'car')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np, envs2026 as E, envs_dyn as D, growCar as G, common, __main__, inspect
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
X0 = D.starts(20); Tref = common.make('car').Tref   # статик-эталон — ориентир (НЕ оракул динамики)
def one(a):
    i, mode, seed = a; r = E.plan(G, A, X0[i])
    if r is None: return dict(i=i, mode=mode, seed=seed, why='no_plan')
    us, dts, Ta = r; H = D.Hazards(mode, seed); rp = D.replay(H, X0[i], us, dts); rp.update(i=i, mode=mode, seed=seed, T_agent=Ta); return rp
if __name__ == '__main__':
    jobs = [(i, 'fixed', 0) for i in range(20)] + [(i, 'random', s) for s in range(5) for i in range(20)]
    with mp.get_context('fork').Pool(int(os.environ.get('NP', 8))) as p: res = p.map(one, jobs)
    for mode in ('fixed', 'random'):
        R = [r for r in res if r['mode'] == mode]; ok = [r for r in R if r.get('ok')]; ta = [r for r in R if 'touch' in r]
        print(mode, 'эпизодов', len(R), 'дошли', len(ok), 'с касанием', sum(r['touch'] for r in ta), 'доля касаний %.2f' % (np.mean([r['touch'] for r in ta]) if ta else np.nan), 'доля дошли без касания %.2f' % (np.mean([bool(r.get('ok')) and not r['touch'] for r in R])), flush=True)
    json.dump(res, open(os.path.join(common.DATA, 'env_dyn_base.json'), 'w'), default=float)
