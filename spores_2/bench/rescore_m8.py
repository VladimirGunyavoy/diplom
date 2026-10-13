"""8D: эталона OCP нет в данных ⇒ T_ref = лучший найденный T среди всех методов на каждом старте (все T — единым проверяльщиком); пишет data/m8_ref.json и пересчитывает сводки в data/m8_*.json (reached, ratio_*). Отношения — к лучшему найденному, не к оптимуму."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
D = common.DATA; files = [f for f in sorted(glob.glob(D + '/m8_*.json')) if 'ref' not in f and 'smoke' not in f]; runs = {}          # имя → список T (None) по стартам
for f in files:
    d = json.load(open(f)); b = os.path.basename(f)
    if 'sst' in b: [runs.__setitem__((b, i), [r['runs'][i].get('T') if r['runs'][i]['has'] else None for r in sorted(d['raw'], key=lambda r: r['i'])]) for i in range(len(d['budgets']))]
    elif 'ppo' in b: runs[(b, None)] = d['T']
    else: runs[(b, None)] = [r.get('T_verify', r.get('T')) for r in d['per_start']]
n = 20; best = np.full(n, np.inf)
for T in runs.values():
    for i, t in enumerate(T):
        if t is not None and np.isfinite(t): best[i] = min(best[i], t)
ref = [None if not np.isfinite(t) else float(t) for t in best]; json.dump(ref, open(D + '/m8_ref.json', 'w')); print('лучший найденный T_ref:', np.round(best, 2))
os.environ['M8REF'] = '1'; import envs_6d; e = common.make('m8')
for f in files:
    d = json.load(open(f)); b = os.path.basename(f)
    def upd(s, T):
        s.update(common.summarize(e, T)); return s
    if 'sst' in b:
        for i, bd in enumerate(d['budgets']): upd(bd, runs[(b, i)])
    else: upd(d, runs[(b, None)])
    d['Tref'] = e.Tref.tolist(); d['Tref_note'] = 'лучший найденный среди методов (m8_ref.json)'; json.dump(d, open(f, 'w'), indent=1, default=float)
print('пересчитано', len(files), 'файлов')
