"""b15: ясновидец OCP (envs_dyn.make_oracle: диски — известные функции времени) на 20 стартах D.starts, режимы fixed и random(зерно 0) → data/env_oracle_<mode>.json. Запуск: python envs_oracle.py (NP=12; aida 2)"""
import os, sys, json, time, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, envs_dyn as D
def one(a):
    i, mode = a; X0 = D.starts(20); H = D.Hazards(mode, 0); rng = np.random.default_rng(i); t0 = time.perf_counter(); r = D.make_oracle()(X0[i], H, D.vias(X0[i], rng))
    return dict(i=i, mode=mode, T=None if r is None else float(r[0]), wall=time.perf_counter() - t0)
if __name__ == '__main__':
    import common
    with mp.get_context('fork').Pool(int(os.environ.get('NP', 12))) as p: res = p.map(one, [(i, m) for m in ('fixed', 'random') for i in range(20)], chunksize=1)
    json.dump(res, open(os.path.join(common.DATA, 'env_oracle.json'), 'w')); print({m: np.nanmedian([r['T'] if r['T'] else np.nan for r in res if r['mode'] == m]) for m in ('fixed', 'random')}, 'нет решения', sum(r['T'] is None for r in res))
