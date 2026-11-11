"""b15: OCP БЕЗ дисков на 20 стартах статик-кара (common.make('car').X0) — эталон остатка «качество атласа»: D.HR=0 снимает ограничения дисков. → data/env_oracle_nodisc.json"""
import os, sys, json, time, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, envs_dyn as D; D.HR = 0.
def one(i):
    import common; X0 = common.make('car').X0; r = D.make_oracle()(X0[i], D.Hazards('fixed', 0), D.vias(X0[i], np.random.default_rng(i))); return dict(i=i, T=None if r is None else float(r[0]))
if __name__ == '__main__':
    import common
    with mp.get_context('fork').Pool(12) as p: res = p.map(one, range(20), chunksize=1)
    json.dump(res, open(os.path.join(common.DATA, 'env_oracle_nodisc.json'), 'w')); print(np.median([r['T'] for r in res if r['T']]), sum(r['T'] is None for r in res))
