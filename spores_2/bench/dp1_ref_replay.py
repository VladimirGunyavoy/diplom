"""CA1 тест: эталон r23 dp1 (CasADi/IPOPT, N=160) → (u_k, dt_k) → единый проверяльщик бенча. aida: PYTHONPATH=~/spore_v5/r23/pylib NQ=20 WMP=3 G=1 python3 dp1_ref_replay.py [старты]"""
import os, sys, json, numpy as np
R23 = os.path.expanduser('~/spore_v5/r23/06_dp1_ref'); sys.path.insert(0, R23); sys.argv = ['ref.py', '160', '4']
import ref, common
e = common.make('dp1'); idx = range(20)
Q = ref.starts(); out = []
for q in idx:
    d = json.loads(open(f'{R23}/out_{q}.jsonl').readline()); k, g = d['k'], d['g']
    rng = np.random.default_rng(1000 * q + 10 * g + 3 * k[0] + k[1] + 4); x0 = Q[q]
    T0 = 2.5 + .7 * np.abs(np.r_[np.asarray(ref.C3) + 2 * np.pi * np.asarray(k) - x0[:2]]).max()
    st, T, U = ref.solve_one(x0, k, rng, T0); N = len(U)
    Tv, why = common.verify(e, e.X0[q], list(U), [T / N] * N)
    out.append(dict(q=q, T_ref=d['T'], T_solve=T, T_verify=Tv, why=why)); print(out[-1], flush=True)
json.dump(out, open(os.path.join(common.DATA, 'dp1_ref_replay.json'), 'w'), indent=1)
print('прошли', sum(o['T_verify'] is not None for o in out), '/', len(out))
