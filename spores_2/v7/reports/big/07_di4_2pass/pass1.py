"""Двухпроходная ДИ 4D, проход 1: ГРУБЫЙ атлас (DTN .4 M 2 MAXC малый) → solve → поле u*: argmin и зазор (2-й минус 1-й J) в узлах → field.npz; тайминги build/solve/поле.
Согласие карты с базой — отдельно (agree.py: поиск ближайшего узла поля на пробах базы). Запуск из ~/spore_v5/b12/spores_2/v7: SEED OUT env; GPULOCK=1 SOLVEGPU=1 STGPU=1."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); import growN as G
t0 = time.time(); A = G.Atlas(seed=int(os.environ.get('SEED', 0))); tb = time.time() - t0; t1 = time.time(); A.solve(); ts = time.time() - t1; t2 = time.time(); A._init_qx(); US = list(G.US); P = A.P
ok = (A.V < G.BIG / 2) & ~A.goal; ix = np.flatnonzero(ok); Pk = P[ix]; arg = np.empty(len(ix), np.int8); gap = np.empty(len(ix), np.float32)
for a in range(0, len(ix), 50000):
    Pc = Pk[a:a + 50000]; J = np.sort(np.stack([np.minimum(A.tgoal(Pc, u), G.DTN + A.vstar(G.step(Pc, u))) for u in US], 1), 1) if False else np.stack([np.minimum(A.tgoal(Pc, u), G.DTN + A.vstar(G.step(Pc, u))) for u in US], 1)
    arg[a:a + 50000] = J.argmin(1); s = np.sort(J, 1); gap[a:a + 50000] = np.minimum(s[:, 1] - s[:, 0], 99.)
tf = time.time() - t2; np.savez(os.path.join(os.environ['OUT'], 'field.npz'), P=Pk, arg=arg, gap=gap)
r = dict(DTN=G.DTN, M=G.M, cells=[len(l) for l in A.layers], nodes=int(A.N), nodes_finite=int(len(ix)), t_build=tb, t_solve=ts, t_field=tf, t_total=tb + ts + tf, arg_share=(np.bincount(arg, minlength=len(US)) / len(arg)).tolist()); print(json.dumps(r), flush=True); json.dump(r, open(os.path.join(os.environ['OUT'], 'pass1.json'), 'w'), indent=1)
